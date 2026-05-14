from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .common import (
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    GREEN,
    RED,
    YELLOW,
    print_color,
)
from .download import get_weights_dir


MODEL_ID_ALIASES = {
    "gemma4": DEFAULT_MODEL_ID,
    "gemma4-e2b": DEFAULT_MODEL_ID,
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "parakeet-tdt": "nvidia/parakeet-tdt-0.6b-v3",
    "whisper": "openai/whisper-small",
    "qwen": "Qwen/Qwen3-1.7B",
    "lfm": "LiquidAI/LFM2-VL-450M",
}


def resolve_model_id_alias(model_id: str) -> str:
    normalized = (model_id or "").strip()
    return MODEL_ID_ALIASES.get(normalized.lower(), normalized)


def _python_runtime_library_path() -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return PROJECT_ROOT / "cactus" / "build" / f"libcactus{suffix}"


def _static_cactus_library_path() -> Path:
    return PROJECT_ROOT / "cactus" / "build" / "libcactus.a"


def _build_static_cactus_library() -> Path:
    build_script = PROJECT_ROOT / "cactus" / "build.sh"
    if not build_script.exists():
        raise RuntimeError(f"The Cactus build script is missing: {build_script}")

    build = subprocess.run([str(build_script)], cwd=PROJECT_ROOT / "cactus")
    if build.returncode != 0:
        raise RuntimeError("Failed to build the Cactus static runtime")

    static_library_path = _static_cactus_library_path()
    if not static_library_path.exists():
        raise RuntimeError(
            "The Cactus build completed, but the static library was not produced.\n"
            f"Expected: {static_library_path}"
        )
    return static_library_path


def _public_cactus_api_symbols(static_library_path: Path) -> list[str]:
    if platform.system() == "Darwin":
        command = ["nm", "-gU", str(static_library_path)]
    else:
        command = ["nm", "-g", "--defined-only", str(static_library_path)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to inspect the Cactus static runtime symbols.\n"
            f"Command: {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )

    symbols: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        symbol = parts[-1].strip()
        normalized = symbol[1:] if symbol.startswith("_") else symbol
        if normalized.startswith("cactus_") and symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise RuntimeError(f"Could not find any public cactus_* symbols in {static_library_path}")
    return symbols


def _link_python_runtime_library(*, static_library_path: Path, library_path: Path) -> None:
    build_dir = library_path.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    if library_path.exists():
        library_path.unlink()

    exported_symbols = _public_cactus_api_symbols(static_library_path)
    if platform.system() == "Darwin":
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            raise RuntimeError("Failed to find a C++ compiler for linking libcactus.dylib")
        command = [
            compiler,
            "-dynamiclib",
            "-o",
            str(library_path),
            *[f"-Wl,-u,{symbol}" for symbol in exported_symbols],
            str(static_library_path),
            "-Wl,-install_name,@rpath/libcactus.dylib",
            "-lcurl",
            "-framework",
            "Accelerate",
            "-framework",
            "CoreML",
            "-framework",
            "Foundation",
            "-framework",
            "Security",
            "-framework",
            "SystemConfiguration",
            "-framework",
            "CFNetwork",
        ]
    else:
        compiler = shutil.which("g++") or shutil.which("c++")
        if not compiler:
            raise RuntimeError("Failed to find a C++ compiler for linking libcactus.so")
        command = [
            compiler,
            "-shared",
            "-o",
            str(library_path),
            *[f"-Wl,--undefined={symbol}" for symbol in exported_symbols],
            str(static_library_path),
            "-lcurl",
            "-pthread",
            "-ldl",
            "-lm",
        ]

    result = subprocess.run(command, cwd=build_dir)
    if result.returncode != 0 or not library_path.exists():
        raise RuntimeError(f"Failed to link the Cactus shared runtime: {library_path}")


def _ensure_python_runtime_library() -> Path:
    library_path = _python_runtime_library_path()
    static_library_path = _static_cactus_library_path()
    if (
        library_path.exists()
        and static_library_path.exists()
        and library_path.stat().st_mtime >= static_library_path.stat().st_mtime
    ):
        return library_path

    print_color(YELLOW, "Preparing Cactus shared runtime for transpiler...")
    if not static_library_path.exists():
        static_library_path = _build_static_cactus_library()
    _link_python_runtime_library(static_library_path=static_library_path, library_path=library_path)
    return library_path


def _weights_dir_looks_transpile_ready(weights_dir: Path) -> bool:
    root = Path(weights_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return False
    if (root / "weights_manifest.json").exists():
        return True
    return any(root.glob("*.weights"))


def _resolve_transpiled_manifest(path_value: str | os.PathLike[str] | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value).expanduser().resolve()
    if not candidate.exists():
        return None
    if candidate.is_file() and candidate.name == "manifest.json":
        return candidate
    for manifest in (candidate / "components" / "manifest.json", candidate / "manifest.json"):
        if manifest.exists():
            return manifest
    return None


def _prepend_python_path(env: dict[str, str]) -> None:
    python_root = str(PROJECT_ROOT / "python")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = python_root if not existing else f"{python_root}{os.pathsep}{existing}"


def cmd_transpile(args) -> int:
    """Invoke the saved-model transpiler entrypoint."""
    try:
        transpile_lib = _ensure_python_runtime_library()
    except RuntimeError as exc:
        print_color(RED, f"Error: {exc}")
        return 1

    model_id = resolve_model_id_alias(args.model_id)
    extra_args = list(getattr(args, "extra_args", []) or [])
    command = [sys.executable, "-m", "cactus.transpile.hf_model", "--model-id", model_id]
    if "--weights-dir" not in extra_args:
        default_weights_dir = get_weights_dir(model_id)
        if _weights_dir_looks_transpile_ready(default_weights_dir):
            command.extend(["--weights-dir", str(default_weights_dir)])
    if not getattr(args, "execute_after_transpile", False) and "--skip-execute" not in extra_args:
        command.append("--skip-execute")
    command.extend(extra_args)

    env = os.environ.copy()
    env["CACTUS_LIB_PATH"] = str(transpile_lib)
    _prepend_python_path(env)
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return result.returncode


def cmd_run_transpiled(args) -> int:
    """Run a saved transpiled component bundle."""
    try:
        transpile_lib = _ensure_python_runtime_library()
    except RuntimeError as exc:
        print_color(RED, f"Error: {exc}")
        return 1

    os.environ["CACTUS_LIB_PATH"] = str(transpile_lib)
    python_root = str(PROJECT_ROOT / "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)

    from cactus.transpile.component_bundle_runtime import run_transpiled_bundle

    bundle_dir = getattr(args, "bundle_dir", None) or getattr(args, "model_id", None)
    image_values: list[str] = []
    for attr_name in ("image_file", "image_files"):
        value = getattr(args, attr_name, None)
        if isinstance(value, str) and value.strip():
            image_values.append(value.strip())
        elif isinstance(value, (list, tuple)):
            image_values.extend(str(item).strip() for item in value if str(item).strip())
    image_arg = getattr(args, "image", None)
    if isinstance(image_arg, str) and image_arg.strip():
        image_values.append(image_arg.strip())

    result = run_transpiled_bundle(
        bundle_dir,
        audio_file=getattr(args, "audio_file", None) or getattr(args, "audio", None),
        image_files=tuple(image_values),
        prompt=getattr(args, "prompt", None),
        input_ids=getattr(args, "input_ids", None),
        weights_dir=getattr(args, "weights_dir", None),
        system_prompt=getattr(args, "system", None),
        enable_thinking=bool(getattr(args, "thinking", False)),
        max_new_tokens=getattr(args, "max_new_tokens", None),
        stop_sequences=tuple(getattr(args, "stop_sequence", []) or ()),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if getattr(args, "result_json", None):
        result_path = Path(args.result_json).expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print_color(GREEN, f"Saved result to {result_path}")
    return 0
