from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converter_adapters import convert_hf_model_weights_with_adapters


@dataclass(frozen=True)
class ModelCase:
    family: str
    source_file: str
    model_id: str | None
    loader: str | None
    supported: bool
    note: str = ""


MODEL_CASES: tuple[ModelCase, ...] = (
    ModelCase("gemma", "model_gemma.cpp", "google/gemma-2b-it", "causal_lm", True),
    ModelCase("gemma3n", "model_gemma3n.cpp", "google/gemma-3n-E2B-it", None, False, "adapter not implemented"),
    ModelCase("gemma4", "gemma4/model_gemma4.cpp", "google/gemma-4-E2B", None, False, "adapter not implemented"),
    ModelCase("lfm2", "model_lfm2.cpp", "LiquidAI/LFM2-350M", None, False, "adapter not implemented"),
    ModelCase("lfm2moe", "model_lfm2moe.cpp", "LiquidAI/LFM2-1.2B", None, False, "adapter not implemented"),
    ModelCase("lfm2vl", "model_lfm2vl.cpp", "LiquidAI/LFM2-VL-450M", None, False, "adapter not implemented"),
    ModelCase("moonshine", "model_moonshine.cpp", "UsefulSensors/moonshine-base", None, False, "adapter not implemented"),
    ModelCase("needle", "model_needle.cpp", None, None, False, "custom checkpoint path required"),
    ModelCase("nomic", "model_nomic.cpp", "nomic-ai/nomic-bert-2048", None, False, "adapter not implemented"),
    ModelCase("parakeet", "model_parakeet.cpp", "nvidia/parakeet-ctc-1.1b", None, False, "adapter not implemented"),
    ModelCase("parakeet_tdt", "model_parakeet_tdt.cpp", "nvidia/parakeet-tdt-0.6b-v3", None, False, "adapter not implemented"),
    ModelCase("pyannote", "model_pyannote.cpp", "pyannote/segmentation-3.0", None, False, "adapter not implemented"),
    ModelCase("qwen", "model_qwen.cpp", "Qwen/Qwen2.5-0.5B-Instruct", "causal_lm", True),
    ModelCase("qwen3p5", "model_qwen3p5.cpp", "Qwen/Qwen3.5-2B", "causal_lm", True),
    ModelCase("siglip2", "model_siglip2.cpp", "google/siglip2-base-patch16-224", None, False, "adapter not implemented"),
    ModelCase("silero_vad", "model_silero_vad.cpp", "snakers4/silero-vad", None, False, "adapter not implemented"),
    ModelCase("wespeaker", "model_wespeaker.cpp", "pyannote/wespeaker-voxceleb-resnet34-LM", None, False, "adapter not implemented"),
    ModelCase("whisper", "model_whisper.cpp", "openai/whisper-small", "auto_model", True),
    ModelCase("youtu", "model_youtu.cpp", None, None, False, "no stable public model id configured"),
)


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-") or "model"


def _artifacts_root() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "adapter_conversion_matrix"


def _resolve_local_snapshot(model_id: str) -> str | None:
    explicit = Path(model_id)
    if explicit.exists():
        return str(explicit)

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + model_id.replace("/", "--"))
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return str(snapshots[-1])


def _ensure_snapshot(model_id: str) -> str:
    snapshot = _resolve_local_snapshot(model_id)
    if snapshot is not None:
        return snapshot

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id)


def _load_model(snapshot: str, loader: str):
    if loader == "causal_lm":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            snapshot,
            torch_dtype="auto",
            device_map=None,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        ).eval()

    if loader == "auto_model":
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            snapshot,
            torch_dtype="auto",
            device_map=None,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        ).eval()

    raise ValueError(f"unsupported loader={loader}")


def run_matrix(*, precision: str, families: set[str] | None = None) -> list[dict[str, Any]]:
    artifacts_root = _artifacts_root()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for case in MODEL_CASES:
        if families is not None and case.family not in families:
            continue

        entry: dict[str, Any] = {
            "family": case.family,
            "source_file": case.source_file,
            "model_id": case.model_id,
            "supported": case.supported,
            "note": case.note,
        }

        if not case.supported:
            entry["status"] = "skipped"
            results.append(entry)
            print(f"[SKIP] family={case.family} reason={case.note}")
            continue

        assert case.model_id is not None
        assert case.loader is not None

        slug = _slug(case.model_id)
        out_dir = artifacts_root / slug
        try:
            print(f"[DOWNLOAD] family={case.family} model={case.model_id}")
            snapshot = _ensure_snapshot(case.model_id)
            print(f"[LOAD] family={case.family} snapshot={snapshot}")
            model = _load_model(snapshot, case.loader)
            print(f"[CONVERT] family={case.family} output={out_dir}")
            config = convert_hf_model_weights_with_adapters(model, out_dir, precision=precision)
            del model
            entry["status"] = "ok"
            entry["snapshot"] = snapshot
            entry["output_dir"] = str(out_dir)
            entry["config"] = config
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            entry["traceback"] = traceback.format_exc()
            print(f"[ERROR] family={case.family} error={exc}")
        results.append(entry)

    summary_path = artifacts_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run adapter-based model conversion matrix.")
    parser.add_argument("--precision", choices=["INT4", "INT8", "FP16"], default="INT4")
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help="Restrict to one or more model families (repeatable).",
    )
    args = parser.parse_args(argv)

    families = set(args.families) if args.families else None
    results = run_matrix(precision=args.precision, families=families)
    failures = [item for item in results if item["status"] == "error"]
    supported = [item for item in results if item["supported"]]
    succeeded = [item for item in supported if item["status"] == "ok"]

    print(
        f"[SUMMARY] supported={len(supported)} converted={len(succeeded)} "
        f"errors={len(failures)} skipped={len(results) - len(supported)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
