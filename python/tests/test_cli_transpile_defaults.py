from __future__ import annotations

from pathlib import Path

from cactus import cli
from cactus.cli import transpile as transpile_cli


def _fake_completed_process(returncode: int = 0):
    class _Result:
        def __init__(self, code: int):
            self.returncode = code

    return _Result(returncode)


def test_cmd_transpile_requires_converted_weights_by_default(monkeypatch) -> None:
    parser = cli.create_parser()
    args = cli.preprocess_eval_args(parser, ["transpile", "gemma4"])

    captured: list[tuple[list[str], Path, dict[str, str]]] = []

    def _unexpected_runtime_build():
        raise AssertionError("runtime build should not run before CQ weights are available")

    monkeypatch.setattr(transpile_cli, "_ensure_python_runtime_library", _unexpected_runtime_build)
    monkeypatch.setattr(transpile_cli, "get_weights_dir", lambda model_id: Path("/tmp/missing-weights"))

    def _fake_run(command, cwd=None, env=None):
        captured.append((list(command), Path(cwd), dict(env or {})))
        return _fake_completed_process(0)

    monkeypatch.setattr(transpile_cli.subprocess, "run", _fake_run)

    rc = transpile_cli.cmd_transpile(args)

    assert rc == 1
    assert not captured


def test_cmd_transpile_allows_unconverted_weights_for_debug(monkeypatch) -> None:
    parser = cli.create_parser()
    args = cli.preprocess_eval_args(parser, ["transpile", "gemma4", "--allow-unconverted-weights"])

    captured: list[tuple[list[str], Path, dict[str, str]]] = []

    monkeypatch.setattr(transpile_cli, "_ensure_python_runtime_library", lambda: Path("/tmp/libcactus.dylib"))
    monkeypatch.setattr(transpile_cli, "get_weights_dir", lambda model_id: Path("/tmp/missing-weights"))

    def _fake_run(command, cwd=None, env=None):
        captured.append((list(command), Path(cwd), dict(env or {})))
        return _fake_completed_process(0)

    monkeypatch.setattr(transpile_cli.subprocess, "run", _fake_run)

    rc = transpile_cli.cmd_transpile(args)

    assert rc == 0
    assert captured
    command, cwd, env = captured[0]
    assert command[:5] == [
        transpile_cli.sys.executable,
        "-m",
        "cactus.transpile.hf_model",
        "--model-id",
        transpile_cli.DEFAULT_MODEL_ID,
    ]
    assert "--allow-unconverted-weights" in command
    assert "--skip-execute" in command
    assert cwd == transpile_cli.PROJECT_ROOT
    assert env["CACTUS_LIB_PATH"] == "/tmp/libcactus.dylib"


def test_cmd_transpile_can_execute_immediately_when_requested(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    args = cli.preprocess_eval_args(
        parser,
        ["transpile", "gemma4", "--execute-after-transpile", "--artifact-dir", "/tmp/gemma4-bundle"],
    )

    captured: list[list[str]] = []

    monkeypatch.setattr(transpile_cli, "_ensure_python_runtime_library", lambda: Path("/tmp/libcactus.dylib"))
    ready_weights_dir = tmp_path / "weights" / "gemma-4-e2b-it"
    ready_weights_dir.mkdir(parents=True)
    (ready_weights_dir / "weights_manifest.json").write_text("{}")

    monkeypatch.setattr(transpile_cli, "get_weights_dir", lambda model_id: ready_weights_dir)

    def _fake_run(command, cwd=None, env=None):
        captured.append(list(command))
        return _fake_completed_process(0)

    monkeypatch.setattr(transpile_cli.subprocess, "run", _fake_run)

    rc = transpile_cli.cmd_transpile(args)

    assert rc == 0
    assert captured
    command = captured[0]
    assert "--skip-execute" not in command
    assert "--artifact-dir" in command
    assert "/tmp/gemma4-bundle" in command


def test_cmd_transpile_rejects_empty_default_weights_dir(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    args = cli.preprocess_eval_args(parser, ["transpile", "whisper-small"])

    empty_weights_dir = tmp_path / "weights" / "whisper-small"
    empty_weights_dir.mkdir(parents=True)

    captured: list[list[str]] = []

    monkeypatch.setattr(transpile_cli, "_ensure_python_runtime_library", lambda: Path("/tmp/libcactus.dylib"))
    monkeypatch.setattr(transpile_cli, "get_weights_dir", lambda model_id: empty_weights_dir)

    def _fake_run(command, cwd=None, env=None):
        captured.append(list(command))
        return _fake_completed_process(0)

    monkeypatch.setattr(transpile_cli.subprocess, "run", _fake_run)

    rc = transpile_cli.cmd_transpile(args)

    assert rc == 1
    assert not captured


def test_cmd_transpile_uses_ready_default_weights_dir(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    args = cli.preprocess_eval_args(parser, ["transpile", "parakeet"])

    ready_weights_dir = tmp_path / "weights" / "parakeet-tdt-0.6b-v3"
    ready_weights_dir.mkdir(parents=True)
    (ready_weights_dir / "weights_manifest.json").write_text("{}")

    captured: list[list[str]] = []

    monkeypatch.setattr(transpile_cli, "_ensure_python_runtime_library", lambda: Path("/tmp/libcactus.dylib"))
    monkeypatch.setattr(transpile_cli, "get_weights_dir", lambda model_id: ready_weights_dir)

    def _fake_run(command, cwd=None, env=None):
        captured.append(list(command))
        return _fake_completed_process(0)

    monkeypatch.setattr(transpile_cli.subprocess, "run", _fake_run)

    rc = transpile_cli.cmd_transpile(args)

    assert rc == 0
    assert captured
    command = captured[0]
    assert "--weights-dir" in command
    assert str(ready_weights_dir) in command


def test_lfm_alias_points_to_vl_450m() -> None:
    assert transpile_cli.MODEL_ID_ALIASES["lfm"] == "LiquidAI/LFM2-VL-450M"
