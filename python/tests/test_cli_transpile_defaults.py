from __future__ import annotations

from pathlib import Path

from src import cli


def _fake_completed_process(returncode: int = 0):
    class _Result:
        def __init__(self, code: int):
            self.returncode = code

    return _Result(returncode)


def test_cmd_transpile_defaults_to_skip_execute(monkeypatch) -> None:
    parser = cli.create_parser()
    args = cli.preprocess_eval_args(parser, ["transpile", "gemma4"])

    captured: list[tuple[list[str], Path, dict[str, str]]] = []

    monkeypatch.setattr(cli, "_ensure_python_runtime_library", lambda: Path("/tmp/libcactus.dylib"))
    monkeypatch.setattr(cli, "get_weights_dir", lambda model_id: Path("/tmp/missing-weights"))

    def _fake_run(command, cwd=None, env=None):
        captured.append((list(command), Path(cwd), dict(env or {})))
        return _fake_completed_process(0)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    rc = cli.cmd_transpile(args)

    assert rc == 0
    assert captured
    command, cwd, env = captured[0]
    assert command[:4] == [
        cli.sys.executable,
        str(cli.PROJECT_ROOT / "python" / "examples" / "transpile_hf_model.py"),
        "--model-id",
        cli.DEFAULT_MODEL_ID,
    ]
    assert "--skip-execute" in command
    assert cwd == cli.PROJECT_ROOT
    assert env["CACTUS_LIB_PATH"] == "/tmp/libcactus.dylib"


def test_cmd_transpile_can_execute_immediately_when_requested(monkeypatch) -> None:
    parser = cli.create_parser()
    args = cli.preprocess_eval_args(
        parser,
        ["transpile", "gemma4", "--execute-after-transpile", "--artifact-dir", "/tmp/gemma4-bundle"],
    )

    captured: list[list[str]] = []

    monkeypatch.setattr(cli, "_ensure_python_runtime_library", lambda: Path("/tmp/libcactus.dylib"))
    monkeypatch.setattr(cli, "get_weights_dir", lambda model_id: Path("/tmp/missing-weights"))

    def _fake_run(command, cwd=None, env=None):
        captured.append(list(command))
        return _fake_completed_process(0)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    rc = cli.cmd_transpile(args)

    assert rc == 0
    assert captured
    command = captured[0]
    assert "--skip-execute" not in command
    assert "--artifact-dir" in command
    assert "/tmp/gemma4-bundle" in command
