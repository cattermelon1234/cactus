from __future__ import annotations

from pathlib import Path

from cactus import cli
from cactus.cli import run as run_cli
from cactus.cli import transpile as transpile_cli


def test_cactus_run_detects_transpiled_bundle_and_uses_main_style_audio(monkeypatch, tmp_path: Path, capsys) -> None:
    bundle_dir = tmp_path / "bundle"
    components_dir = bundle_dir / "components"
    components_dir.mkdir(parents=True)
    (components_dir / "manifest.json").write_text(
        '{"model_id":"example/model","family":"generic","task":"causal_lm_logits","components":[]}',
        encoding="utf-8",
    )
    audio_file = tmp_path / "input.wav"
    audio_file.write_bytes(b"RIFF")

    calls = []

    def _fake_run_transpiled(args):
        calls.append(args)
        print("hello from transpiled")
        return 0

    monkeypatch.setattr(transpile_cli, "cmd_run_transpiled", _fake_run_transpiled)

    parser = cli.create_parser()
    args = parser.parse_args(
        [
            "run",
            str(bundle_dir),
            "--audio",
            str(audio_file),
            "--prompt",
            "Hello",
        ]
    )

    rc = run_cli.cmd_run(args)

    assert rc == 0
    assert len(calls) == 1
    forwarded = calls[0]
    assert forwarded.bundle_dir == str(bundle_dir)
    assert forwarded.audio == str(audio_file.resolve())
    assert forwarded.audio_file == str(audio_file.resolve())
    assert forwarded.prompt == "Hello"
    assert forwarded._transpiled_from_run is True
    captured = capsys.readouterr().out
    assert "Starting Cactus Chat with model:" in captured
    assert "hello from transpiled" in captured


def test_run_transpiled_human_result_prints_response(capsys) -> None:
    transpile_cli._print_transpiled_run_result(
        {
            "response": "  generated text  ",
            "transcript": "not used",
        }
    )

    assert capsys.readouterr().out == "generated text\n"
