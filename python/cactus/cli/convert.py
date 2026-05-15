from __future__ import annotations

import argparse

from .common import (
    GREEN,
    PROJECT_ROOT,
    RED,
    YELLOW,
    get_weights_dir,
    print_color,
)
from .transpile import cmd_transpile, resolve_model_id_alias
from cactus.transpile.component_plan import infer_component_plan_from_output


_DEFAULT_MULTIMODAL_PROMPT = (
    "Respond with 2 lines. The first should be a description of the image, "
    "and the second should be a transcription of the audio"
)


def _default_multimodal_asset_args() -> tuple[list[str], str | None]:
    assets_dir = PROJECT_ROOT / "cactus-engine" / "tests" / "assets"
    image_file = assets_dir / "test_monkey.png"
    audio_file = assets_dir / "test.wav"
    image_args = [str(image_file)] if image_file.exists() else []
    audio_arg = str(audio_file) if audio_file.exists() else None
    return image_args, audio_arg


def _default_audio_asset_arg() -> str | None:
    _, audio_file = _default_multimodal_asset_args()
    return audio_file


def cmd_convert(args):
    """Convert a HuggingFace model to CQ format and transpile it in place."""
    model_id = resolve_model_id_alias(args.model_name)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(get_weights_dir(model_id))

    bits = getattr(args, "bits", 4) or 4
    token = getattr(args, "token", None)
    cache_dir = getattr(args, "cache_dir", None)

    try:
        from ..convert.cli import main as cq_main

        cq_args = [
            "convert",
            "--model",
            model_id,
            "--out",
            str(output_dir),
            "--bits",
            str(bits),
        ]
        if token:
            cq_args.extend(["--token", token])
        if cache_dir:
            cq_args.extend(["--cache-dir", cache_dir])
        cq_args.append("--force")

        cq_main(cq_args)

        task = getattr(args, "task", "auto") or "auto"
        prompt = getattr(args, "prompt", None) or _DEFAULT_MULTIMODAL_PROMPT
        image_files = [str(path) for path in (getattr(args, "image_file", None) or []) if str(path).strip()]
        audio_file = getattr(args, "audio_file", None)
        component_pipeline = getattr(args, "component_pipeline", "auto") or "auto"
        components = getattr(args, "components", None)

        plan = infer_component_plan_from_output(str(output_dir), model_id=model_id)

        if task == "auto":
            task = plan.task if plan is not None else "auto"

        if task == "multimodal_causal_lm_logits":
            needs_image = bool(plan.needs_image) if plan is not None else bool(image_files)
            needs_audio = bool(plan.needs_audio) if plan is not None else bool(audio_file)
            if (needs_image and not image_files) or (needs_audio and not audio_file):
                default_images, default_audio = _default_multimodal_asset_args()
                if needs_image and not image_files:
                    image_files = default_images
                if needs_audio and not audio_file:
                    audio_file = default_audio
                print_color(
                    YELLOW,
                    "Multimodal transpile needs representative media shapes; "
                    "using bundled tiny test assets.",
                )
            if needs_image and not image_files:
                print_color(
                    RED,
                    "Multimodal transpile requires --image-file for this model.",
                )
                return 1
            if needs_audio and not audio_file:
                print_color(
                    RED,
                    "Multimodal transpile requires --audio-file for this model.",
                )
                return 1
            if component_pipeline == "auto" and (plan is None or plan.force_component_pipeline):
                component_pipeline = "on"
            if components is None and plan is not None and plan.components:
                components = ",".join(plan.components)

        if task in {"tdt_transcription", "seq2seq_transcription", "ctc_logits"} and not audio_file:
            audio_file = _default_audio_asset_arg()
            if audio_file:
                print_color(
                    YELLOW,
                    f"{task} transpile needs a representative audio shape; "
                    "using bundled tiny test audio asset.",
                )
            else:
                print_color(RED, f"{task} transpile requires --audio-file.")
                return 1

        extra_args = [
            "--weights-dir",
            str(output_dir),
            "--artifact-dir",
            str(output_dir),
            "--task",
            task,
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(getattr(args, "max_new_tokens", 32) or 32),
            "--component-pipeline",
            component_pipeline,
        ]
        if components:
            extra_args.extend(["--components", str(components)])
        for image_file in image_files:
            extra_args.extend(["--image-file", image_file])
        if audio_file:
            extra_args.extend(["--audio-file", str(audio_file)])
        if getattr(args, "system_prompt", None):
            extra_args.extend(["--system-prompt", str(args.system_prompt)])
        if token:
            extra_args.extend(["--token", token])
        if getattr(args, "trust_remote_code", False) or task == "multimodal_causal_lm_logits":
            extra_args.append("--trust-remote-code")
        if getattr(args, "local_files_only", False):
            extra_args.append("--local-files-only")

        transpile_args = argparse.Namespace(
            model_id=model_id,
            execute_after_transpile=False,
            allow_unconverted_weights=False,
            extra_args=extra_args,
        )
        rc = cmd_transpile(transpile_args)
        if rc != 0:
            return rc

        print_color(GREEN, f"Model converted and transpiled to {output_dir}")
        return 0
    except SystemExit as e:
        return e.code if e.code else 0
    except Exception as e:
        print_color(RED, f"Conversion error: {e}")
        return 1
