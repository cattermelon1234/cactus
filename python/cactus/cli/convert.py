from __future__ import annotations

import argparse

from .common import (
    GREEN,
    RED,
    get_weights_dir,
    print_color,
)
from .transpile import cmd_transpile, resolve_model_id_alias


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

        transpile_args = argparse.Namespace(
            model_id=model_id,
            execute_after_transpile=False,
            allow_unconverted_weights=False,
            extra_args=[
                "--weights-dir",
                str(output_dir),
                "--artifact-dir",
                str(output_dir),
            ],
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
