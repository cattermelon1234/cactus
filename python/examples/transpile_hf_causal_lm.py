from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    if "--task" not in sys.argv[1:]:
        sys.argv[1:1] = ["--task", "causal_lm_logits"]
    script_path = Path(__file__).with_name("transpile_hf_model.py")
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
