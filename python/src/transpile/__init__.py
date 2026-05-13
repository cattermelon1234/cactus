"""Expose the packaged transpiler through the historical ``src`` namespace."""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parents[2] / "cactus" / "transpile"),
]
