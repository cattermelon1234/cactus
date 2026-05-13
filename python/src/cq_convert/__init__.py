"""Expose the packaged CQ converter through the historical ``src`` namespace."""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parents[2] / "cactus" / "convert"),
]
