"""Compatibility wrapper for the packaged Cactus CLI."""

from cactus.cli import *  # noqa: F401,F403
from cactus.cli import main


if __name__ == "__main__":
    main()
