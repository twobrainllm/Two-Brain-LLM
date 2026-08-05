"""Enables `python -m two_brain_router`."""
from __future__ import annotations

import sys

from two_brain_router.cli import main

if __name__ == "__main__":
    sys.exit(main())
