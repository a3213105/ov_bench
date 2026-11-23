"""``python -m ov_bench`` entry point for the Python ov_bench backend."""

import sys

from .engine import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))
