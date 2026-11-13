"""Allow ``python -m comfy_detective ...`` without installing the package."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
