"""Compatibility entry; implementation is in src.3g.runner."""
from importlib import import_module
import sys
_implementation = import_module("src.3g.runner")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
