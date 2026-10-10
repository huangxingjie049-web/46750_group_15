"""Compatibility entry; implementation is in src.2c.runner."""
from importlib import import_module
import sys
_implementation = import_module("src.2c.runner")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
