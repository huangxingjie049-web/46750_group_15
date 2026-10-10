"""Compatibility entry; implementation is in src.3d.validation."""
from importlib import import_module
import sys
_implementation = import_module("src.3d.validation")
sys.modules[__name__] = _implementation
