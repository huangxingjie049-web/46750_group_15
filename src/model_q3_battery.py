"""Compatibility entry; implementation is in src.3g.model."""
from importlib import import_module
import sys
_implementation = import_module("src.3g.model")
sys.modules[__name__] = _implementation
