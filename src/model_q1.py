"""Compatibility entry; implementation is in src.1e.model."""
from importlib import import_module
import sys
_implementation = import_module("src.1e.model")
sys.modules[__name__] = _implementation
