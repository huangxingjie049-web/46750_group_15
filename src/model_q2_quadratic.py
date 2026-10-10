"""Compatibility entry; implementation is in src.2c.model."""
from importlib import import_module
import sys
_implementation = import_module("src.2c.model")
sys.modules[__name__] = _implementation
