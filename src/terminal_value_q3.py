"""Compatibility entry; implementation is in src.3g.terminal_value."""
from importlib import import_module
import sys
_implementation = import_module("src.3g.terminal_value")
sys.modules[__name__] = _implementation
