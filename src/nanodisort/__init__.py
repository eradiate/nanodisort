"""
nanodisort - Python bindings for cdisort radiative transfer solver.

This package provides Python bindings to cdisort, a C implementation of the
DISORT (Discrete Ordinates Radiative Transfer) program for solving the
radiative transfer equation.

Examples
--------
>>> import nanodisort as nd
>>> # Create solver state
>>> state = nd.DisortState()
>>> # Configure dimensions
>>> state.nstr = 8
>>> state.nlyr = 6
>>> # ... (more configuration)
>>> state.allocate()
>>> state.solve()
"""

from nanodisort._core import DisortState

__version__ = "0.1.0"
__all__ = ["DisortState"]
