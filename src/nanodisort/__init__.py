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

import itertools

from nanodisort._core import DisortState as _DisortState

from ._version import _version as __version__


class DisortState(_DisortState):
    """
    DISORT solver state.
    """

    def __init__(self) -> None:
        super().__init__()

    def print_state(self, pad: int | None = 0) -> None:
        dimensions = [
            "nstr",
            "nlyr",
            "nmom",
            "ntau",
            "numu",
            "nphi",
        ]
        flags = [
            "usrtau",
            "usrang",
            "lamber",
            "planck",
            "onlyfl",
            "quiet",
            "intensity_correction",
            "spher",
        ]
        boundary_conditions = [
            "fbeam",
            "umu0",
            "phi0",
            "fisot",
            "fluor",
            "albedo",
            "btemp",
            "ttemp",
            "temis",
        ]
        others_scalar = ["accur", "wvnmlo", "wvnmhi"]
        others_array = [
            "dtauc",
            "ssalb",
            "umu",
            "phi",
            "utau",
            "temper",
        ]

        if pad is None:
            all_fields = itertools.chain(
                dimensions, flags, boundary_conditions, others_scalar, others_array
            )
            pad = max(map(len, all_fields)) + 1

        sections = []

        for title, section in [
            ("Flags", flags),
            ("Dimensions", dimensions),
            ("Boundary conditions", boundary_conditions),
            ("Others", others_scalar),
        ]:
            values = "\n".join(
                [f"  {field:<{pad}} = {getattr(self, field)}," for field in section]
            )
            sections.append("\n".join([f"  # {title}", values]))

        if self.allocated:
            sections.extend(
                [
                    f"  {field:<{pad}} = {getattr(self, field)},"
                    for field in others_array
                ]
            )

        body = "\n".join([f"  {'allocated':<{pad}} = {self.allocated},"] + sections)
        print("\n".join(["DisortState[", body, "]"]))


__all__ = ["__version__", "DisortState"]
