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

from __future__ import annotations

import itertools

from nanodisort._core import DisortState as _DisortState

from ._version import _version as __version__


class DisortState(_DisortState):
    """
    DISORT solver state.

    This class provides an interface to the CDISORT radiative transfer model.
    CDISORT is a C port of the original DISORT Fortran package. The main
    differences are:

    * **Improved memory management.** CDISORT allocates memory dynamically to
      use memory buffers of the right size, and internally encapsulates data
      in structs. This overall improves performance compared to the Fortran
      implementation.
    * **Double precision.** CDISORT operates entirely with double-precision
      arithmetics for improved numerical stability.
    * **Correction of intensity fields** by Buras-Emde algorithm, included by
      Robert Buras.
    * **Pseudospherical geometry** for direct beam source, included by Arve
      Kylling.
    * **Solution for a general source term**, included by Arve Kylling.
    """

    def __init__(self) -> None:
        """
        Initialize the DISORT state. All parameters are initialized with their
        default values.
        """
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
