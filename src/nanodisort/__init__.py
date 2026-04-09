# SPDX-FileCopyrightText: 2025 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

import numpy as np
from numpy.typing import NDArray

from nanodisort._core import BatchSolver as _BatchSolver
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

    # Type annotations for properties inherited from C++ extension
    allocated: bool
    nstr: int
    nlyr: int
    nmom: int
    ntau: int
    numu: int
    nphi: int
    usrtau: bool
    usrang: bool
    lamber: bool
    planck: bool
    onlyfl: bool
    quiet: bool
    intensity_correction: bool
    spher: bool
    fbeam: float
    umu0: float
    phi0: float
    fisot: float
    fluor: float
    albedo: float
    btemp: float
    ttemp: float
    temis: float
    accur: float
    wvnmlo: float
    wvnmhi: float
    dtauc: NDArray[np.float64]
    ssalb: NDArray[np.float64]
    pmom: NDArray[np.float64]
    umu: NDArray[np.float64]
    phi: NDArray[np.float64]
    utau: NDArray[np.float64]
    temper: NDArray[np.float64]
    rfldir: NDArray[np.float64]
    rfldn: NDArray[np.float64]
    flup: NDArray[np.float64]
    dfdt: NDArray[np.float64]
    uavg: NDArray[np.float64]
    uavgdn: NDArray[np.float64]
    uavgup: NDArray[np.float64]
    uavgso: NDArray[np.float64]
    uu: NDArray[np.float64]
    u0u: NDArray[np.float64]
    uum: NDArray[np.float64]
    albmed: NDArray[np.float64]
    trnmed: NDArray[np.float64]
    old_intensity_correction: bool

    def __init__(self) -> None:
        """
        Initialize the DISORT state. All parameters are initialized with their
        default values.
        """
        super().__init__()

    def print_state(self, pad: int | None = 0) -> None:
        """
        Print to the terminal the full state of the DISORT solver.

        Parameters
        ----------
        pad : int or None, default: 0
            Minimal amount of padding space between variable name and first "="
            sign. Set to ``None`` to automatically align "=" signs. By default,
            no padding is applied.
        """
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


class BatchSolver(_BatchSolver):
    """
    Parallel batch DISORT solver.

    Solves multiple independent spectral problems in parallel using a C++
    thread pool. Shared geometry and control flags are configured once on
    the solver instance; spectrally-varying optical properties (optical
    depths, single-scattering albedos, phase function moments, beam
    intensities, surface albedos) are provided as arrays with a leading
    batch dimension.

    Thread safety requires ``quiet=True`` and ``lamber=True``. These are
    enforced at allocation time.

    Parameters
    ----------
    nthreads : int, optional
        Number of worker threads. Defaults to the hardware concurrency.

    Examples
    --------
    >>> import nanodisort as nd
    >>> import numpy as np
    >>> solver = nd.BatchSolver(nthreads=4)
    >>> solver.nstr = 8
    >>> solver.nlyr = 1
    >>> solver.nmom = 8
    >>> solver.ntau = 2
    >>> solver.usrtau = True
    >>> solver.usrang = False
    >>> solver.lamber = True
    >>> solver.onlyfl = True
    >>> solver.quiet = True
    >>> solver.umu0 = 1.0
    >>> solver.phi0 = 0.0
    >>> solver.set_utau(np.array([0.0, 1.0]))
    >>> nbatch = 10
    >>> solver.allocate(nbatch)
    >>> solver.set_dtauc(np.ones((nbatch, 1)))
    >>> solver.set_ssalb(np.full((nbatch, 1), 0.9))
    >>> pmom = np.zeros((9, 1, nbatch))
    >>> pmom[0, :, :] = 1.0
    >>> solver.set_pmom(pmom)
    >>> solver.set_fbeam(np.full(nbatch, np.pi))
    >>> solver.set_albedo(np.zeros(nbatch))
    >>> solver.solve()
    >>> rfldir = solver.rfldir  # shape (nbatch, 2)
    """

    # Type annotations for properties
    nbatch: int
    nthreads: int
    allocated: bool
    solved: bool
    nstr: int
    nlyr: int
    nmom: int
    ntau: int
    numu: int
    nphi: int
    usrtau: bool
    usrang: bool
    lamber: bool
    planck: bool
    onlyfl: bool
    quiet: bool
    intensity_correction: bool
    old_intensity_correction: bool
    spher: bool
    umu0: float
    phi0: float
    fisot: float
    fluor: float
    btemp: float
    ttemp: float
    temis: float
    accur: float
    wvnmlo: float
    wvnmhi: float
    rfldir: NDArray[np.float64]
    rfldn: NDArray[np.float64]
    flup: NDArray[np.float64]
    dfdt: NDArray[np.float64]
    uavg: NDArray[np.float64]
    uavgdn: NDArray[np.float64]
    uavgup: NDArray[np.float64]
    uavgso: NDArray[np.float64]
    uu: NDArray[np.float64]

    def __init__(self, nthreads: int = 0) -> None:
        """
        Initialize the batch solver.

        Parameters
        ----------
        nthreads : int, optional
            Number of worker threads. 0 (default) uses hardware concurrency.
        """
        super().__init__(nthreads)


__all__ = ["__version__", "BatchSolver", "DisortState"]
