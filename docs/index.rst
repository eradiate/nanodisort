nanodisort documentation
========================

**nanodisort** provides Python bindings for cdisort, a C implementation of the
DISORT (Discrete Ordinates Radiative Transfer) program for solving the
radiative transfer equation.

Overview
--------

DISORT is a widely-used solver for radiative transfer in plane-parallel
atmospheres. It uses the discrete ordinates method to solve the radiative
transfer equation, handling:

- Multiple scattering
- Absorption and emission
- Various boundary conditions
- Thermal emission (Planck function)
- Pseudo-spherical geometry for the direct beam

The cdisort library is a C translation of the original Fortran DISORT code,
offering improved performance through dynamic memory allocation and full
double precision arithmetic.

Installation
------------

From source:

.. code-block:: bash

    git clone https://github.com/rayference/nanodisort.git
    cd nanodisort
    uv sync --extra dev
    uv pip install -e .

Quick Start
-----------

.. code-block:: python

    import nanodisort as nd

    # Create solver state
    state = nd.DisortState()

    # Configure dimensions
    state.nstr = 8   # Number of streams
    state.nlyr = 6   # Number of layers
    state.nmom = 8   # Number of phase function moments
    state.ntau = 5   # Number of output optical depths
    state.numu = 4   # Number of output polar angles
    state.nphi = 3   # Number of output azimuthal angles

    # Allocate memory
    state.allocate()

    # Set optical properties, geometry, boundary conditions
    # ... (see API documentation)

    # Solve
    state.solve()

Why another Python DISORT wrapper?
----------------------------------

This project fills a specific niche that was not addressed by existing Python-centric DISORT implementations or wrappers. Hereafter follows a non-exhaustive review of existing solutions and how nanodisort relates to them.

Fortran and C DISORT implementations
    This Fortran implementation :cite:p:`Stamnes1988DISORT,Stamnes2000DISORTReport` has issues that were solved later by CDISORT. The CDISORT paper :cite:p:`Buras2011CDISORTCorrection` motivates the CDISORT project and explains why we decided to wrap CDISORT rather than DISORT.

`pydisort <https://github.com/zoeyzyhu/pydisort>`__
    This project wraps CDISORT as well, with `pybind11 <https://github.com/pybind/pybind11>`__. Its goals differ from this project's in that nandisort has more modest ambitions: we do not aim to expose a specific C++ interface to CDISORT, and we do not seek integration with ML frameworks.

`Pythonic-DISORT <https://github.com/LDEO-CREW/Pythonic-DISORT>`__
    This project is very different from nanodisort: it is a reimplementation of the original DISORT algorithm in Python. It uses Numpy and vectorizes many performance-critical operations, which makes it a serious alternative to consider for people who want to use the original DISORT in Python. It also provides a comprehensive introduction to DISORT.

`pyDISORT <https://github.com/mjwolff/pyDISORT>`__
    *TBD*

`pyRT_DISORT <https://github.com/mjwolff/pyRT_DISORT>`__
    *TBD*

Knowing that, the following major goals we set and choices we made for nanodisort are:

* A **thin Python wrapper around CDISORT**: there is no intermediate C++ API.
* An API that is **as close to the CDISORT API as possible**: no hidden operations, only error handling has been modified to allow raising exceptions instead of terminating execution.
* Full **interoperability with Numpy buffers**: all input data arrays can be assigned Numpy data.
* **Lightweight**: no required dependency on large computational frameworks (e.g. PyTorch, JAX or TensorFlow).

Citation
--------

If you use nanodisort in your research, please cite the cdisort paper :cite:p:`Buras2011CDISORTCorrection` and the nanodisort repository.

License
-------

nanodisort is licensed under the GNU General Public License v3.0 or later
(GPL-3.0-or-later), consistent with the cdisort library it wraps.

.. toctree::
    :maxdepth: 2
    :hidden:

    api/index
    references
