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

Citation
--------

If you use nanodisort in your research, please cite the cdisort paper :cite:p:`Buras2011CDISORTCorrection`.

Contents
--------

.. toctree::
   :maxdepth: 2

   api
   references

License
-------

nanodisort is licensed under the GNU General Public License v3.0 or later
(GPL-3.0-or-later), consistent with the cdisort library it wraps.
