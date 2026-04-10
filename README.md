# nanodisort

[![PyPI version](https://img.shields.io/pypi/v/nanodisort?color=blue)](https://pypi.org/project/nanodisort)
[![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/eradiate/nanodisort/ci.yml?branch=main)](https://github.com/eradiate/nanodisort/actions/workflows/ci.yml)
[![Documentation Status](https://img.shields.io/readthedocs/nanodisort)](https://nanodisort.readthedocs.io)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Python bindings for [CDISORT](http://www.libradtran.org/doku.php?id=cdisort), a C implementation of the DISORT (Discrete Ordinates Radiative Transfer) solver.

## Overview

DISORT is a widely-used radiative transfer solver for plane-parallel atmospheres. It uses the discrete ordinates method to solve the radiative transfer equation with support for:

- Multiple scattering
- Absorption and emission
- Thermal emission (Planck function)
- Various boundary conditions
- Pseudo-spherical geometry for direct beam

The CDISORT library offers improved performance over the original Fortran version through dynamic memory allocation and full double precision arithmetic.

## Installation

```bash
pip install nanodisort
```

Prerequisites:

- Supported platforms: Linux, macOS
- Python 3.9 to 3.13
- NumPy 1.20 or later

## Quickstart

```python
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
# ... (see documentation)

# Solve
state.solve()
```

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later), consistent with the CDISORT library it wraps.

## Acknowledgments

nanodisort authors are grateful to the developers of the many implementations of DISORT, and, in particular, to

- Timothy E. Dowling (original CDISORT C translation);
- Robert Buras (phase function extensions);
- Arve Kylling (pseudo-spherical approximation, testing);
- the original DISORT Fortran authors.

## Citation

If you use nanodisort in your research, please cite the [CDISORT paper](https://doi.org/10.1016/j.jqsrt.2011.03.019) and the nanodisort repository:

```bibtex
@software{Leroy_nanodisort,
    author = {Leroy, Vincent and Emde, Claudia},
    license = {GPL-3.0-or-later},
    title = {{nanodisort}},
    url = {https://github.com/eradiate/nanodisort},
    version = {0.1.0}
}
```
