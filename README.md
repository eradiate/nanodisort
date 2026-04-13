# nanodisort

[![PyPI version](https://img.shields.io/pypi/v/nanodisort?color=blue)](https://pypi.org/project/nanodisort)
[![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/eradiate/nanodisort/test.yml?branch=main)](https://github.com/eradiate/nanodisort/actions/workflows/test.yml)
[![Documentation Status](https://img.shields.io/readthedocs/nanodisort)](https://nanodisort.readthedocs.io/latest)
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

Single Rayleigh-scattering layer illuminated by a solar beam at 60° zenith:

```python
import numpy as np
import nanodisort as nd
from nanodisort.utils import phase_functions as pf

# Configure dimensions and allocate
ds = nd.DisortState()
ds.nstr = 16  # number of streams
ds.nlyr = 1   # number of layers
ds.nmom = 16  # phase function moments (>= nstr)
ds.ntau = 2   # output optical depth levels
ds.numu = 6   # output polar angles
ds.nphi = 1   # output azimuthal angles
ds.allocate()

# Flags
ds.usrtau = True            # user-specified output levels
ds.usrang = True            # user-specified viewing angles
ds.lamber = True            # Lambertian lower boundary
ds.quiet = True             # suppress C-level output
ds.intensity_correction = True  # Nakajima-Tanaka correction

# Optical properties
ds.dtauc = np.array([0.1])          # extinction optical thickness
ds.ssalb = np.array([1.0])          # single-scattering albedo (pure scattering)
ds.pmom = pf.rayleigh(ds.nmom).reshape(-1, 1)  # Rayleigh phase function moments

# Output levels and viewing angles
ds.utau = np.array([0.0, 0.1])                    # TOA and surface
ds.umu = np.array([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])  # cosines of polar angles
ds.phi = np.array([0.0])                           # azimuthal angle [degrees]

# Boundary conditions
ds.fbeam = np.pi  # solar irradiance [W/m²]
ds.umu0 = 0.5     # cosine of solar zenith angle (60°)
ds.phi0 = 0.0     # solar azimuth [degrees]
ds.albedo = 0.0   # black surface
ds.fisot = 0.0    # no diffuse illumination from above

# Solve
ds.solve()

# Fluxes at each output level: shape (ntau,)
print(ds.rfldir)  # direct-beam downward irradiance
print(ds.rfldn)   # diffuse downward irradiance
print(ds.flup)    # diffuse upward irradiance

# Radiance field: shape (numu, ntau, nphi)
print(ds.uu.shape)
```

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later), consistent with the CDISORT library it wraps.

## Acknowledgements

nanodisort authors are grateful to the developers of the many implementations of DISORT, and, in particular, to

- Timothy E. Dowling (original CDISORT C translation);
- Robert Buras (phase function extensions);
- Arve Kylling (pseudo-spherical approximation, testing);
- the original DISORT Fortran authors (Knuth Stamnes, Si-Chee Tsay, Warren Wiscombe and Kolf Jayaweera).

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
