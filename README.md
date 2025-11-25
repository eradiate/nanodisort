# nanodisort

Python bindings for [cdisort](http://www.libradtran.org/doku.php?id=cdisort), a C implementation of the DISORT (Discrete Ordinates Radiative Transfer) solver.

## Overview

DISORT is a widely-used radiative transfer solver for plane-parallel atmospheres. It uses the discrete ordinates method to solve the radiative transfer equation with support for:

- Multiple scattering
- Absorption and emission
- Thermal emission (Planck function)
- Various boundary conditions
- Pseudo-spherical geometry for direct beam

The cdisort library offers improved performance over the original Fortran version through dynamic memory allocation and full double precision arithmetic.

## Installation

### Prerequisites

- Python 3.9 or later
- C compiler (gcc, clang, or MSVC)
- CMake 3.15 or later
- [uv](https://github.com/astral-sh/uv) (recommended)

### From source

```bash
git clone https://github.com/rayference/nanodisort.git
cd nanodisort
uv sync --dev
uv pip install -e .
```

## Quick Start

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

## Development

### Setup development environment

```bash
uv sync --dev
```

### (Re)build C++ module

```bash
uv pip install -ve.
```

### Run tests

```bash
uv run task test
```

### Run linting

```bash
uv run task lint
```

### Format code

```bash
uv run task format
```

### Build documentation

```bash
uv run task docs
```

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later), consistent with the cdisort library it wraps.

## Acknowledgments

- Timothy E. Dowling (original cdisort C translation)
- Robert Buras (phase function extensions)
- Arve Kylling (pseudo-spherical approximation, testing)
- Original DISORT Fortran authors
