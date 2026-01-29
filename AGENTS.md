# AGENTS.md

This file provides guidance to coding agents when working with this repository.

## Project Overview

nanodisort is a Python binding for cdisort, a C implementation of the DISORT (Discrete Ordinates Radiative Transfer) solver. It provides a Pythonic interface to solve the radiative transfer equation in plane-parallel atmospheres with support for multiple scattering, absorption, emission, and various boundary conditions.

## Build & Development Commands

```bash
# Setup development environment
uv sync --dev

# (Re)build C++ extension module
uv pip install -ve .

# Run all tests
uv run task test

# Run tests with coverage
uv run task test-cov

# Run a single test file
uv run pytest tests/test_disotest.py -v

# Run a specific test
uv run pytest tests/test_disotest.py::TestDisort01::test_case_1a -v

# Lint
uv run task lint

# Format
uv run task format

# Check (lint + format check)
uv run task check

# Build docs
uv run task docs
```

## Architecture

### Directory Structure
- `src/nanodisort/` - Python package with C++ bindings
- `src/nanodisort/_core.cpp` - nanobind C++ extension wrapping cdisort
- `src/nanodisort/utils/phase_functions.py` - Phase function moment generators
- `ext/cdisort-2.1.3/` - Vendored cdisort C library
- `tests/` - pytest test suite

### Build System
- scikit-build-core for Python/CMake integration
- nanobind for C++17/Python bindings
- CMake builds cdisort as a static library, then links to the `_core` extension

### Core Components

**DisortState** (`__init__.py` wraps `_core.cpp`): Main solver interface
- Configure dimensions: `nstr`, `nlyr`, `nmom`, `ntau`, `numu`, `nphi`
- Set flags: `usrtau`, `usrang`, `lamber`, `planck`, `onlyfl`, `quiet`, `intensity_correction`, `spher`
- Set boundary conditions: `fbeam`, `umu0`, `phi0`, `fisot`, `albedo`, etc.
- Set optical properties: `dtauc` (layer optical depths), `ssalb` (single scattering albedo), `pmom` (phase function moments)
- Call `allocate()` then `solve()`
- Read outputs: `rfldir`, `rfldn`, `flup`, `dfdt`, `uu` (intensity), etc.

**Phase Functions** (`utils/phase_functions.py`): Generate Legendre moment arrays
- `isotropic(nmom)`, `rayleigh(nmom)`, `henyey_greenstein(gg, nmom)`
- `haze_l(nmom)`, `cloud_c1(nmom)` - tabulated aerosol/cloud phase functions

### Array Conventions
- C++ bindings handle Fortran-to-C array order conversion automatically
- `pmom` array shape: `(nmom_nstr+1, nlyr)` in C order
- `uu` intensity output shape: `(numu, ntau, nphi)` in C order

### Testing
- `test_disotest.py` - Ported tests from cdisort's disotest.c, validated against Van de Hulst, Sweigart, Garcia-Siewert benchmarks
- `test_core.py` - Basic solver functionality
- `test_error_handling.py` - Error condition tests

## Key Implementation Details

- Error handling: cdisort calls a registered callback that throws C++ exceptions, converted to Python RuntimeError by nanobind
- Memory: `DisortState.allocate()` must be called after setting dimensions and before setting arrays or calling `solve()`
- The `nmom` parameter controls phase function truncation; set `nmom >= nstr` for proper scattering