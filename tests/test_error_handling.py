# SPDX-FileCopyrightText: 2025 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Test error handling in nanodisort.

This test suite verifies that cdisort errors are properly converted to Python
exceptions instead of causing process crashes.
"""

import numpy as np
import pytest

import nanodisort as nd


def test_solve_without_optical_properties():
    """
    Test that calling solve without setting optical properties raises an exception.
    """
    disort = nd.DisortState()
    disort.nstr = 8
    disort.nlyr = 6
    disort.nmom = 8
    disort.ntau = 5
    disort.numu = 4
    disort.nphi = 3
    disort.allocate()

    # Should raise RuntimeError with "DISORT error", not crash
    with pytest.raises(RuntimeError, match="DISORT error"):
        disort.solve()


@pytest.mark.parametrize(
    "nstr, nlyr, nmom, match",
    [
        (7, 6, 8, "nstr must be positive and even"),  # odd nstr
        (3, 1, 0, "nstr must be positive and even"),  # odd nstr, informative message
        (0, 1, 0, "nstr must be positive and even"),  # nstr < 2
        (8, 0, 8, "nlyr must be positive"),  # nlyr == 0
    ],
)
def test_invalid_dimensions(nstr, nlyr, nmom, match):
    """Test that invalid solver dimensions are caught at allocation."""
    disort = nd.DisortState()
    disort.nstr = nstr
    disort.nlyr = nlyr
    disort.nmom = nmom
    disort.ntau = 1
    disort.numu = 0
    disort.nphi = 0

    with pytest.raises(RuntimeError, match=match):
        disort.allocate()


def test_nmom_too_large():
    """Test that nmom > nstr is caught."""
    disort = nd.DisortState()
    disort.nstr = 8
    disort.nlyr = 1
    disort.nmom = 16  # Invalid: nmom should be <= nstr
    disort.ntau = 1
    disort.numu = 0
    disort.nphi = 0

    # Should raise during allocation or solve
    with pytest.raises(RuntimeError, match="DISORT error"):
        disort.allocate()
        disort.solve()


def test_valid_configuration_works():
    """Test that a valid configuration still works after error handling changes."""
    disort = nd.DisortState()
    disort.nstr = 8
    disort.nlyr = 1
    disort.nmom = 8
    disort.ntau = 2
    disort.numu = 0
    disort.nphi = 0

    disort.usrtau = True
    disort.usrang = False
    disort.onlyfl = True
    disort.lamber = True  # Lambertian surface

    # Should not raise
    disort.allocate()

    # Set minimal optical properties
    disort.dtauc = np.array([0.5])
    disort.ssalb = np.array([0.9])
    pmom = np.zeros((disort.nmom + 1, disort.nlyr))
    pmom[0, :] = 1.0
    disort.pmom = pmom

    disort.utau = np.array([0.0, 0.5])

    # Set boundary conditions
    disort.fbeam = 1.0
    disort.umu0 = 0.5
    disort.phi0 = 0.0
    disort.albedo = 0.0

    # Should work without errors
    disort.solve()

    # Verify output is available
    rfldir = disort.rfldir
    assert len(rfldir) == 2
    assert np.all(np.isfinite(rfldir))


def test_multiple_errors_sequential():
    """Test that multiple sequential error conditions are handled correctly."""
    # First error - invalid optical properties
    disort1 = nd.DisortState()
    disort1.nstr = 8
    disort1.nlyr = 1
    disort1.nmom = 8
    disort1.ntau = 1
    disort1.numu = 0
    disort1.nphi = 0
    disort1.lamber = True
    disort1.allocate()

    # Set invalid optical thickness (negative)
    disort1.dtauc = np.array([-0.5])
    disort1.ssalb = np.array([0.9])
    pmom = np.zeros((disort1.nmom + 1, disort1.nlyr))
    pmom[0, :] = 1.0
    disort1.pmom = pmom
    disort1.fbeam = 1.0
    disort1.umu0 = 0.5
    disort1.albedo = 0.0

    with pytest.raises(RuntimeError, match="DISORT error"):
        disort1.solve()

    # Second error - should still work
    disort2 = nd.DisortState()
    disort2.nstr = 8
    disort2.nlyr = 1
    disort2.nmom = 8
    disort2.ntau = 1
    disort2.numu = 0
    disort2.nphi = 0
    disort2.lamber = True
    disort2.allocate()

    # Set invalid single scatter albedo (>1)
    disort2.dtauc = np.array([0.5])
    disort2.ssalb = np.array([1.5])
    pmom = np.zeros((disort2.nmom + 1, disort2.nlyr))
    pmom[0, :] = 1.0
    disort2.pmom = pmom
    disort2.fbeam = 1.0
    disort2.umu0 = 0.5
    disort2.albedo = 0.0

    with pytest.raises(RuntimeError, match="DISORT error"):
        disort2.solve()


@pytest.mark.parametrize(
    "dtauc, ssalb",
    [
        (-0.5, 0.9),  # negative optical thickness
        (0.5, 1.5),  # single scattering albedo > 1
    ],
)
def test_invalid_optical_properties(dtauc, ssalb):
    """Test that invalid optical property values are caught during solve."""
    disort = nd.DisortState()
    disort.nstr = 8
    disort.nlyr = 1
    disort.nmom = 8
    disort.ntau = 2
    disort.numu = 0
    disort.nphi = 0

    disort.usrtau = True
    disort.usrang = False
    disort.onlyfl = True
    disort.lamber = True

    disort.allocate()

    # Set optical properties
    disort.dtauc = np.array([dtauc])
    disort.ssalb = np.array([ssalb])
    pmom = np.zeros((disort.nmom + 1, disort.nlyr))
    pmom[0, :] = 1.0
    disort.pmom = pmom

    disort.utau = np.array([0.0, 0.5])

    # Set boundary conditions
    disort.fbeam = 1.0
    disort.umu0 = 0.5
    disort.phi0 = 0.0
    disort.albedo = 0.0

    # Should raise during solve
    with pytest.raises(RuntimeError, match="DISORT error"):
        disort.solve()
