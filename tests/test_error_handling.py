"""
Test error handling in nanodisort.

This test suite verifies that cdisort errors are properly converted to Python
exceptions instead of causing process crashes.
"""

import numpy as np
import pytest

import nanodisort as nd


def test_solve_without_optical_properties():
    """Test that calling solve without setting optical properties raises an exception."""
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


def test_invalid_nstr_odd():
    """Test that invalid nstr (odd number) is caught."""
    disort = nd.DisortState()
    disort.nstr = 7  # Invalid: must be even
    disort.nlyr = 6
    disort.nmom = 8
    disort.ntau = 5
    disort.numu = 4
    disort.nphi = 3

    # Should raise during allocation or solve
    with pytest.raises(RuntimeError, match="DISORT error"):
        disort.allocate()
        disort.solve()


def test_invalid_nstr_too_small():
    """Test that nstr < 2 is caught."""
    disort = nd.DisortState()
    disort.nstr = 0
    disort.nlyr = 1
    disort.nmom = 0
    disort.ntau = 1
    disort.numu = 0
    disort.nphi = 0

    # Should raise during allocation or solve
    with pytest.raises(RuntimeError, match="DISORT error"):
        disort.allocate()
        disort.solve()


def test_invalid_nlyr_zero():
    """Test that nlyr = 0 is caught."""
    disort = nd.DisortState()
    disort.nstr = 8
    disort.nlyr = 0  # Invalid
    disort.nmom = 8
    disort.ntau = 1
    disort.numu = 0
    disort.nphi = 0

    # Should raise during allocation
    # Note: Some invalid configurations may hang during allocation
    # Skip this test for now as nlyr=0 causes allocation issues
    pytest.skip("nlyr=0 causes allocation hangs - needs investigation")


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


def test_error_message_preserved():
    """Test that cdisort error messages are preserved in Python exceptions."""
    disort = nd.DisortState()
    disort.nstr = 3  # Invalid: must be even
    disort.nlyr = 1
    disort.nmom = 0
    disort.ntau = 1
    disort.numu = 0
    disort.nphi = 0

    # The exception should contain information about the error
    with pytest.raises(RuntimeError) as exc_info:
        disort.allocate()
        disort.solve()

    # Verify the exception message contains "DISORT error"
    assert "DISORT error" in str(exc_info.value)


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

    # Set invalid optical depth (negative)
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


def test_invalid_optical_depth():
    """Test that invalid optical depth values are caught."""
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
    disort.dtauc = np.array([-0.5])  # Invalid: negative optical depth
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

    # Should raise during solve
    with pytest.raises(RuntimeError, match="DISORT error"):
        disort.solve()


def test_invalid_single_scatter_albedo():
    """Test that invalid single scatter albedo values are caught."""
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
    disort.dtauc = np.array([0.5])
    disort.ssalb = np.array([1.5])  # Invalid: > 1
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
