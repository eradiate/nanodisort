# SPDX-FileCopyrightText: 2025 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Basic test cases for nanodisort solver

Tests include:
- Single homogeneous layer with pure scattering
- Multi-layer atmosphere with varying optical properties
- Property access and error handling
"""

import numpy as np
import pytest

from nanodisort import DisortState


def test_basic_beam_attenuation():
    """Test basic beam attenuation through a scattering layer."""
    # Create solver state
    ds = DisortState()

    # Set dimensions
    ds.nstr = 8  # Number of streams
    ds.nlyr = 1  # Single layer
    ds.nmom = 8  # Phase function moments
    ds.ntau = 2  # Output at boundaries (top and bottom)
    ds.numu = 0  # No user polar angles (only fluxes)
    ds.nphi = 0  # No azimuthal angles

    # Set control flags
    ds.usrtau = True  # User optical depths
    ds.usrang = False  # No user angles
    ds.lamber = True  # Lambertian bottom boundary
    ds.planck = False  # No thermal emission
    ds.onlyfl = True  # Only fluxes (no intensities)
    ds.quiet = True  # Suppress output

    # Allocate memory
    ds.allocate()

    # Set optical properties
    # Single layer with optical depth 1.0, pure scattering
    ds.dtauc = np.array([1.0])
    ds.ssalb = np.array([1.0])  # Single scattering albedo = 1 (no absorption)

    # Isotropic phase function (all moments = 0 except first)
    pmom = np.zeros((ds.nmom + 1, ds.nlyr))
    pmom[0, 0] = 1.0  # Normalization
    ds.pmom = pmom

    # Set output optical depths (boundaries)
    ds.utau = np.array([0.0, 1.0])

    # Set beam parameters
    ds.fbeam = np.pi  # Incident beam flux
    ds.umu0 = 1.0  # Normal incidence (cos(0) = 1)
    ds.phi0 = 0.0  # Azimuth angle

    # Bottom boundary albedo
    ds.albedo = 0.0  # No reflection from bottom

    # Other boundary conditions
    ds.fisot = 0.0  # No isotropic top illumination
    ds.fluor = 0.0  # No bottom illumination

    # Run solver
    ds.solve()

    # Get output fluxes
    rfldir = ds.rfldir  # Direct beam flux
    rfldn = ds.rfldn  # Diffuse downward flux
    flup = ds.flup  # Diffuse upward flux

    # Debug: print actual values
    print("\nDebug output:")
    print(f"  rfldir = {rfldir}")
    print(f"  rfldn = {rfldn}")
    print(f"  flup = {flup}")

    # Verify shapes
    assert rfldir.shape == (2,), f"Expected shape (2,), got {rfldir.shape}"
    assert rfldn.shape == (2,), f"Expected shape (2,), got {rfldn.shape}"
    assert flup.shape == (2,), f"Expected shape (2,), got {flup.shape}"

    # Basic sanity checks
    # Direct beam should attenuate exponentially: exp(-tau)
    expected_direct_bottom = np.pi * np.exp(-1.0)
    assert np.isclose(rfldir[0], np.pi, rtol=1e-4), (
        f"Top direct flux should be {np.pi}, got {rfldir[0]}"
    )
    assert np.isclose(rfldir[1], expected_direct_bottom, rtol=1e-2), (
        f"Bottom direct flux should be ~{expected_direct_bottom}, got {rfldir[1]}"
    )

    # Diffuse fluxes should be non-negative (with tolerance for numerical noise)
    assert np.all(rfldn >= -1e-6), "Diffuse downward flux should be non-negative"
    assert np.all(flup >= -1e-6), "Diffuse upward flux should be non-negative"

    # Top diffuse downward should be zero (no incident diffuse)
    assert np.isclose(rfldn[0], 0.0, atol=1e-6), (
        f"Top diffuse downward should be ~0, got {rfldn[0]}"
    )

    print(f"  Direct beam at top: {rfldir[0]:.6f}")
    print(f"  Direct beam at bottom: {rfldir[1]:.6f}")
    print(f"  Diffuse downward at bottom: {rfldn[1]:.6f}")
    print(f"  Diffuse upward at top: {flup[0]:.6f}")


def test_multilayer_atmosphere():
    """Test beam attenuation through a multi-layer atmosphere."""
    # Create solver state
    ds = DisortState()

    # Set dimensions - 3 layers with different optical properties
    ds.nstr = 8  # Number of streams
    ds.nlyr = 3  # Three layers
    ds.nmom = 8  # Phase function moments
    ds.ntau = 4  # Output at boundaries (top and bottom of each layer)
    ds.numu = 0  # No user polar angles (only fluxes)
    ds.nphi = 0  # No azimuthal angles

    # Set control flags
    ds.usrtau = True  # User optical depths
    ds.usrang = False  # No user angles
    ds.lamber = True  # Lambertian bottom boundary
    ds.planck = False  # No thermal emission
    ds.onlyfl = True  # Only fluxes (no intensities)
    ds.quiet = True  # Suppress output

    # Allocate memory
    ds.allocate()

    # Set optical properties for each layer
    # Layer 0 (top): thin, mostly scattering
    # Layer 1 (middle): thicker, moderately absorbing
    # Layer 2 (bottom): thick, more absorbing
    ds.dtauc = np.array([0.5, 1.0, 2.0])
    ds.ssalb = np.array([0.95, 0.85, 0.75])  # Decreasing single scattering albedo

    # Isotropic phase function for all layers
    pmom = np.zeros((ds.nmom + 1, ds.nlyr))
    pmom[0, :] = 1.0  # Normalization for all layers
    ds.pmom = pmom

    # Set output optical depths (layer boundaries)
    # Cumulative optical depth: 0.0 (top), 0.5, 1.5, 3.5 (bottom)
    ds.utau = np.array([0.0, 0.5, 1.5, 3.5])

    # Set beam parameters
    ds.fbeam = np.pi  # Incident beam flux
    ds.umu0 = 1.0  # Normal incidence (cos(0) = 1)
    ds.phi0 = 0.0  # Azimuth angle

    # Bottom boundary albedo
    ds.albedo = 0.0  # No reflection from bottom

    # Other boundary conditions
    ds.fisot = 0.0  # No isotropic top illumination
    ds.fluor = 0.0  # No bottom illumination

    # Run solver
    ds.solve()

    # Get output fluxes
    rfldir = ds.rfldir  # Direct beam flux
    rfldn = ds.rfldn  # Diffuse downward flux
    flup = ds.flup  # Diffuse upward flux

    # Debug: print actual values
    print("\nMultilayer atmosphere output:")
    print(f"  rfldir = {rfldir}")
    print(f"  rfldn = {rfldn}")
    print(f"  flup = {flup}")

    # Verify shapes
    assert rfldir.shape == (4,), f"Expected shape (4,), got {rfldir.shape}"
    assert rfldn.shape == (4,), f"Expected shape (4,), got {rfldn.shape}"
    assert flup.shape == (4,), f"Expected shape (4,), got {flup.shape}"

    # Verify direct beam attenuation follows Beer's law
    expected_direct = np.pi * np.exp(-np.array([0.0, 0.5, 1.5, 3.5]))
    assert np.isclose(rfldir[0], expected_direct[0], rtol=1e-4), (
        f"Top direct flux should be {expected_direct[0]}, got {rfldir[0]}"
    )
    for i in range(1, 4):
        assert np.isclose(rfldir[i], expected_direct[i], rtol=1e-2), (
            f"Direct flux at level {i} should be ~{expected_direct[i]}, got {rfldir[i]}"
        )

    # Direct beam should monotonically decrease
    assert np.all(np.diff(rfldir) <= 0), "Direct beam should decrease with depth"

    # Diffuse fluxes should be non-negative
    assert np.all(rfldn >= -1e-6), "Diffuse downward flux should be non-negative"
    assert np.all(flup >= -1e-6), "Diffuse upward flux should be non-negative"

    # Top diffuse downward should be zero (no incident diffuse)
    assert np.isclose(rfldn[0], 0.0, atol=1e-6), (
        f"Top diffuse downward should be ~0, got {rfldn[0]}"
    )

    # Diffuse fluxes should increase with depth (more scattering accumulated)
    # then decrease as beam is attenuated
    print(f"  Direct beam attenuation: {rfldir}")
    print(f"  Expected direct beam: {expected_direct}")
    print(f"  Diffuse downward: {rfldn}")
    print(f"  Diffuse upward: {flup}")


def test_property_access():
    """Test that all properties can be accessed correctly."""
    ds = DisortState()

    # Test dimension properties
    ds.nstr = 8
    assert ds.nstr == 8

    ds.nlyr = 2
    assert ds.nlyr == 2

    ds.nmom = 4
    assert ds.nmom == 4

    # Test flag properties
    ds.usrtau = True
    assert ds.usrtau == True

    ds.lamber = False
    assert ds.lamber == False

    ds.planck = True
    assert ds.planck == True

    # Test boundary condition properties
    ds.fbeam = 3.14159
    assert np.isclose(ds.fbeam, 3.14159)

    ds.umu0 = 0.5
    assert np.isclose(ds.umu0, 0.5)

    ds.albedo = 0.3
    assert np.isclose(ds.albedo, 0.3)


def test_array_access_before_allocation():
    """Test that accessing arrays before allocation raises an error."""
    ds = DisortState()
    ds.nlyr = 2

    # Should raise error before allocation
    with pytest.raises(RuntimeError, match="not allocated"):
        _ = ds.dtauc

    with pytest.raises(RuntimeError, match="not allocated"):
        ds.dtauc = np.array([0.1, 0.2])
