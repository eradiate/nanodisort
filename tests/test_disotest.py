# SPDX-FileCopyrightText: 2025 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Port of disotest.c test cases to Python.

These tests verify the nanodisort implementation against known benchmark
results from the literature. The tests cover various scattering scenarios,
boundary conditions, and solver configurations.

References (as cited in original disotest.c):
    VH1, VH2: Van de Hulst, H.C., 1980: Multiple Light Scattering
    SW: Sweigart A, 1970: Radiative Transfer in Atmospheres Scattering
        According to the Rayleigh Phase Function with Absorption
    GS: Garcia RDM, Siewert CE, 1985: Benchmark Results in Radiative Transfer

The tests verify both:
- Flux outputs: rfldir (direct beam), rfldn (diffuse down), flup (diffuse up), dfdt
- Intensity outputs: uu (intensity at user angles) [numu, ntau, nphi]
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nanodisort import BRDFType, DisortState
from nanodisort.utils import phase_functions as pf


class TestDisort01:
    """
    Test Problem 1: Isotropic Scattering

    Compare to Ref. VH1, Table 12

    This test examines isotropic scattering with various combinations of:
    - Optical thicknesses: 0.03125, 32.0
    - Single scattering albedos: 0.2, 0.99, 1.0
    - Sources: beam, isotropic
    """

    _TEST_01_PARAMS = {
        # Case 1a: tau=0.03125, ssalb=0.2, beam source
        "a": {
            "utau": 0.03125,
            "ssalb": 0.2,
            "umu0": 0.1,
            "fisot": 0.0,
            "expected_rfldir": [3.14159, 2.29844],
            "expected_rfldn": [0.0, 7.94108e-02],
            "expected_flup": [7.99451e-02, 0.0],
            "expected_dfdt": [2.54067e01, 1.86531e01],
            "expected_uu": np.array(
                [
                    [[0.0], [1.33826e-02]],  # umu = -1.0
                    [[0.0], [2.63324e-02]],  # umu = -0.5
                    [[0.0], [1.15898e-01]],  # umu = -0.1
                    [[1.17771e-01], [0.0]],  # umu = 0.1
                    [[2.64170e-02], [0.0]],  # umu = 0.5
                    [[1.34041e-02], [0.0]],  # umu = 1.0
                ]
            ),
        },
        # Case 1b: tau=0.03125, ssalb=1.0, beam source
        "b": {
            "utau": 0.03125,
            "ssalb": 1.0,
            "umu0": 0.1,
            "fisot": 0.0,
            "expected_rfldir": [3.14159, 2.29844],
            "expected_rfldn": [0.0, 4.20233e-01],
            "expected_flup": [4.22922e-01, 0.0],
            "expected_dfdt": [0.0, 0.0],
            "expected_uu": np.array(
                [
                    [[0.0], [7.08109e-02]],  # umu = -1.0
                    [[0.0], [1.39337e-01]],  # umu = -0.5
                    [[0.0], [6.13458e-01]],  # umu = -0.1
                    [[6.22884e-01], [0.0]],  # umu = 0.1
                    [[1.39763e-01], [0.0]],  # umu = 0.5
                    [[7.09192e-02], [0.0]],  # umu = 1.0
                ]
            ),
        },
        # Case 1c: tau=0.03125, ssalb=0.99, isotropic source
        "c": {
            "utau": 0.03125,
            "ssalb": 0.99,
            "umu0": None,  # no incident beam
            "fisot": 1.0,
            "expected_rfldir": [0.0, 0.0],
            "expected_rfldn": [3.14159, 3.04897],
            "expected_flup": [9.06556e-02, 0.0],
            "expected_dfdt": [6.66870e-02, 5.88936e-02],
            "expected_uu": np.array(
                [
                    [[1.0], [9.84447e-01]],  # umu = -1.0
                    [[1.0], [9.69363e-01]],  # umu = -0.5
                    [[1.0], [8.63946e-01]],  # umu = -0.1
                    [[1.33177e-01], [0.0]],  # umu = 0.1
                    [[2.99879e-02], [0.0]],  # umu = 0.5
                    [[1.52233e-02], [0.0]],  # umu = 1.0
                ]
            ),
        },
        # Case 1d: tau=32.0, ssalb=0.2, beam source
        "d": {
            "utau": 32.0,
            "ssalb": 0.2,
            "umu0": 0.1,
            "fisot": 0.0,
            "expected_rfldir": [3.14159, 0.0],
            "expected_rfldn": [0.0, 0.0],
            "expected_flup": [2.59686e-01, 0.0],
            "expected_dfdt": [2.57766e01, 0.0],
            "expected_uu": np.array(
                [
                    [[0.0], [1.22980e-15]],  # umu = -1.0
                    [[0.0], [1.30698e-17]],  # umu = -0.5
                    [[0.0], [6.88840e-18]],  # umu = -0.1
                    [[2.62972e-01], [0.0]],  # umu = 0.1
                    [[9.06967e-02], [0.0]],  # umu = 0.5
                    [[5.02853e-02], [0.0]],  # umu = 1.0
                ]
            ),
        },
        # Case 1e: tau=32.0, ssalb=1.0, beam source
        "e": {
            "utau": 32.0,
            "ssalb": 1.0,
            "umu0": 0.1,
            "fisot": 0.0,
            "expected_rfldir": [3.14159, 0.0],
            "expected_rfldn": [0.0, 6.76954e-02],
            "expected_flup": [3.07390, 0.0],
            "expected_dfdt": [0.0, 0.0],
            "expected_uu": np.array(
                [
                    [[0.0], [2.71316e-02]],  # umu = -1.0
                    [[0.0], [1.87805e-02]],  # umu = -0.5
                    [[0.0], [1.16385e-02]],  # umu = -0.1
                    [[1.93321e00], [0.0]],  # umu = 0.1
                    [[1.02732e00], [0.0]],  # umu = 0.5
                    [[7.97199e-01], [0.0]],  # umu = 1.0
                ]
            ),
        },
        "f": {
            "utau": 32.0,
            "ssalb": 0.99,
            "umu0": None,  # no incident beam
            "fisot": 1.0,
            "expected_rfldir": [0.0, 0.0],
            "expected_rfldn": [3.14159, 4.60048e-03],
            "expected_flup": [2.49618, 0.0],
            "expected_dfdt": [1.14239e-01, 7.93633e-05],
            "expected_uu": np.array(
                [
                    [[1.0], [1.86840e-03]],  # umu = -1.0
                    [[1.0], [1.26492e-03]],  # umu = -0.5
                    [[1.0], [7.79280e-04]],  # umu = -0.1
                    [[8.77510e-01], [0.0]],  # umu = 0.1
                    [[8.15136e-01], [0.0]],  # umu = 0.5
                    [[7.52715e-01], [0.0]],  # umu = 1.0
                ]
            ),
        },
    }

    @pytest.mark.parametrize(
        "case_id, utau, ssalb, umu0, fisot, expected_rfldir, "
        "expected_rfldn, expected_flup, expected_dfdt, expected_uu",
        [
            (
                k,  # case_id
                v["utau"],
                v["ssalb"],
                v["umu0"],
                v["fisot"],
                v["expected_rfldir"],
                v["expected_rfldn"],
                v["expected_flup"],
                v["expected_dfdt"],
                v["expected_uu"],
            )
            for k, v in _TEST_01_PARAMS.items()
        ],
        ids=list(_TEST_01_PARAMS.keys()),
    )
    def test(
        self,
        case_id,
        utau,
        ssalb,
        umu0,
        fisot,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
        expected_uu,
    ):
        """
        Run a single test case for Test 1.

        Parameters
        ----------
        case_id : str
            Subcase ID.

        utau : float
            Layer optical thickness.

        ssalb : float
            Single scattering albedo.

        umu0 : float or None
            Polar angle cosine of incident beam. None for isotropic source.

        fisot : float
            Isotropic top illumination.

        expected_* : array-like
            Expected flux values at [top, bottom].

        expected_uu : array-like
            Expected intensity values, shape (numu, ntau, nphi).
        """
        ds = DisortState()

        # Set dimensions
        ds.nstr = 16
        ds.nlyr = 1
        ds.nmom = 16
        ds.ntau = 2
        ds.numu = 6
        ds.nphi = 1

        # Set flags
        ds.usrtau = True
        ds.usrang = True
        ds.lamber = True
        ds.planck = False
        ds.onlyfl = False
        ds.quiet = True
        ds.intensity_correction = True
        ds.old_intensity_correction = True

        # Allocate memory
        ds.allocate()

        # Set optical properties - isotropic phase function
        ds.dtauc = np.array([utau])
        ds.ssalb = np.array([ssalb])
        pmom = pf.isotropic(ds.nmom).reshape(-1, 1)
        ds.pmom = pmom

        # Set output optical thicknesses
        ds.utau = np.array([0.0, utau])

        # Set angles
        ds.umu = np.array([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
        ds.phi = np.array([0.0])

        # Set beam parameters
        fbeam = np.pi / umu0 if umu0 is not None else 0.0
        ds.fbeam = fbeam
        ds.umu0 = umu0 if umu0 is not None else 1.0
        ds.phi0 = 0.0

        # Set boundary conditions
        ds.albedo = 0.0
        ds.fisot = fisot
        ds.fluor = 0.0

        # Run solver
        ds.solve()

        # Check flux outputs
        print(
            f"\nTest 1{case_id}: tau={utau}, ssalb={ssalb}, beam={fbeam}, isot={fisot}"
        )
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        # Compare with expected values
        assert_allclose(ds.rfldir, expected_rfldir, rtol=1e-4, atol=1e-9)
        assert_allclose(ds.rfldn, expected_rfldn, rtol=1e-4, atol=1e-9)
        assert_allclose(ds.flup, expected_flup, rtol=1e-4, atol=1e-9)
        assert_allclose(ds.dfdt, expected_dfdt, rtol=1e-3, atol=1e-9)

        # Verify intensity
        uu = ds.uu
        print(f"  uu shape: {uu.shape}")
        print(f"  uu[:, 0, 0] (top): {uu[:, 0, 0]}")
        print(f"  uu[:, 1, 0] (bottom): {uu[:, 1, 0]}")
        assert_allclose(uu, expected_uu, rtol=1e-3, atol=1e-9)


class TestDisort02:
    """
    Test Problem 2: Rayleigh Scattering, Beam Source

    Compare to Ref. SW, Table 1

    This test examines Rayleigh scattering with:
    - Optical thicknesses: 0.2, 5.0
    - Single scattering albedos: 0.5, 1.0
    """

    _TEST_02_PARAMS = {
        # Case 2a: tau=0.2, ssalb=0.5
        "a": {
            "utau": 0.2,
            "ssalb": 0.5,
            "expected_rfldir": [2.52716e-01, 2.10311e-02],
            "expected_rfldn": [0.0, 4.41791e-02],
            "expected_flup": [5.35063e-02, 0.0],
            "expected_dfdt": [1.66570e00, 1.89848e-01],
            "expected_uu": np.array(
                [
                    [[0.0], [7.71897e-03]],  # umu = -0.981986
                    [[0.0], [2.00778e-02]],  # umu = -0.538263
                    [[0.0], [2.57685e-02]],  # umu = -0.018014
                    [[1.61796e-01], [0.0]],  # umu = 0.018014
                    [[2.11501e-02], [0.0]],  # umu = 0.538263
                    [[7.86713e-03], [0.0]],  # umu = 0.981986
                ]
            ),
        },
        # Case 2b: tau=0.2, ssalb=1.0
        "b": {
            "utau": 0.2,
            "ssalb": 1.0,
            "expected_rfldir": [2.52716e-01, 2.10311e-02],
            "expected_rfldn": [0.0, 1.06123e-01],
            "expected_flup": [1.25561e-01, 0.0],
            "expected_dfdt": [0.0, 0.0],
            "expected_uu": np.array(
                [
                    [[0.0], [1.86027e-02]],  # umu = -0.981986
                    [[0.0], [4.64061e-02]],  # umu = -0.538263
                    [[0.0], [6.77603e-02]],  # umu = -0.018014
                    [[3.47678e-01], [0.0]],  # umu = 0.018014
                    [[4.87120e-02], [0.0]],  # umu = 0.538263
                    [[1.89387e-02], [0.0]],  # umu = 0.981986
                ]
            ),
        },
        # Case 2c: tau=5.0, ssalb=0.5
        "c": {
            "utau": 5.0,
            "ssalb": 0.5,
            "expected_rfldir": [2.52716e-01, 2.56077e-28],
            "expected_rfldn": [0.0, 2.51683e-04],
            "expected_flup": [6.24730e-02, 0.0],
            "expected_dfdt": [1.67462e00, 1.75464e-04],
            "expected_uu": np.array(
                [
                    [[0.0], [1.70004e-04]],  # umu = -0.981986
                    [[0.0], [3.97168e-05]],  # umu = -0.538263
                    [[0.0], [1.32472e-05]],  # umu = -0.018014
                    [[1.62566e-01], [0.0]],  # umu = 0.018014
                    [[2.45786e-02], [0.0]],  # umu = 0.538263
                    [[1.01498e-02], [0.0]],  # umu = 0.981986
                ]
            ),
        },
        # Case 2d: tau=5.0, ssalb=1.0
        "d": {
            "utau": 5.0,
            "ssalb": 1.0,
            "expected_rfldir": [2.52716e-01, 0.0],
            "expected_rfldn": [0.0, 2.68008e-02],
            "expected_flup": [2.25915e-01, 0.0],
            "expected_dfdt": [0.0, 0.0],
            "expected_uu": np.array(
                [
                    [[0.0], [1.05950e-02]],  # umu = -0.981986
                    [[0.0], [7.69337e-03]],  # umu = -0.538263
                    [[0.0], [3.79276e-03]],  # umu = -0.018014
                    [[3.64010e-01], [0.0]],  # umu = 0.018014
                    [[8.26993e-02], [0.0]],  # umu = 0.538263
                    [[4.92370e-02], [0.0]],  # umu = 0.981986
                ]
            ),
        },
    }

    @pytest.mark.parametrize(
        "case_id, utau, ssalb, expected_rfldir, expected_rfldn, expected_flup, "
        "expected_dfdt, expected_uu",
        [
            (
                k,  # case_id
                v["utau"],
                v["ssalb"],
                v["expected_rfldir"],
                v["expected_rfldn"],
                v["expected_flup"],
                v["expected_dfdt"],
                v["expected_uu"],
            )
            for k, v in _TEST_02_PARAMS.items()
        ],
        ids=list(_TEST_02_PARAMS.keys()),
    )
    def test(
        self,
        case_id,
        utau,
        ssalb,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
        expected_uu,
    ):
        """
        Run a single test case for Test 2.

        Parameters
        ----------
        case_id : str
            Subcase ID.

        utau : float
            Layer optical thickness.

        ssalb : float
            Single scattering albedo.

        expected_* : array-like
            Expected flux values at [top, bottom].

        expected_uu : array-like
            Expected intensity values, shape (numu, ntau, nphi).
        """
        ds = DisortState()

        # Set dimensions
        ds.nstr = 16
        ds.nlyr = 1
        ds.nmom = 16
        ds.ntau = 2
        ds.numu = 6
        ds.nphi = 1

        # Set flags
        ds.usrtau = True
        ds.usrang = True
        ds.lamber = True
        ds.planck = False
        ds.onlyfl = False
        ds.quiet = True
        ds.intensity_correction = True
        ds.old_intensity_correction = True

        # Allocate memory
        ds.allocate()

        # Set optical properties - Rayleigh phase function
        ds.dtauc = np.array([utau])
        ds.ssalb = np.array([ssalb])
        pmom = pf.rayleigh(ds.nmom).reshape(-1, 1)
        ds.pmom = pmom

        # Set output optical thicknesses
        ds.utau = np.array([0.0, utau])

        # Set angles (from Ref. SW)
        ds.umu = np.array(
            [-0.981986, -0.538263, -0.018014, 0.018014, 0.538263, 0.981986]
        )
        ds.phi = np.array([0.0])

        # Set beam parameters
        ds.fbeam = np.pi
        ds.umu0 = 0.080442
        ds.phi0 = 0.0

        # Set boundary conditions
        ds.albedo = 0.0
        ds.fisot = 0.0
        ds.fluor = 0.0

        # Run solver
        ds.solve()

        # Check outputs
        print(f"\nTest 2{case_id}: tau={utau}, ssalb={ssalb}")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        # Compare with expected values
        assert_allclose(ds.rfldir, expected_rfldir, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.rfldn, expected_rfldn, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.flup, expected_flup, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.dfdt, expected_dfdt, rtol=1e-3, atol=1e-9)

        # Verify intensity
        uu = ds.uu
        print(f"  uu shape: {uu.shape}")
        print(f"  uu[:, 0, 0] (top): {uu[:, 0, 0]}")
        print(f"  uu[:, 1, 0] (bottom): {uu[:, 1, 0]}")
        assert_allclose(uu, expected_uu, rtol=1e-3, atol=1e-9)


class TestDisort03:
    """
    Test Problem 3: Henyey-Greenstein Scattering

    Compare to Ref. VH2, Table 37

    This test examines Henyey-Greenstein phase function with asymmetry factor
    g=0.75, testing:
    - Optical thicknesses: 1.0, 8.0
    - Single scattering albedo: 1.0
    - nmom=32 (more moments than nstr=16)
    """

    _TEST_03_PARAMS = {
        "a": {
            "utau": 1.0,
            "expected_rfldir": [3.14159, 1.15573],
            "expected_rfldn": [0.0, 1.73849],
            "expected_flup": [2.47374e-01, 0.0],
            "expected_dfdt": [0.0, 0.0],
            "expected_uu": np.array(
                [
                    [[0.0], [3.05855e00]],  # umu = -1.0
                    [[0.0], [2.66648e-01]],  # umu = -0.5
                    [[0.0], [2.13750e-01]],  # umu = -0.1
                    [[1.51159e-01], [0.0]],  # umu = 0.1
                    [[1.01103e-01], [0.0]],  # umu = 0.5
                    [[3.95460e-02], [0.0]],  # umu = 1.0
                ]
            ),
        },
        "b": {
            "utau": 8.0,
            "expected_rfldir": [3.14159, 1.05389e-03],
            "expected_rfldn": [0.0, 1.54958],
            "expected_flup": [1.59096, 0.0],
            "expected_dfdt": [0.0, 0.0],
            "expected_uu": np.array(
                [
                    [[0.0], [6.69581e-01]],  # umu = -1.0
                    [[0.0], [4.22350e-01]],  # umu = -0.5
                    [[0.0], [2.36362e-01]],  # umu = -0.1
                    [[3.79740e-01], [0.0]],  # umu = 0.1
                    [[5.19598e-01], [0.0]],  # umu = 0.5
                    [[4.93302e-01], [0.0]],  # umu = 1.0
                ]
            ),
        },
    }

    @pytest.mark.parametrize(
        "case_id, utau, expected_rfldir, expected_rfldn, expected_flup, "
        "expected_dfdt, expected_uu",
        [
            (
                k,  # case_id
                v["utau"],
                v["expected_rfldir"],
                v["expected_rfldn"],
                v["expected_flup"],
                v["expected_dfdt"],
                v["expected_uu"],
            )
            for k, v in _TEST_03_PARAMS.items()
        ],
        ids=list(_TEST_03_PARAMS.keys()),
    )
    def test(
        self,
        case_id,
        utau,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
        expected_uu,
    ):
        """
        Run a single test case for Test 3.

        Parameters
        ----------
        case_id : str
            Subcase ID.

        utau : float
            Layer optical thickness.

        expected_* : array-like
            Expected flux values at [top, bottom].

        expected_uu : array-like
            Expected intensity values, shape (numu, ntau, nphi).
        """
        ds = DisortState()

        # Set dimensions
        ds.nstr = 16
        ds.nlyr = 1
        ds.nmom = 32  # Note: more moments than streams
        ds.ntau = 2
        ds.numu = 6
        ds.nphi = 1

        # Set flags
        ds.usrtau = True
        ds.usrang = True
        ds.lamber = True
        ds.planck = False
        ds.onlyfl = False
        ds.quiet = True
        ds.intensity_correction = True
        ds.old_intensity_correction = True

        # Allocate memory
        ds.allocate()

        # Set optical properties - Henyey-Greenstein with g=0.75
        ds.dtauc = np.array([utau])
        ds.ssalb = np.array([1.0])
        pmom = pf.henyey_greenstein(0.75, ds.nmom).reshape(-1, 1)
        ds.pmom = pmom

        # Set output optical thicknesses
        ds.utau = np.array([0.0, utau])

        # Set angles
        ds.umu = np.array([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
        ds.phi = np.array([0.0])

        # Set beam parameters (normal incidence)
        ds.fbeam = np.pi  # pi/umu0 where umu0=1.0
        ds.umu0 = 1.0
        ds.phi0 = 0.0

        # Set boundary conditions
        ds.albedo = 0.0
        ds.fisot = 0.0
        ds.fluor = 0.0

        # Run solver
        ds.solve()

        # Check outputs
        print(f"\nTest 3{case_id}: tau={utau}, HG(g=0.75)")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        # Compare with expected values
        assert_allclose(ds.rfldir, expected_rfldir, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.rfldn, expected_rfldn, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.flup, expected_flup, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.dfdt, expected_dfdt, rtol=1e-3, atol=1e-9)

        # Verify intensity field if expected values provided
        uu = ds.uu
        print(f"  uu shape: {uu.shape}")
        print(f"  uu[:, 0, 0] (top): {uu[:, 0, 0]}")
        print(f"  uu[:, 1, 0] (bottom): {uu[:, 1, 0]}")
        assert_allclose(uu, expected_uu, rtol=1e-3, atol=1e-9)


class TestDisort04:
    """
    Test Problem 4: Haze-L Scattering, Beam Source

    Compare to Ref. GS (Garcia-Siewert), Tables 12-16

    This test examines realistic atmospheric haze scattering using the
    Garcia-Siewert Haze-L phase function with:
    - Optical thickness: 1.0
    - Single scattering albedos: 1.0, 0.9
    - Multiple azimuthal angles (case c)
    """

    _TEST_04_PARAMS = {
        # Case 4a: tau=1.0, Haze-L, ssalb=1.0, normal incidence (Table 12)
        "a": {
            "ssalb": 1.0,
            "umu0": 1.0,
            "nphi": 1,
            "phi_values": [0.0],
            "expected_rfldir": [3.14159, 1.90547, 1.15573],
            "expected_rfldn": [0.0, 1.17401, 1.81264],
            "expected_flup": [1.73223e-01, 1.11113e-01, 0.0],
            "expected_dfdt": [0.0, 0.0, 0.0],
        },
        # Case 4b: tau=1.0, Haze-L, ssalb=0.9, normal incidence (Table 13)
        "b": {
            "ssalb": 0.9,
            "umu0": 1.0,
            "nphi": 1,
            "phi_values": [0.0],
            "expected_rfldir": [3.14159, 1.90547, 1.15573],
            "expected_rfldn": [0.0, 1.01517, 1.51554],
            "expected_flup": [1.23665e-01, 7.88690e-02, 0.0],
            "expected_dfdt": [3.43724e-01, 3.52390e-01, 3.19450e-01],
        },
        # Case 4c: tau=1.0, Haze-L, ssalb=0.9, oblique incidence (Tables 14-16)
        "c": {
            "ssalb": 0.9,
            "umu0": 0.5,
            "nphi": 3,
            "phi_values": [0.0, 90.0, 180.0],
            "expected_rfldir": [1.57080, 5.77864e-01, 2.12584e-01],
            "expected_rfldn": [0.0, 7.02764e-01, 8.03294e-01],
            "expected_flup": [2.25487e-01, 1.23848e-01, 0.0],
            "expected_dfdt": [3.85003e-01, 3.37317e-01, 2.16403e-01],
        },
    }

    @pytest.mark.parametrize(
        "case_id, ssalb, umu0, nphi, phi_values, expected_rfldir, "
        "expected_rfldn, expected_flup, expected_dfdt",
        [
            (
                k,  # case_id
                v["ssalb"],
                v["umu0"],
                v["nphi"],
                v["phi_values"],
                v["expected_rfldir"],
                v["expected_rfldn"],
                v["expected_flup"],
                v["expected_dfdt"],
            )
            for k, v in _TEST_04_PARAMS.items()
        ],
        ids=list(_TEST_04_PARAMS.keys()),
    )
    def test(
        self,
        case_id,
        ssalb,
        umu0,
        nphi,
        phi_values,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
    ):
        """
        Run a single test case for Test 4.

        Parameters
        ----------
        case_id : str
            Subcase ID.

        ssalb : float
            Single scattering albedo.

        umu0 : float
            Cosine of incident beam angle.

        nphi : int
            Number of azimuthal angles.

        phi_values : list
            Azimuthal angles in degrees.

        expected_* : list
            Expected flux values at [0.0, 0.5, 1.0] optical thicknesses.
        """
        ds = DisortState()

        # Set dimensions
        ds.nstr = 32
        ds.nlyr = 1
        ds.nmom = 32
        ds.ntau = 3
        ds.numu = 6
        ds.nphi = nphi

        # Set flags
        ds.usrtau = True
        ds.usrang = True
        ds.lamber = True
        ds.planck = False
        ds.onlyfl = False
        ds.quiet = True

        # Allocate memory
        ds.allocate()

        # Set optical properties - Haze-L phase function
        ds.dtauc = np.array([1.0])
        ds.ssalb = np.array([ssalb])
        pmom = pf.haze_l(ds.nmom).reshape(-1, 1)
        ds.pmom = pmom

        # Set output optical thicknesses
        ds.utau = np.array([0.0, 0.5, 1.0])

        # Set angles
        ds.umu = np.array([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
        ds.phi = np.array(phi_values)

        # Set beam parameters
        ds.fbeam = np.pi
        ds.umu0 = umu0
        ds.phi0 = 0.0

        # Set boundary conditions
        ds.albedo = 0.0
        ds.fisot = 0.0
        ds.fluor = 0.0

        # Run solver
        ds.solve()

        # Check outputs
        print(f"\nTest 4{case_id}: tau=1.0, Haze-L, ssalb={ssalb}, umu0={umu0}")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        # Compare with expected values
        assert_allclose(ds.rfldir, expected_rfldir, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.rfldn, expected_rfldn, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.flup, expected_flup, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.dfdt, expected_dfdt, rtol=1e-3, atol=1e-9)


class TestDisort05:
    """
    Test Problem 5: Cloud C.1 Scattering, Beam Source

    Compare to Ref. GS (Garcia-Siewert), Tables 19-20

    This test examines realistic cloud scattering using the Garcia-Siewert
    Cloud C.1 phase function with:
    - Optical thickness: 64.0
    - Single scattering albedos: 1.0, 0.9
    - Multiple output levels within the layer
    """

    _TEST_05_PARAMS = {
        # Case 5a: tau=64.0, Cloud C.1, ssalb=1.0 (Table 19)
        "a": {
            "ssalb": 1.0,
            "utau": [0.0, 32.0, 64.0],
            "expected_rfldir": [3.14159, 3.97856e-14, 5.03852e-28],
            "expected_rfldn": [0.0, 2.24768, 4.79851e-01],
            "expected_flup": [2.66174, 1.76783, 0.0],
            "expected_dfdt": [0.0, 0.0, 0.0],
        },
        # Case 5b: tau=64.0, Cloud C.1, ssalb=0.9 (Table 20)
        "b": {
            "ssalb": 0.9,
            "utau": [3.2, 12.8, 48.0],
            "expected_rfldir": [1.28058e-01, 8.67322e-06, 4.47729e-21],
            "expected_rfldn": [1.74767, 2.33975e-01, 6.38345e-05],
            "expected_flup": [2.70485e-01, 3.74252e-02, 1.02904e-05],
            "expected_dfdt": [3.10129e-01, 4.52671e-02, 1.25021e-05],
        },
    }

    @pytest.mark.parametrize(
        "case_id, ssalb, utau, expected_rfldir, expected_rfldn, expected_flup, "
        "expected_dfdt",
        [
            (
                k,  # case_id
                v["ssalb"],
                v["utau"],
                v["expected_rfldir"],
                v["expected_rfldn"],
                v["expected_flup"],
                v["expected_dfdt"],
            )
            for k, v in _TEST_05_PARAMS.items()
        ],
        ids=list(_TEST_05_PARAMS.keys()),
    )
    def test(
        self,
        case_id,
        ssalb,
        utau,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
    ):
        """
        Run a single test case for Test 5.

        Parameters
        ----------
        case_id : str
            Subcase ID.

        ssalb : float
            Single scattering albedo.

        utau : list
            Output optical thickness levels.

        expected_* : list
            Expected flux values at specified optical thicknesses.
        """
        ds = DisortState()

        # Set dimensions
        ds.nstr = 48
        ds.nlyr = 1
        ds.nmom = 299  # Cloud C.1 has 298 tabulated moments
        ds.ntau = 3
        ds.numu = 6
        ds.nphi = 1

        # Set flags
        ds.usrtau = True
        ds.usrang = True
        ds.lamber = True
        ds.planck = False
        ds.onlyfl = False
        ds.quiet = True

        # Allocate memory
        ds.allocate()

        # Set optical properties - Cloud C.1 phase function
        ds.dtauc = np.array([64.0])
        ds.ssalb = np.array([ssalb])
        pmom = pf.cloud_c1(ds.nmom).reshape(-1, 1)
        ds.pmom = pmom

        # Set output optical thicknesses
        ds.utau = np.array(utau)

        # Set angles
        ds.umu = np.array([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
        ds.phi = np.array([0.0])

        # Set beam parameters (normal incidence)
        ds.fbeam = np.pi
        ds.umu0 = 1.0
        ds.phi0 = 0.0

        # Set boundary conditions
        ds.albedo = 0.0
        ds.fisot = 0.0
        ds.fluor = 0.0

        # Run solver
        ds.solve()

        # Check outputs
        print(f"\nTest 5{case_id}: Cloud C.1, ssalb={ssalb}")
        print(f"  utau: {utau}")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        # Compare with expected values
        # Use looser tolerances for very small values
        assert_allclose(ds.rfldir, expected_rfldir, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.rfldn, expected_rfldn, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.flup, expected_flup, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.dfdt, expected_dfdt, rtol=1e-3, atol=1e-9)


class TestDisort06:
    """
    Test Problem 6: No Scattering, Increasingly Complex Sources

    Single layer, no scattering (ssalb=0). Starting from a simple beam source
    in a transparent medium, sub-cases progressively add optical thickness, surface
    reflection (Lambertian and Hapke BRDF), and thermal emission (boundary and
    internal). 8 sub-cases total.
    """

    _TEST_06_PARAMS = {
        # Case 6a: transparent medium, beam source, no reflection
        "a": {
            "lamber": True,
            "brdf_type": BRDFType.NONE,
            "planck": False,
            "ntau": 2,
            "dtauc": [0.0],
            "utau": [0.0, 0.0],
            "albedo": 0.0,
            "fisot": 0.0,
            "ttemp": None,
            "btemp": None,
            "temis": None,
            "temper": None,
            "wvnmlo": None,
            "wvnmhi": None,
            "expected_rfldir": [100.0, 100.0],
            "expected_rfldn": [0.0, 0.0],
            "expected_flup": [0.0, 0.0],
            "expected_dfdt": [200.0, 200.0],
            "expected_uu": np.zeros((4, 2, 1)),
        },
        # Case 6b: add optical thickness, beam source, no reflection
        "b": {
            "lamber": True,
            "brdf_type": BRDFType.NONE,
            "planck": False,
            "ntau": 3,
            "dtauc": [1.0],
            "utau": [0.0, 0.5, 1.0],
            "albedo": 0.0,
            "fisot": 0.0,
            "ttemp": None,
            "btemp": None,
            "temis": None,
            "temper": None,
            "wvnmlo": None,
            "wvnmhi": None,
            "expected_rfldir": [100.0, 3.67879e1, 1.35335e1],
            "expected_rfldn": [0.0, 0.0, 0.0],
            "expected_flup": [0.0, 0.0, 0.0],
            "expected_dfdt": [200.0, 7.35759e1, 2.70671e1],
            "expected_uu": np.zeros((4, 3, 1)),
        },
        # Case 6c: Lambertian surface reflection (albedo=0.5)
        "c": {
            "lamber": True,
            "brdf_type": BRDFType.NONE,
            "planck": False,
            "ntau": 3,
            "dtauc": [1.0],
            "utau": [0.0, 0.5, 1.0],
            "albedo": 0.5,
            "fisot": 0.0,
            "ttemp": None,
            "btemp": None,
            "temis": None,
            "temper": None,
            "wvnmlo": None,
            "wvnmhi": None,
            "expected_rfldir": [100.0, 3.67879e1, 1.35335e1],
            "expected_rfldn": [0.0, 0.0, 0.0],
            "expected_flup": [1.48450, 2.99914, 6.76676],
            "expected_dfdt": [2.02010e2, 7.79962e1, 4.06006e1],
            "expected_uu": np.array(
                [
                    [[0.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0]],
                    [[9.77882e-5], [1.45131e-2], [2.15393]],
                    [[7.92386e-1], [1.30642], [2.15393]],
                ]
            ),
        },
        # Case 6d: non-Lambertian Hapke surface, no thermal emission
        "d": {
            "lamber": False,
            "brdf_type": BRDFType.HAPKE,
            "planck": False,
            "ntau": 3,
            "dtauc": [1.0],
            "utau": [0.0, 0.5, 1.0],
            "albedo": 0.0,
            "fisot": 0.0,
            "ttemp": None,
            "btemp": None,
            "temis": None,
            "temper": None,
            "wvnmlo": None,
            "wvnmhi": None,
            "expected_rfldir": [100.0, 3.67879e1, 1.35335e1],
            "expected_rfldn": [0.0, 0.0, 0.0],
            "expected_flup": [6.70783e-1, 1.39084, 3.31655],
            "expected_dfdt": [2.00936e2, 7.57187e1, 3.45317e1],
            "expected_uu": np.array(
                [
                    [[0.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0]],
                    [[6.80068e-5], [1.00931e-2], [1.49795]],
                    [[3.15441e-1], [5.20074e-1], [8.57458e-1]],
                ]
            ),
        },
        # Case 6e: add bottom-boundary thermal emission (btemp=300 K)
        "e": {
            "lamber": False,
            "brdf_type": BRDFType.HAPKE,
            "planck": True,
            "ntau": 3,
            "dtauc": [1.0],
            "utau": [0.0, 0.5, 1.0],
            "albedo": 0.0,
            "fisot": 0.0,
            "ttemp": 0.0,
            "btemp": 300.0,
            "temis": 1.0,
            "temper": [0.0, 0.0],
            "wvnmlo": 0.0,
            "wvnmhi": 50000.0,
            "expected_rfldir": [100.0, 3.67879e1, 1.35335e1],
            "expected_rfldn": [0.0, 0.0, 0.0],
            "expected_flup": [7.95458e1, 1.59902e2, 3.56410e2],
            "expected_dfdt": [3.07079e2, 3.07108e2, 7.17467e2],
            "expected_uu": np.array(
                [
                    [[0.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0]],
                    [[4.53789e-3], [6.73483e-1], [9.99537e1]],
                    [[4.33773e1], [7.15170e1], [1.17912e2]],
                ]
            ),
        },
        # Case 6f: add top-boundary diffuse incidence and top emission
        "f": {
            "lamber": False,
            "brdf_type": BRDFType.HAPKE,
            "planck": True,
            "ntau": 3,
            "dtauc": [1.0],
            "utau": [0.0, 0.5, 1.0],
            "albedo": 0.0,
            "fisot": 100.0 / np.pi,
            "ttemp": 250.0,
            "btemp": 300.0,
            "temis": 1.0,
            "temper": [0.0, 0.0],
            "wvnmlo": 0.0,
            "wvnmhi": 50000.0,
            "expected_rfldir": [100.0, 3.67879e1, 1.35335e1],
            "expected_rfldn": [3.21497e2, 1.42493e2, 7.05305e1],
            "expected_flup": [8.27917e1, 1.66532e2, 3.71743e2],
            "expected_dfdt": [9.54523e2, 5.27085e2, 8.45341e2],
            "expected_uu": np.array(
                [
                    [[1.02336e2], [6.20697e1], [3.76472e1]],
                    [[1.02336e2], [6.89532e-1], [4.64603e-3]],
                    [[4.80531e-3], [7.13172e-1], [1.05844e2]],
                    [[4.50168e1], [7.42191e1], [1.22368e2]],
                ]
            ),
        },
        # Case 6g: add internal thermal emission (layer temperatures 250->300 K)
        "g": {
            "lamber": False,
            "brdf_type": BRDFType.HAPKE,
            "planck": True,
            "ntau": 3,
            "dtauc": [1.0],
            "utau": [0.0, 0.5, 1.0],
            "albedo": 0.0,
            "fisot": 100.0 / np.pi,
            "ttemp": 250.0,
            "btemp": 300.0,
            "temis": 1.0,
            "temper": [250.0, 300.0],
            "wvnmlo": 0.0,
            "wvnmhi": 50000.0,
            "expected_rfldir": [100.0, 3.67879e1, 1.35335e1],
            "expected_rfldn": [3.21497e2, 3.04775e2, 3.63632e2],
            "expected_flup": [3.35292e2, 4.12540e2, 4.41125e2],
            "expected_dfdt": [5.80394e2, 1.27117e2, -1.68003e2],
            "expected_uu": np.array(
                [
                    [[1.02336e2], [9.78748e1], [1.10061e2]],
                    [[1.02336e2], [1.01048e2], [1.38631e2]],
                    [[7.80733e1], [1.15819e2], [1.38695e2]],
                    [[1.16430e2], [1.34966e2], [1.40974e2]],
                ]
            ),
        },
        # Case 6h: increase optical thickness to 10
        "h": {
            "lamber": False,
            "brdf_type": BRDFType.HAPKE,
            "planck": True,
            "ntau": 3,
            "dtauc": [10.0],
            "utau": [0.0, 1.0, 10.0],
            "albedo": 0.0,
            "fisot": 100.0 / np.pi,
            "ttemp": 250.0,
            "btemp": 300.0,
            "temis": 1.0,
            "temper": [250.0, 300.0],
            "wvnmlo": 0.0,
            "wvnmhi": 50000.0,
            "expected_rfldir": [100.0, 1.35335e1, 2.06115e-7],
            "expected_rfldn": [3.21497e2, 2.55455e2, 4.43444e2],
            "expected_flup": [2.37350e2, 2.61130e2, 4.55861e2],
            "expected_dfdt": [4.23780e2, 6.19828e1, -3.11658e1],
            "expected_uu": np.array(
                [
                    [[1.02336e2], [8.49992e1], [1.38631e2]],
                    [[1.02336e2], [7.73186e1], [1.45441e2]],
                    [[7.12616e1], [7.88310e1], [1.44792e2]],
                    [[7.80736e1], [8.56423e1], [1.45163e2]],
                ]
            ),
        },
    }

    @pytest.mark.parametrize(
        "case_id, lamber, brdf_type, planck, ntau, dtauc, utau, albedo, fisot, "
        "ttemp, btemp, temis, temper, wvnmlo, wvnmhi, "
        "expected_rfldir, expected_rfldn, expected_flup, expected_dfdt, expected_uu",
        [
            (
                k,
                v["lamber"],
                v["brdf_type"],
                v["planck"],
                v["ntau"],
                v["dtauc"],
                v["utau"],
                v["albedo"],
                v["fisot"],
                v["ttemp"],
                v["btemp"],
                v["temis"],
                v["temper"],
                v["wvnmlo"],
                v["wvnmhi"],
                v["expected_rfldir"],
                v["expected_rfldn"],
                v["expected_flup"],
                v["expected_dfdt"],
                v["expected_uu"],
            )
            for k, v in _TEST_06_PARAMS.items()
        ],
        ids=list(_TEST_06_PARAMS.keys()),
    )
    def test(
        self,
        case_id,
        lamber,
        brdf_type,
        planck,
        ntau,
        dtauc,
        utau,
        albedo,
        fisot,
        ttemp,
        btemp,
        temis,
        temper,
        wvnmlo,
        wvnmhi,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
        expected_uu,
    ):
        """
        Run a single test case for Test 6.

        Parameters
        ----------
        case_id : str
            Subcase ID.
        """
        ds = DisortState()

        # Set dimensions
        ds.nstr = 16
        ds.nlyr = 1
        ds.nmom = 0
        ds.ntau = ntau
        ds.numu = 4
        ds.nphi = 1

        # Set flags
        ds.usrtau = True
        ds.usrang = True
        ds.lamber = lamber
        ds.planck = planck
        ds.onlyfl = False
        ds.quiet = True
        ds.intensity_correction = True
        ds.old_intensity_correction = True
        ds.spher = False
        ds.brdf_type = brdf_type

        if planck:
            ds.wvnmlo = wvnmlo
            ds.wvnmhi = wvnmhi

        # Allocate memory
        ds.allocate()

        # Set optical properties (no scattering); pmom shape must be (nstr+1, nlyr)
        ds.dtauc = np.array(dtauc)
        ds.ssalb = np.array([0.0])
        ds.pmom = pf.isotropic(ds.nstr).reshape(-1, 1)

        # Set output optical thicknesses and angles
        ds.utau = np.array(utau)
        ds.umu = np.array([-1.0, -0.1, 0.1, 1.0])
        ds.phi = np.array([90.0])

        # Set boundary conditions
        ds.fbeam = 200.0
        ds.umu0 = 0.5
        ds.phi0 = 0.0
        ds.fisot = fisot
        ds.fluor = 0.0
        ds.albedo = albedo

        if planck:
            ds.ttemp = ttemp
            ds.btemp = btemp
            ds.temis = temis
            ds.temper = np.array(temper)

        # Run solver
        ds.solve()

        print(f"\nTest 6{case_id}")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        assert_allclose(ds.rfldir, expected_rfldir, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.rfldn, expected_rfldn, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.flup, expected_flup, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.dfdt, expected_dfdt, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.uu, expected_uu, rtol=1e-3, atol=1e-9)


class TestDisort07:
    """
    Test Problem 7: Absorption + Scattering + All Possible Sources, Various
    Surface Reflectivities (One Layer)

    Compare 7a fluxes to Ref. KS (Kylling & Stamnes), Table I, and 7b to
    Table II. Sub-cases progressively vary the surface treatment from black to
    Lambertian to non-Lambertian Hapke BRDF, with all source types active.
    """

    def _setup_common_7cde(self, ds: DisortState) -> None:
        """Configure geometry shared by sub-cases 7c, 7d, and 7e."""
        ds.nstr = 12
        ds.nlyr = 1
        ds.nmom = 12
        ds.ntau = 3
        ds.numu = 4
        ds.nphi = 2
        ds.usrtau = True
        ds.usrang = True
        ds.planck = True
        ds.onlyfl = False
        ds.quiet = True
        ds.intensity_correction = True
        ds.old_intensity_correction = True
        ds.spher = False
        ds.wvnmlo = 0.0
        ds.wvnmhi = 50000.0

    def test_7a(self):
        """
        7a: Absorption + scattering, internal thermal sources only (flux only).

        Compare to Ref. KS, Table I (tau=1.0, a=0.1, g=0.05).
        """
        ds = DisortState()
        ds.nstr = 16
        ds.nlyr = 1
        ds.nmom = 16
        ds.ntau = 2
        ds.numu = 2
        ds.nphi = 1
        ds.usrtau = True
        ds.usrang = False
        ds.lamber = True
        ds.planck = True
        ds.onlyfl = True
        ds.quiet = True
        ds.intensity_correction = True
        ds.old_intensity_correction = True
        ds.spher = False
        ds.wvnmlo = 300.0
        ds.wvnmhi = 800.0

        ds.allocate()

        ds.dtauc = np.array([1.0])
        ds.ssalb = np.array([0.1])
        ds.pmom = pf.henyey_greenstein(0.05, 16).reshape(-1, 1)
        ds.utau = np.array([0.0, 1.0])
        ds.umu = np.array([-1.0, 1.0])
        ds.phi = np.array([0.0])
        ds.temper = np.array([200.0, 300.0])

        ds.fbeam = 0.0
        ds.umu0 = 0.5
        ds.phi0 = 0.0
        ds.fisot = 0.0
        ds.fluor = 0.0
        ds.albedo = 0.0
        ds.ttemp = 0.0
        ds.btemp = 0.0
        ds.temis = 1.0

        ds.solve()

        print("\nTest 7a: KS Table I, tau=1.0, a=0.1, g=0.05")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        assert_allclose(ds.rfldir, [0.0, 0.0], rtol=1e-3, atol=1e-9)
        assert_allclose(ds.rfldn, [0.0, 1.21204e2], rtol=1e-3, atol=1e-9)
        assert_allclose(ds.flup, [8.62936e1, 0.0], rtol=1e-3, atol=1e-9)
        assert_allclose(ds.dfdt, [-5.13731e1, -5.41036e2], rtol=1e-3, atol=1e-9)

    def test_7b(self):
        """
        7b: Very thick layer, narrow spectral band.

        Compare to Ref. KS, Table II (tau=100, a=0.95, g=0.75).
        """
        ds = DisortState()
        ds.nstr = 16
        ds.nlyr = 1
        ds.nmom = 16
        ds.ntau = 2
        ds.numu = 2
        ds.nphi = 1
        ds.usrtau = True
        ds.usrang = True
        ds.lamber = True
        ds.planck = True
        ds.onlyfl = False
        ds.quiet = True
        ds.intensity_correction = True
        ds.old_intensity_correction = True
        ds.spher = False
        ds.wvnmlo = 2702.99
        ds.wvnmhi = 2703.01

        ds.allocate()

        ds.dtauc = np.array([100.0])
        ds.ssalb = np.array([0.95])
        ds.pmom = pf.henyey_greenstein(0.75, 16).reshape(-1, 1)
        ds.utau = np.array([0.0, 100.0])
        ds.umu = np.array([-1.0, 1.0])
        ds.phi = np.array([0.0])
        ds.temper = np.array([200.0, 300.0])

        ds.fbeam = 0.0
        ds.umu0 = 0.5
        ds.phi0 = 0.0
        ds.fisot = 0.0
        ds.fluor = 0.0
        ds.albedo = 0.0
        ds.ttemp = 0.0
        ds.btemp = 0.0
        ds.temis = 1.0

        ds.solve()

        print("\nTest 7b: KS Table II, tau=100, a=0.95, g=0.75")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")
        print(f"  uu:     {ds.uu}")

        assert_allclose(ds.rfldir, [0.0, 0.0], rtol=1e-3, atol=1e-9)
        assert_allclose(ds.rfldn, [0.0, 2.07786e-5], rtol=1e-3, atol=1e-9)
        assert_allclose(ds.flup, [1.10949e-6, 0.0], rtol=1e-3, atol=1e-9)
        assert_allclose(ds.dfdt, [8.23219e-8, -5.06461e-6], rtol=1e-3, atol=1e-9)
        # uu shape: (numu=2, ntau=2, nphi=1)
        expected_uu = np.array(
            [
                [[0.0], [7.52311e-6]],
                [[4.65744e-7], [0.0]],
            ]
        )
        assert_allclose(ds.uu, expected_uu, rtol=1e-3, atol=1e-12)

    def test_7c(self):
        """7c: All sources, black surface (albedo=0)."""
        ds = DisortState()
        self._setup_common_7cde(ds)
        ds.lamber = True

        ds.allocate()

        ds.dtauc = np.array([1.0])
        ds.ssalb = np.array([0.5])
        ds.pmom = pf.henyey_greenstein(0.8, 12).reshape(-1, 1)
        ds.utau = np.array([0.0, 0.5, 1.0])
        ds.umu = np.array([-1.0, -0.1, 0.1, 1.0])
        ds.phi = np.array([0.0, 90.0])
        ds.temper = np.array([300.0, 200.0])

        ds.fbeam = 200.0
        ds.umu0 = 0.5
        ds.phi0 = 0.0
        ds.fisot = 100.0
        ds.fluor = 0.0
        ds.albedo = 0.0
        ds.ttemp = 100.0
        ds.btemp = 320.0
        ds.temis = 1.0

        ds.solve()

        print("\nTest 7c: All sources, albedo=0")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        assert_allclose(ds.rfldir, [100.0, 3.67879e1, 1.35335e1], rtol=1e-3, atol=1e-9)
        assert_allclose(
            ds.rfldn, [3.19830e2, 3.54099e2, 3.01334e2], rtol=1e-3, atol=1e-9
        )
        assert_allclose(
            ds.flup, [4.29572e2, 4.47018e2, 5.94576e2], rtol=1e-3, atol=1e-9
        )
        assert_allclose(
            ds.dfdt, [-8.04270e1, 2.51589e2, 7.15964e2], rtol=1e-3, atol=1e-9
        )

        # uu shape: (numu=4, ntau=3, nphi=2); phi order: [0, 90]
        expected_uu = np.array(
            [
                [
                    [1.01805e2, 1.01805e2],
                    [1.06583e2, 1.06583e2],
                    [9.66519e1, 9.66519e1],
                ],
                [
                    [1.01805e2, 1.01805e2],
                    [1.28565e2, 1.06408e2],
                    [8.65854e1, 7.49310e1],
                ],
                [
                    [1.46775e2, 1.29641e2],
                    [1.04464e2, 9.48418e1],
                    [1.89259e2, 1.89259e2],
                ],
                [
                    [1.49033e2, 1.49033e2],
                    [1.59054e2, 1.59054e2],
                    [1.89259e2, 1.89259e2],
                ],
            ]
        )
        assert_allclose(ds.uu, expected_uu, rtol=1e-3, atol=1e-9)

    def test_7d(self):
        """7d: All sources, white Lambertian surface (albedo=1)."""
        ds = DisortState()
        self._setup_common_7cde(ds)
        ds.lamber = True

        ds.allocate()

        ds.dtauc = np.array([1.0])
        ds.ssalb = np.array([0.5])
        ds.pmom = pf.henyey_greenstein(0.8, 12).reshape(-1, 1)
        ds.utau = np.array([0.0, 0.5, 1.0])
        ds.umu = np.array([-1.0, -0.1, 0.1, 1.0])
        ds.phi = np.array([0.0, 90.0])
        ds.temper = np.array([300.0, 200.0])

        ds.fbeam = 200.0
        ds.umu0 = 0.5
        ds.phi0 = 0.0
        ds.fisot = 100.0
        ds.fluor = 0.0
        ds.albedo = 1.0
        ds.ttemp = 100.0
        ds.btemp = 320.0
        ds.temis = 1.0

        ds.solve()

        print("\nTest 7d: All sources, albedo=1 (Lambertian)")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        assert_allclose(ds.rfldir, [100.0, 3.67879e1, 1.35335e1], rtol=1e-3, atol=1e-9)
        assert_allclose(
            ds.rfldn, [3.19830e2, 3.50555e2, 2.92063e2], rtol=1e-3, atol=1e-9
        )
        assert_allclose(
            ds.flup, [3.12563e2, 2.68126e2, 3.05596e2], rtol=1e-3, atol=1e-9
        )
        assert_allclose(
            ds.dfdt, [-1.68356e2, 1.01251e2, 4.09326e2], rtol=1e-3, atol=1e-9
        )

        expected_uu = np.array(
            [
                [
                    [1.01805e2, 1.01805e2],
                    [1.06203e2, 1.06203e2],
                    [9.56010e1, 9.56010e1],
                ],
                [
                    [1.01805e2, 1.01805e2],
                    [1.23126e2, 1.00969e2],
                    [7.25576e1, 6.09031e1],
                ],
                [
                    [1.40977e2, 1.23843e2],
                    [9.19545e1, 8.23318e1],
                    [9.72743e1, 9.72743e1],
                ],
                [
                    [9.62764e1, 9.62764e1],
                    [8.89528e1, 8.89528e1],
                    [9.72743e1, 9.72743e1],
                ],
            ]
        )
        assert_allclose(ds.uu, expected_uu, rtol=1e-3, atol=1e-9)

    def test_7e(self):
        """7e: All sources, non-Lambertian Hapke surface."""
        ds = DisortState()
        self._setup_common_7cde(ds)
        ds.lamber = False
        ds.brdf_type = BRDFType.HAPKE

        ds.allocate()

        ds.dtauc = np.array([1.0])
        ds.ssalb = np.array([0.5])
        ds.pmom = pf.henyey_greenstein(0.8, 12).reshape(-1, 1)
        ds.utau = np.array([0.0, 0.5, 1.0])
        ds.umu = np.array([-1.0, -0.1, 0.1, 1.0])
        ds.phi = np.array([0.0, 90.0])
        ds.temper = np.array([300.0, 200.0])

        ds.fbeam = 200.0
        ds.umu0 = 0.5
        ds.phi0 = 0.0
        ds.fisot = 100.0
        ds.fluor = 0.0
        ds.ttemp = 100.0
        ds.btemp = 320.0
        ds.temis = 1.0

        ds.solve()

        print("\nTest 7e: All sources, Hapke BRDF")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        assert_allclose(ds.rfldir, [100.0, 3.67879e1, 1.35335e1], rtol=1e-3, atol=1e-9)
        assert_allclose(
            ds.rfldn, [3.19830e2, 3.53275e2, 2.99002e2], rtol=1e-3, atol=1e-9
        )
        assert_allclose(
            ds.flup, [4.04300e2, 4.07843e2, 5.29248e2], rtol=1e-3, atol=1e-9
        )
        assert_allclose(
            ds.dfdt, [-9.98568e1, 2.17387e2, 6.38461e2], rtol=1e-3, atol=1e-9
        )

        expected_uu = np.array(
            [
                [
                    [1.01805e2, 1.01805e2],
                    [1.06496e2, 1.06496e2],
                    [9.63993e1, 9.63993e1],
                ],
                [
                    [1.01805e2, 1.01805e2],
                    [1.27296e2, 1.05111e2],
                    [8.29009e1, 7.11248e1],
                ],
                [
                    [1.45448e2, 1.28281e2],
                    [1.01395e2, 9.16726e1],
                    [1.60734e2, 1.59286e2],
                ],
                [
                    [1.38554e2, 1.38554e2],
                    [1.45229e2, 1.45229e2],
                    [1.71307e2, 1.71307e2],
                ],
            ]
        )
        assert_allclose(ds.uu, expected_uu, rtol=1e-3, atol=1e-9)


class TestDisort08:
    """
    Test Problem 8: Absorbing/Isotropic-Scattering Medium, Two Layers

    Compare to Ref. OS (Stamnes & Tsay), Table 1

    This test examines a two-layer medium with:
    - Isotropic scattering in both layers
    - Different optical properties in each layer
    - Isotropic incident radiation (fisot) rather than beam source
    """

    _TEST_08_PARAMS = {
        # Case 8a: 2-layer, dtauc=[0.25,0.25], ssalb=[0.5,0.3] (OS Table 1, Line 4)
        "a": {
            "dtauc": [0.25, 0.25],
            "ssalb": [0.5, 0.3],
            "utau": [0.0, 0.25, 0.5],
            "expected_rfldir": [0.0, 0.0, 0.0],
            "expected_rfldn": [1.0, 7.22235e-01, 5.13132e-01],
            "expected_flup": [9.29633e-02, 2.78952e-02, 0.0],
            "expected_dfdt": [1.12474, 6.51821e-01, 5.63361e-01],
        },
        # Case 8b: 2-layer, dtauc=[0.25,0.25], ssalb=[0.8,0.95] (OS Table 1, Line 1)
        "b": {
            "dtauc": [0.25, 0.25],
            "ssalb": [0.8, 0.95],
            "utau": [0.0, 0.25, 0.5],
            "expected_rfldir": [0.0, 0.0, 0.0],
            "expected_rfldn": [1.0, 7.95332e-01, 6.50417e-01],
            "expected_flup": [2.25136e-01, 1.26349e-01, 0.0],
            "expected_dfdt": [5.12692e-01, 3.56655e-01, 5.68095e-02],
        },
        # Case 8c: 2-layer, dtauc=[1.0,2.0], ssalb=[0.8,0.95] (OS Table 1, Line 13)
        "c": {
            "dtauc": [1.0, 2.0],
            "ssalb": [0.8, 0.95],
            "utau": [0.0, 1.0, 3.0],
            "expected_rfldir": [0.0, 0.0, 0.0],
            "expected_rfldn": [1.0, 4.86157e-01, 1.59984e-01],
            "expected_flup": [3.78578e-01, 2.43397e-01, 0.0],
            "expected_dfdt": [5.65095e-01, 2.76697e-01, 1.35679e-02],
        },
    }

    @pytest.mark.parametrize(
        "case_id, dtauc, ssalb, utau, expected_rfldir, expected_rfldn, "
        "expected_flup, expected_dfdt,",
        [
            (
                k,  # case_id
                v["dtauc"],
                v["ssalb"],
                v["utau"],
                v["expected_rfldir"],
                v["expected_rfldn"],
                v["expected_flup"],
                v["expected_dfdt"],
            )
            for k, v in _TEST_08_PARAMS.items()
        ],
        ids=list(_TEST_08_PARAMS.keys()),
    )
    def test(
        self,
        case_id,
        dtauc,
        ssalb,
        utau,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
    ):
        """
        Run a single test case for Test 8.

        Parameters
        ----------
        case_id : str
            Subcase ID.

        dtauc : list
            Optical thickness of each layer.

        ssalb : list
            Single scattering albedo of each layer.

        utau : list
            Output optical thickness levels.

        expected_* : list
            Expected flux values at specified optical thicknesses.
        """
        ds = DisortState()

        # Set dimensions
        ds.nstr = 8
        ds.nlyr = 2
        ds.nmom = 8
        ds.ntau = 3
        ds.numu = 4
        ds.nphi = 1

        # Set flags
        ds.usrtau = True
        ds.usrang = True
        ds.lamber = True
        ds.planck = False
        ds.onlyfl = False
        ds.quiet = True

        # Allocate memory
        ds.allocate()

        # Set optical properties - isotropic scattering for both layers
        ds.dtauc = np.array(dtauc)
        ds.ssalb = np.array(ssalb)

        # Isotropic phase function for both layers
        pmom = pf.isotropic(ds.nmom).reshape(-1, 1)
        ds.pmom = np.tile(pmom, (1, ds.nlyr))

        # Set output optical thicknesses
        ds.utau = np.array(utau)

        # Set angles
        ds.umu = np.array([-1.0, -0.2, 0.2, 1.0])
        ds.phi = np.array([60.0])

        # Set source - isotropic incident radiation (no beam)
        ds.fbeam = 0.0
        ds.fisot = 1.0 / np.pi
        ds.umu0 = 0.5  # Set even when fbeam=0 (matches C code)
        ds.phi0 = 0.0

        # Set boundary conditions
        ds.albedo = 0.0
        ds.fluor = 0.0

        # Run solver
        ds.solve()

        # Check outputs
        print(f"\nTest 8{case_id}: 2-layer, Isotropic")
        print(f"  dtauc: {dtauc}, ssalb: {ssalb}")
        print(f"  utau: {utau}")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        # Compare with expected values
        assert_allclose(ds.rfldir, expected_rfldir, rtol=1e-4, atol=1e-9)
        assert_allclose(ds.rfldn, expected_rfldn, rtol=1e-4, atol=1e-9)
        assert_allclose(ds.flup, expected_flup, rtol=1e-4, atol=1e-9)
        assert_allclose(ds.dfdt, expected_dfdt, rtol=1e-4, atol=1e-9)
