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

Note: Some tests check intensity fields (uu) which are not yet exposed in
the Python bindings, so those assertions are omitted for now.
"""

import numpy as np
from numpy.testing import assert_allclose

from nanodisort import DisortState
from nanodisort.utils import phase_functions as pf


class TestDisort01:
    """
    Test Problem 1: Isotropic Scattering

    Compare to Ref. VH1, Table 12

    This test examines isotropic scattering with various combinations of:
    - Optical depths: 0.03125, 32.0
    - Single scattering albedos: 0.2, 0.99, 1.0
    - Sources: beam, isotropic
    """

    def run_case_01(
        self,
        case_idx,
        utau2,
        ssalb,
        fbeam,
        fisot,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
    ):
        """
        Helper to run a single test case for Test 1.

        Parameters
        ----------
        case_idx : int
            Case index (for labeling)
        utau2 : float
            Bottom boundary optical depth
        ssalb : float
            Single scattering albedo
        fbeam : float
            Incident beam flux
        fisot : float
            Isotropic top illumination
        expected_* : list
            Expected values at [top, bottom]
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

        # Allocate memory
        ds.allocate()

        # Set optical properties - isotropic phase function
        ds.dtauc = np.array([utau2])
        ds.ssalb = np.array([ssalb])
        pmom = pf.isotropic(ds.nmom).reshape(-1, 1)
        ds.pmom = pmom

        # Set output optical depths
        ds.utau = np.array([0.0, utau2])

        # Set angles
        ds.umu = np.array([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
        ds.phi = np.array([0.0])

        # Set beam parameters
        ds.fbeam = fbeam
        ds.umu0 = 0.1
        ds.phi0 = 0.0

        # Set boundary conditions
        ds.albedo = 0.0
        ds.fisot = fisot
        ds.fluor = 0.0

        # Run solver
        ds.solve()

        # Check flux outputs
        print(
            f"\nTest 1{chr(ord('a') + case_idx - 1)}: "
            f"tau={utau2}, ssalb={ssalb}, beam={fbeam}, isot={fisot}"
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

    def test_case_1a(self):
        """Case 1a: tau=0.03125, ssalb=0.2, beam source"""
        self.run_case_01(
            case_idx=1,
            utau2=0.03125,
            ssalb=0.2,
            fbeam=np.pi / 0.1,  # pi/umu0
            fisot=0.0,
            expected_rfldir=[3.14159, 2.29844],
            expected_rfldn=[0.0, 7.94108e-02],
            expected_flup=[7.99451e-02, 0.0],
            expected_dfdt=[2.54067e01, 1.86531e01],
        )

    def test_case_1b(self):
        """Case 1b: tau=0.03125, ssalb=1.0, beam source"""
        self.run_case_01(
            case_idx=2,
            utau2=0.03125,
            ssalb=1.0,
            fbeam=np.pi / 0.1,
            fisot=0.0,
            expected_rfldir=[3.14159, 2.29844],
            expected_rfldn=[0.0, 4.20233e-01],
            expected_flup=[4.22922e-01, 0.0],
            expected_dfdt=[0.0, 0.0],
        )

    def test_case_1c(self):
        """Case 1c: tau=0.03125, ssalb=0.99, isotropic source"""
        self.run_case_01(
            case_idx=3,
            utau2=0.03125,
            ssalb=0.99,
            fbeam=0.0,
            fisot=1.0,
            expected_rfldir=[0.0, 0.0],
            expected_rfldn=[3.14159, 3.04897],
            expected_flup=[9.06556e-02, 0.0],
            expected_dfdt=[6.66870e-02, 5.88936e-02],
        )

    def test_case_1d(self):
        """Case 1d: tau=32.0, ssalb=0.2, beam source"""
        self.run_case_01(
            case_idx=4,
            utau2=32.0,
            ssalb=0.2,
            fbeam=np.pi / 0.1,
            fisot=0.0,
            expected_rfldir=[3.14159, 0.0],
            expected_rfldn=[0.0, 0.0],
            expected_flup=[2.59686e-01, 0.0],
            expected_dfdt=[2.57766e01, 0.0],
        )

    def test_case_1e(self):
        """Case 1e: tau=32.0, ssalb=1.0, beam source"""
        self.run_case_01(
            case_idx=5,
            utau2=32.0,
            ssalb=1.0,
            fbeam=np.pi / 0.1,
            fisot=0.0,
            expected_rfldir=[3.14159, 0.0],
            expected_rfldn=[0.0, 6.76954e-02],
            expected_flup=[3.07390, 0.0],
            expected_dfdt=[0.0, 0.0],
        )

    def test_case_1f(self):
        """Case 1f: tau=32.0, ssalb=0.99, isotropic source"""
        self.run_case_01(
            case_idx=6,
            utau2=32.0,
            ssalb=0.99,
            fbeam=0.0,
            fisot=1.0,
            expected_rfldir=[0.0, 0.0],
            expected_rfldn=[3.14159, 4.60048e-03],
            expected_flup=[2.49618, 0.0],
            expected_dfdt=[1.14239e-01, 7.93633e-05],
        )


class TestDisort02:
    """
    Test Problem 2: Rayleigh Scattering, Beam Source

    Compare to Ref. SW, Table 1

    This test examines Rayleigh scattering with:
    - Optical depths: 0.2, 5.0
    - Single scattering albedos: 0.5, 1.0
    """

    def run_case_02(
        self,
        case_idx,
        utau2,
        ssalb,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
    ):
        """Helper to run a single test case for Test 2."""
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

        # Allocate memory
        ds.allocate()

        # Set optical properties - Rayleigh phase function
        ds.dtauc = np.array([utau2])
        ds.ssalb = np.array([ssalb])
        pmom = pf.rayleigh(ds.nmom).reshape(-1, 1)
        ds.pmom = pmom

        # Set output optical depths
        ds.utau = np.array([0.0, utau2])

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
        label = chr(ord("a") + case_idx - 1)
        print(f"\nTest 2{label}: tau={utau2}, ssalb={ssalb}")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        # Compare with expected values
        assert_allclose(ds.rfldir, expected_rfldir, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.rfldn, expected_rfldn, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.flup, expected_flup, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.dfdt, expected_dfdt, rtol=1e-3, atol=1e-9)

    def test_case_2a(self):
        """Case 2a: tau=0.2, ssalb=0.5"""
        self.run_case_02(
            case_idx=1,
            utau2=0.2,
            ssalb=0.5,
            expected_rfldir=[2.52716e-01, 2.10311e-02],
            expected_rfldn=[0.0, 4.41791e-02],
            expected_flup=[5.35063e-02, 0.0],
            expected_dfdt=[1.66570e00, 1.89848e-01],
        )

    def test_case_2b(self):
        """Case 2b: tau=0.2, ssalb=1.0"""
        self.run_case_02(
            case_idx=2,
            utau2=0.2,
            ssalb=1.0,
            expected_rfldir=[2.52716e-01, 2.10311e-02],
            expected_rfldn=[0.0, 1.06123e-01],
            expected_flup=[1.25561e-01, 0.0],
            expected_dfdt=[0.0, 0.0],
        )

    def test_case_2c(self):
        """Case 2c: tau=5.0, ssalb=0.5"""
        self.run_case_02(
            case_idx=3,
            utau2=5.0,
            ssalb=0.5,
            expected_rfldir=[2.52716e-01, 2.56077e-28],
            expected_rfldn=[0.0, 2.51683e-04],
            expected_flup=[6.24730e-02, 0.0],
            expected_dfdt=[1.67462e00, 1.75464e-04],
        )

    def test_case_2d(self):
        """Case 2d: tau=5.0, ssalb=1.0"""
        self.run_case_02(
            case_idx=4,
            utau2=5.0,
            ssalb=1.0,
            expected_rfldir=[2.52716e-01, 0.0],
            expected_rfldn=[0.0, 2.68008e-02],
            expected_flup=[2.25915e-01, 0.0],
            expected_dfdt=[0.0, 0.0],
        )


class TestDisort03:
    """
    Test Problem 3: Henyey-Greenstein Scattering

    Compare to Ref. VH2, Table 37

    This test examines Henyey-Greenstein phase function with asymmetry factor
    g=0.75, testing:
    - Optical depths: 1.0, 8.0
    - Single scattering albedo: 1.0
    - nmom=32 (more moments than nstr=16)
    """

    def run_case_03(
        self,
        case_idx,
        utau2,
        expected_rfldir,
        expected_rfldn,
        expected_flup,
        expected_dfdt,
    ):
        """Helper to run a single test case for Test 3."""
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

        # Allocate memory
        ds.allocate()

        # Set optical properties - Henyey-Greenstein with g=0.75
        ds.dtauc = np.array([utau2])
        ds.ssalb = np.array([1.0])
        pmom = pf.henyey_greenstein(0.75, ds.nmom).reshape(-1, 1)
        ds.pmom = pmom

        # Set output optical depths
        ds.utau = np.array([0.0, utau2])

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
        label = chr(ord("a") + case_idx - 1)
        print(f"\nTest 3{label}: tau={utau2}, HG(g=0.75)")
        print(f"  rfldir: {ds.rfldir}")
        print(f"  rfldn:  {ds.rfldn}")
        print(f"  flup:   {ds.flup}")
        print(f"  dfdt:   {ds.dfdt}")

        # Compare with expected values
        assert_allclose(ds.rfldir, expected_rfldir, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.rfldn, expected_rfldn, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.flup, expected_flup, rtol=1e-3, atol=1e-9)
        assert_allclose(ds.dfdt, expected_dfdt, rtol=1e-3, atol=1e-9)

    def test_case_3a(self):
        """Case 3a: tau=1.0, HG(g=0.75), ssalb=1.0"""
        self.run_case_03(
            case_idx=1,
            utau2=1.0,
            expected_rfldir=[3.14159, 1.15573],
            expected_rfldn=[0.0, 1.73849],
            expected_flup=[2.47374e-01, 0.0],
            expected_dfdt=[0.0, 0.0],
        )

    def test_case_3b(self):
        """Case 3b: tau=8.0, HG(g=0.75), ssalb=1.0"""
        self.run_case_03(
            case_idx=2,
            utau2=8.0,
            expected_rfldir=[3.14159, 1.05389e-03],
            expected_rfldn=[0.0, 1.54958],
            expected_flup=[1.59096, 0.0],
            expected_dfdt=[0.0, 0.0],
        )
