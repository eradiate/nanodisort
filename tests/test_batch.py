"""
Tests for BatchSolver parallel spectral solving.

Validates that batch solving produces identical results to sequential
DisortState solving, and that input validation works correctly.
"""

import numpy as np
import pytest

from nanodisort import BatchSolver, DisortState


def make_single_layer_config():
    """Return a dict with configuration for a single-layer test problem."""
    return {
        "nstr": 8,
        "nlyr": 1,
        "nmom": 8,
        "ntau": 2,
        "numu": 0,
        "nphi": 0,
        "usrtau": True,
        "usrang": False,
        "lamber": True,
        "planck": False,
        "onlyfl": True,
        "quiet": True,
        "umu0": 1.0,
        "phi0": 0.0,
        "fisot": 0.0,
        "fluor": 0.0,
        "albedo": 0.0,
        "fbeam": np.pi,
        "utau": np.array([0.0, 1.0]),
        # Optical properties (vary per batch item in batch tests)
        "dtauc": np.array([1.0]),
        "ssalb": np.array([0.9]),
        "pmom": None,  # Will be constructed based on nmom
    }


def solve_single(config, dtauc, ssalb, pmom, fbeam=np.pi, albedo=0.0):
    """Solve a single problem with DisortState and return flux outputs."""
    ds = DisortState()
    ds.nstr = config["nstr"]
    ds.nlyr = config["nlyr"]
    ds.nmom = config["nmom"]
    ds.ntau = config["ntau"]
    ds.numu = config["numu"]
    ds.nphi = config["nphi"]
    ds.usrtau = config["usrtau"]
    ds.usrang = config["usrang"]
    ds.lamber = config["lamber"]
    ds.planck = config["planck"]
    ds.onlyfl = config["onlyfl"]
    ds.quiet = config["quiet"]
    ds.umu0 = config["umu0"]
    ds.phi0 = config["phi0"]
    ds.fisot = config["fisot"]
    ds.fluor = config["fluor"]
    ds.fbeam = fbeam
    ds.albedo = albedo

    ds.allocate()
    ds.dtauc = dtauc
    ds.ssalb = ssalb
    ds.pmom = pmom
    ds.utau = config["utau"]

    ds.solve()
    return {
        "rfldir": np.array(ds.rfldir),
        "rfldn": np.array(ds.rfldn),
        "flup": np.array(ds.flup),
        "dfdt": np.array(ds.dfdt),
    }


class TestBatchMatchesSequential:
    """Verify that batch results match sequential single-solve results."""

    def test_varying_optical_properties(self):
        """Batch of N problems with different dtauc/ssalb matches N sequential solves."""
        config = make_single_layer_config()
        nbatch = 8

        # All dtauc values >= 1.0 so utau=[0.0, 1.0] is always valid
        dtauc_batch = np.linspace(1.0, 3.0, nbatch).reshape(nbatch, 1)
        ssalb_batch = np.linspace(0.5, 1.0, nbatch).reshape(nbatch, 1)
        fbeam_batch = np.full(nbatch, np.pi)
        albedo_batch = np.zeros(nbatch)

        nmom_nstr = max(config["nmom"], config["nstr"])
        # Isotropic phase function for all
        pmom_single = np.zeros((nmom_nstr + 1, config["nlyr"]))
        pmom_single[0, :] = 1.0

        # Batch pmom: (nmom_nstr+1, nlyr, nbatch) Fortran-contiguous
        pmom_batch = np.zeros(
            (nmom_nstr + 1, config["nlyr"], nbatch), order="F"
        )
        pmom_batch[0, :, :] = 1.0

        # Solve sequentially
        sequential_results = []
        for i in range(nbatch):
            result = solve_single(
                config,
                dtauc_batch[i],
                ssalb_batch[i],
                pmom_single,
                fbeam=fbeam_batch[i],
                albedo=albedo_batch[i],
            )
            sequential_results.append(result)

        # Solve with BatchSolver
        solver = BatchSolver(nthreads=4)
        solver.nstr = config["nstr"]
        solver.nlyr = config["nlyr"]
        solver.nmom = config["nmom"]
        solver.ntau = config["ntau"]
        solver.numu = config["numu"]
        solver.nphi = config["nphi"]
        solver.usrtau = config["usrtau"]
        solver.usrang = config["usrang"]
        solver.lamber = config["lamber"]
        solver.planck = config["planck"]
        solver.onlyfl = config["onlyfl"]
        solver.quiet = config["quiet"]
        solver.umu0 = config["umu0"]
        solver.phi0 = config["phi0"]
        solver.fisot = config["fisot"]
        solver.fluor = config["fluor"]
        solver.set_utau(config["utau"])

        solver.allocate(nbatch)
        solver.set_dtauc(dtauc_batch)
        solver.set_ssalb(ssalb_batch)
        solver.set_pmom(pmom_batch)
        solver.set_fbeam(fbeam_batch)
        solver.set_albedo(albedo_batch)

        solver.solve()

        # Compare outputs
        batch_rfldir = np.array(solver.rfldir)
        batch_rfldn = np.array(solver.rfldn)
        batch_flup = np.array(solver.flup)
        batch_dfdt = np.array(solver.dfdt)

        for i in range(nbatch):
            np.testing.assert_allclose(
                batch_rfldir[i], sequential_results[i]["rfldir"],
                rtol=1e-14,
                err_msg=f"rfldir mismatch at batch {i}",
            )
            np.testing.assert_allclose(
                batch_rfldn[i], sequential_results[i]["rfldn"],
                rtol=1e-14,
                err_msg=f"rfldn mismatch at batch {i}",
            )
            np.testing.assert_allclose(
                batch_flup[i], sequential_results[i]["flup"],
                rtol=1e-14,
                err_msg=f"flup mismatch at batch {i}",
            )
            np.testing.assert_allclose(
                batch_dfdt[i], sequential_results[i]["dfdt"],
                rtol=1e-14,
                err_msg=f"dfdt mismatch at batch {i}",
            )

    def test_varying_fbeam_and_albedo(self):
        """Batch with varying beam intensity and surface albedo."""
        config = make_single_layer_config()
        nbatch = 5

        dtauc_batch = np.ones((nbatch, 1))
        ssalb_batch = np.full((nbatch, 1), 0.8)
        fbeam_batch = np.linspace(1.0, 10.0, nbatch)
        albedo_batch = np.linspace(0.0, 0.5, nbatch)

        nmom_nstr = max(config["nmom"], config["nstr"])
        pmom_single = np.zeros((nmom_nstr + 1, 1))
        pmom_single[0, 0] = 1.0

        pmom_batch = np.zeros((nmom_nstr + 1, 1, nbatch), order="F")
        pmom_batch[0, :, :] = 1.0

        sequential_results = []
        for i in range(nbatch):
            result = solve_single(
                config, dtauc_batch[i], ssalb_batch[i], pmom_single,
                fbeam=fbeam_batch[i], albedo=albedo_batch[i],
            )
            sequential_results.append(result)

        solver = BatchSolver(nthreads=2)
        solver.nstr = config["nstr"]
        solver.nlyr = config["nlyr"]
        solver.nmom = config["nmom"]
        solver.ntau = config["ntau"]
        solver.numu = config["numu"]
        solver.nphi = config["nphi"]
        solver.usrtau = config["usrtau"]
        solver.usrang = config["usrang"]
        solver.lamber = config["lamber"]
        solver.planck = config["planck"]
        solver.onlyfl = config["onlyfl"]
        solver.quiet = config["quiet"]
        solver.umu0 = config["umu0"]
        solver.phi0 = config["phi0"]
        solver.fisot = config["fisot"]
        solver.fluor = config["fluor"]
        solver.set_utau(config["utau"])

        solver.allocate(nbatch)
        solver.set_dtauc(dtauc_batch)
        solver.set_ssalb(ssalb_batch)
        solver.set_pmom(pmom_batch)
        solver.set_fbeam(fbeam_batch)
        solver.set_albedo(albedo_batch)

        solver.solve()

        for i in range(nbatch):
            np.testing.assert_allclose(
                np.array(solver.rfldir)[i], sequential_results[i]["rfldir"],
                rtol=1e-14,
            )
            np.testing.assert_allclose(
                np.array(solver.flup)[i], sequential_results[i]["flup"],
                rtol=1e-14,
            )


class TestBatchIdenticalInputs:
    """Verify that N identical inputs produce N identical outputs."""

    def test_identical_outputs(self):
        config = make_single_layer_config()
        nbatch = 10

        dtauc_batch = np.ones((nbatch, 1)) * 1.0
        ssalb_batch = np.full((nbatch, 1), 0.95)
        fbeam_batch = np.full(nbatch, np.pi)
        albedo_batch = np.zeros(nbatch)

        nmom_nstr = max(config["nmom"], config["nstr"])
        pmom_batch = np.zeros((nmom_nstr + 1, 1, nbatch), order="F")
        pmom_batch[0, :, :] = 1.0

        solver = BatchSolver(nthreads=4)
        solver.nstr = config["nstr"]
        solver.nlyr = config["nlyr"]
        solver.nmom = config["nmom"]
        solver.ntau = config["ntau"]
        solver.numu = config["numu"]
        solver.nphi = config["nphi"]
        solver.usrtau = config["usrtau"]
        solver.usrang = config["usrang"]
        solver.lamber = config["lamber"]
        solver.planck = config["planck"]
        solver.onlyfl = config["onlyfl"]
        solver.quiet = config["quiet"]
        solver.umu0 = config["umu0"]
        solver.phi0 = config["phi0"]
        solver.fisot = config["fisot"]
        solver.fluor = config["fluor"]
        solver.set_utau(config["utau"])

        solver.allocate(nbatch)
        solver.set_dtauc(dtauc_batch)
        solver.set_ssalb(ssalb_batch)
        solver.set_pmom(pmom_batch)
        solver.set_fbeam(fbeam_batch)
        solver.set_albedo(albedo_batch)

        solver.solve()

        rfldir = np.array(solver.rfldir)
        rfldn = np.array(solver.rfldn)
        flup = np.array(solver.flup)

        # All rows should be identical
        for i in range(1, nbatch):
            np.testing.assert_array_equal(rfldir[0], rfldir[i])
            np.testing.assert_array_equal(rfldn[0], rfldn[i])
            np.testing.assert_array_equal(flup[0], flup[i])


class TestBatchEdgeCases:
    """Edge cases and error handling."""

    def test_nbatch_one(self):
        """Single-item batch should work and match sequential."""
        config = make_single_layer_config()

        dtauc = np.array([[1.0]])
        ssalb = np.array([[0.9]])
        fbeam = np.array([np.pi])
        albedo = np.array([0.0])

        nmom_nstr = max(config["nmom"], config["nstr"])
        pmom_single = np.zeros((nmom_nstr + 1, 1))
        pmom_single[0, 0] = 1.0

        pmom_batch = np.zeros((nmom_nstr + 1, 1, 1), order="F")
        pmom_batch[0, :, :] = 1.0

        ref = solve_single(config, dtauc[0], ssalb[0], pmom_single)

        solver = BatchSolver()
        solver.nstr = config["nstr"]
        solver.nlyr = config["nlyr"]
        solver.nmom = config["nmom"]
        solver.ntau = config["ntau"]
        solver.numu = config["numu"]
        solver.nphi = config["nphi"]
        solver.usrtau = config["usrtau"]
        solver.usrang = config["usrang"]
        solver.lamber = config["lamber"]
        solver.planck = config["planck"]
        solver.onlyfl = config["onlyfl"]
        solver.quiet = config["quiet"]
        solver.umu0 = config["umu0"]
        solver.phi0 = config["phi0"]
        solver.fisot = config["fisot"]
        solver.fluor = config["fluor"]
        solver.set_utau(config["utau"])

        solver.allocate(1)
        solver.set_dtauc(dtauc)
        solver.set_ssalb(ssalb)
        solver.set_pmom(pmom_batch)
        solver.set_fbeam(fbeam)
        solver.set_albedo(albedo)

        solver.solve()

        np.testing.assert_allclose(
            np.array(solver.rfldir)[0], ref["rfldir"], rtol=1e-14
        )
        np.testing.assert_allclose(
            np.array(solver.flup)[0], ref["flup"], rtol=1e-14
        )


class TestBatchShapeValidation:
    """Shape validation errors for mismatched inputs."""

    def _make_allocated_solver(self, nbatch=4):
        solver = BatchSolver()
        solver.nstr = 8
        solver.nlyr = 1
        solver.nmom = 8
        solver.ntau = 2
        solver.numu = 0
        solver.nphi = 0
        solver.usrtau = True
        solver.usrang = False
        solver.lamber = True
        solver.planck = False
        solver.onlyfl = True
        solver.quiet = True
        solver.umu0 = 1.0
        solver.phi0 = 0.0
        solver.set_utau(np.array([0.0, 1.0]))
        solver.allocate(nbatch)
        return solver

    def test_dtauc_wrong_batch_dim(self):
        solver = self._make_allocated_solver(4)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            solver.set_dtauc(np.ones((3, 1)))

    def test_dtauc_wrong_layer_dim(self):
        solver = self._make_allocated_solver(4)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            solver.set_dtauc(np.ones((4, 2)))

    def test_ssalb_wrong_shape(self):
        solver = self._make_allocated_solver(4)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            solver.set_ssalb(np.ones((4, 3)))

    def test_fbeam_wrong_size(self):
        solver = self._make_allocated_solver(4)
        with pytest.raises(RuntimeError, match="size mismatch"):
            solver.set_fbeam(np.ones(5))

    def test_not_allocated_raises(self):
        solver = BatchSolver()
        solver.nstr = 8
        solver.nlyr = 1
        solver.quiet = True
        solver.lamber = True
        with pytest.raises(RuntimeError, match="Not allocated"):
            solver.set_dtauc(np.ones((1, 1)))

    def test_quiet_required(self):
        solver = BatchSolver()
        solver.nstr = 8
        solver.nlyr = 1
        solver.nmom = 8
        solver.ntau = 2
        solver.lamber = True
        solver.quiet = False  # Violates contract
        with pytest.raises(RuntimeError, match="quiet=True"):
            solver.allocate(1)

    def test_lamber_required(self):
        solver = BatchSolver()
        solver.nstr = 8
        solver.nlyr = 1
        solver.nmom = 8
        solver.ntau = 2
        solver.quiet = True
        solver.lamber = False  # Violates contract
        with pytest.raises(RuntimeError, match="lamber=True"):
            solver.allocate(1)

    def test_not_solved_raises(self):
        solver = self._make_allocated_solver(2)
        with pytest.raises(RuntimeError, match="Not solved"):
            _ = solver.rfldir
