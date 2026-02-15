import numpy as np

import nanodisort as nd
from nanodisort.utils import phase_functions as pf

# Benchmark parameters
NSTR = 16
NLYR = 120
NMOM = 16
NBATCH = 100

# Shared configuration applied to both DisortState and BatchSolver
_shared_config = {
    "nstr": NSTR,
    "nlyr": NLYR,
    "nmom": NMOM,
    "ntau": 0,
    "numu": 0,
    "nphi": 0,
    "usrtau": False,
    "usrang": False,
    "lamber": True,
    "planck": False,
    "onlyfl": True,
    "quiet": True,
    "intensity_correction": False,
    "spher": False,
    "umu0": 0.5,
    "phi0": 0.0,
    "fisot": 0.0,
    "fluor": 0.0,
    "albedo": 0.1,
    "accur": 0.0,
    "wvnmlo": 0.0,
    "wvnmhi": 0.0,
    "btemp": 0.0,
    "ttemp": 0.0,
    "temis": 0.0,
}

# Pre-compute shared data
_dtauc = np.full((NBATCH, NLYR), 0.5)
_ssalb = np.random.default_rng(1).uniform(0.5, 1.0, (NBATCH, NLYR))
_pmom_single = pf.henyey_greenstein(0.7, NMOM).reshape(-1, 1) * np.ones(NLYR)
_pmom_batch = _pmom_single[:, :, np.newaxis] * np.ones(NBATCH)
_fbeam = np.full(NBATCH, np.pi)
_albedo = np.full(NBATCH, 0.1)


def _apply_config(target):
    for key, value in _shared_config.items():
        setattr(target, key, value)


def _make_single_state() -> nd.DisortState:
    ds = nd.DisortState()
    _apply_config(ds)
    ds.allocate()
    return ds


def _make_batch_solver() -> nd.BatchSolver:
    solver = nd.BatchSolver(nthreads=2)
    _apply_config(solver)
    solver.allocate(NBATCH)
    solver.set_dtauc(_dtauc)
    solver.set_ssalb(_ssalb)
    solver.set_pmom(_pmom_batch)
    solver.set_fbeam(_fbeam)
    solver.set_albedo(_albedo)
    return solver


class BenchmarkBatch:
    def benchmark_single(self, benchmark):
        ds = _make_single_state()

        def run():
            for i in range(NBATCH):
                ds.dtauc = _dtauc[i]
                ds.ssalb = _ssalb[i]
                ds.fbeam = _fbeam[i]
                ds.pmom = _pmom_single
                ds.solve()

        benchmark(run)

    def benchmark_batch(self, benchmark):
        solver = _make_batch_solver()

        def run():
            solver.solve()

        benchmark(run)
