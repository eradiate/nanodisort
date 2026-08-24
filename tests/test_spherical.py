# SPDX-FileCopyrightText: 2025 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Test the pseudo-spherical correction to the direct beam.

The reference values are analytic. cdisort's Chapman function is pure
spherical-shell geometry (no refraction, extinction constant within a layer),
so it can be reimplemented in a few lines of numpy and compared to round-off.

The DISORT test suite provides no reference data here: ``disotest.c`` enables
``spher`` only for ``c_twostr``, which nanodisort does not bind.
"""

import numpy as np
import pytest

import nanodisort as nd

RADIUS = 6371.0  # km, Earth mean radius
FBEAM = np.pi
SZA = [0.0, 30.0, 60.0, 80.0, 88.0]  # degrees

# Six-layer atmosphere, purely absorbing, used by most tests below
ZD = np.array([60.0, 40.0, 25.0, 15.0, 8.0, 3.0, 0.0])
DTAUC = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8])


def chapman(zd, dtauc, radius, sza, lc, taup):
    """
    Slant optical depth from the top of the atmosphere down to the point at
    fraction ``taup`` of layer ``lc`` (0 at its bottom, 1 at its top).

    Reimplements the :c:func:`c_chapman` C function for solar zenith angles
    below 90 degrees. Layer indices are 1-based, matching cdisort.
    """
    zd = np.asarray(zd, dtype=float)
    dtauc = np.asarray(dtauc, dtype=float)
    r = radius + zd
    xp = radius + zd[lc] + (zd[lc - 1] - zd[lc]) * taup
    p = xp * np.sin(np.radians(sza))  # impact parameter

    j = np.arange(1, lc + 1)
    rj = r[j - 1]
    rjp1 = r[j]
    rjp1[-1] = xp  # the ray stops partway through layer lc
    dsj = np.sqrt(rj**2 - p**2) - np.sqrt(rjp1**2 - p**2)
    return np.sum(dtauc[j - 1] * dsj / (zd[j - 1] - zd[j]))


def effective_umu0(zd, dtauc, radius, sza, lc):
    """
    Effective beam cosine ``CH(lc)`` that cdisort substitutes for ``umu0`` in
    layer ``lc``: the vertical optical depth down to the layer mid-point
    divided by the slant optical depth to the same point.
    """
    tauc = np.concatenate([[0.0], np.cumsum(dtauc)])
    taup_mid = tauc[lc - 1] + dtauc[lc - 1] / 2.0
    return taup_mid / chapman(zd, dtauc, radius, sza, lc, 0.5)


def rfldir_expected(zd, dtauc, radius, sza, utau, lyu):
    """
    Direct beam flux predicted for user levels ``utau``, each lying in the
    (1-based) layer given by ``lyu``.
    """
    ch = np.array([effective_umu0(zd, dtauc, radius, sza, lc) for lc in lyu])
    return abs(np.cos(np.radians(sza))) * FBEAM * np.exp(-np.asarray(utau) / ch)


def make_state(zd, dtauc, utau, sza, radius, spher=True):
    """Build a solved, purely absorbing, flux-only DISORT state."""
    nlyr = len(dtauc)
    state = nd.DisortState()
    state.nstr = 8
    state.nlyr = nlyr
    state.nmom = 8
    state.ntau = len(utau)
    state.numu = 0
    state.nphi = 0
    state.usrtau = True
    state.usrang = False
    state.lamber = True
    state.onlyfl = True
    state.quiet = True
    state.spher = spher
    state.allocate()

    state.dtauc = dtauc
    state.ssalb = np.zeros(nlyr)  # no scattering, hence no delta-M scaling
    pmom = np.zeros((max(state.nmom, state.nstr) + 1, nlyr))
    pmom[0, :] = 1.0
    state.pmom = pmom
    state.utau = utau
    state.fbeam = FBEAM
    state.umu0 = np.cos(np.radians(sza))
    state.albedo = 0.0
    if spher:
        state.zd = zd
        state.radius = radius

    state.solve()
    return state


@pytest.mark.parametrize("sza", SZA)
def test_single_layer(sza):
    """Direct beam through one spherical shell matches the Chapman geometry."""
    zd = np.array([30.0, 0.0])
    dtauc = np.array([0.5])
    utau = np.array([0.0, 0.25, 0.5])  # all inside the single layer

    state = make_state(zd, dtauc, utau, sza, RADIUS)
    expected = rfldir_expected(zd, dtauc, RADIUS, sza, utau, lyu=[1, 1, 1])
    np.testing.assert_allclose(state.rfldir, expected, rtol=1e-12)


@pytest.mark.parametrize("sza", SZA)
def test_multilayer(sza):
    """Same, accumulated over six shells."""
    tauc = np.concatenate([[0.0], np.cumsum(DTAUC)])
    utau = tauc[:-1] + DTAUC / 2.0  # layer mid-points, so lyu is unambiguous
    lyu = np.arange(1, len(DTAUC) + 1)

    state = make_state(ZD, DTAUC, utau, sza, RADIUS)
    expected = rfldir_expected(ZD, DTAUC, RADIUS, sza, utau, lyu)
    np.testing.assert_allclose(state.rfldir, expected, rtol=1e-12)


@pytest.mark.parametrize("sza", SZA[:-1])
def test_plane_parallel_limit(sza):
    """
    A huge radius flattens the shells back onto the plane-parallel result.

    The tolerance is set by the geometry, not by the binding. Residual
    curvature falls off as 1/radius, but the Chapman path lengths are
    differences of square roots of nearly equal numbers, so raising the radius
    much beyond 1e9 km loses more to cancellation in double precision than it
    gains in flatness. The grazing case (88 degrees) is excluded for the same
    reason; it is covered exactly by the tests above.
    """
    utau = np.concatenate([[0.0], np.cumsum(DTAUC)])

    spherical = make_state(ZD, DTAUC, utau, sza, radius=1e9)
    plane = make_state(ZD, DTAUC, utau, sza, RADIUS, spher=False)
    np.testing.assert_allclose(spherical.rfldir, plane.rfldir, rtol=1e-5)


def test_airmass_below_plane_parallel():
    """
    Curvature shortens the slant path relative to 1/cos(theta), so the beam is
    less attenuated, increasingly so towards the horizon.
    """
    utau = np.array([np.sum(DTAUC)])  # bottom of the atmosphere
    ratios = []

    for sza in SZA:
        spherical = make_state(ZD, DTAUC, utau, sza, RADIUS)
        plane = make_state(ZD, DTAUC, utau, sza, RADIUS, spher=False)
        ratios.append(spherical.rfldir[0] / plane.rfldir[0])

    ratios = np.array(ratios)
    assert ratios[0] == pytest.approx(1.0)  # overhead sun: no difference
    assert np.all(ratios[1:] > 1.0)
    assert np.all(np.diff(ratios) > 0.0)


def _unsolved_state(spher=True):
    """A state ready to solve, with the spherical geometry left unset."""
    state = nd.DisortState()
    state.nstr = 8
    state.nlyr = len(DTAUC)
    state.nmom = 8
    state.ntau = 1
    state.numu = 0
    state.nphi = 0
    state.usrtau = True
    state.usrang = False
    state.lamber = True
    state.onlyfl = True
    state.quiet = True
    state.spher = spher
    state.allocate()

    state.dtauc = DTAUC
    state.ssalb = np.zeros(len(DTAUC))
    pmom = np.zeros((9, len(DTAUC)))
    pmom[0, :] = 1.0
    state.pmom = pmom
    state.utau = np.array([0.0])
    state.fbeam = FBEAM
    state.umu0 = 0.5
    state.albedo = 0.0
    return state


@pytest.mark.parametrize(
    "radius, zd, match",
    [
        (0.0, ZD, "radius must be positive"),  # radius never set
        (-1.0, ZD, "radius must be positive"),
        (RADIUS, np.zeros(len(ZD)), "strictly decreasing"),  # zd never set
        (RADIUS, [60.0, 40.0, 25.0, 25.0, 8.0, 3.0, 0.0], "strictly decreasing"),
        (RADIUS, [60.0, 40.0, 25.0, 15.0, 8.0, 3.0, 1.0], r"zd\[nlyr\] must be 0"),
    ],
)
def test_invalid_geometry(radius, zd, match):
    """Incomplete or degenerate geometry raises instead of solving."""
    state = _unsolved_state()
    state.radius = radius
    state.zd = np.asarray(zd, dtype=float)

    with pytest.raises(RuntimeError, match=match):
        state.solve()


def test_geometry_ignored_when_plane_parallel():
    """The geometry is not validated unless spher is enabled."""
    state = _unsolved_state(spher=False)  # radius and zd left at zero
    state.solve()
    assert state.rfldir[0] == pytest.approx(0.5 * FBEAM)


def make_batch(nbatch, spher=True, radius=RADIUS, zd=ZD):
    """Build a batch solver over identical members of the six-layer atmosphere."""
    solver = nd.BatchSolver(nthreads=2)
    solver.nstr = 8
    solver.nlyr = len(DTAUC)
    solver.nmom = 8
    solver.ntau = len(DTAUC)
    solver.numu = 0
    solver.nphi = 0
    solver.usrtau = True
    solver.usrang = False
    solver.lamber = True
    solver.onlyfl = True
    solver.quiet = True
    solver.spher = spher
    solver.umu0 = np.cos(np.radians(60.0))
    solver.radius = radius

    tauc = np.concatenate([[0.0], np.cumsum(DTAUC)])
    solver.set_utau(tauc[:-1] + DTAUC / 2.0)
    if spher:
        solver.set_zd(np.asarray(zd, dtype=float))

    solver.allocate(nbatch)
    solver.set_dtauc(np.tile(DTAUC, (nbatch, 1)))
    solver.set_ssalb(np.zeros((nbatch, len(DTAUC))))
    pmom = np.zeros((9, len(DTAUC), nbatch), order="F")
    pmom[0, :, :] = 1.0
    solver.set_pmom(pmom)
    solver.set_fbeam(np.full(nbatch, FBEAM))
    solver.set_albedo(np.zeros(nbatch))
    return solver


def test_batch_matches_single():
    """BatchSolver reproduces the DisortState result for the same geometry."""
    nbatch = 4
    solver = make_batch(nbatch)
    solver.solve()

    tauc = np.concatenate([[0.0], np.cumsum(DTAUC)])
    utau = tauc[:-1] + DTAUC / 2.0
    reference = make_state(ZD, DTAUC, utau, 60.0, RADIUS).rfldir

    for i in range(nbatch):
        np.testing.assert_allclose(solver.rfldir[i], reference, rtol=1e-12)


@pytest.mark.parametrize(
    "radius, zd, match",
    [
        (0.0, ZD, "radius must be positive"),
        (RADIUS, None, "zd must be set"),  # set_zd never called
        (RADIUS, np.zeros(len(ZD)), "strictly decreasing"),
    ],
)
def test_batch_invalid_geometry(radius, zd, match):
    """The batch solver validates the shared geometry once, at allocation."""
    with pytest.raises(RuntimeError, match=match):
        if zd is None:
            solver = nd.BatchSolver(nthreads=1)
            solver.nstr = 8
            solver.nlyr = len(DTAUC)
            solver.nmom = 8
            solver.ntau = 1
            solver.lamber = True
            solver.quiet = True
            solver.spher = True
            solver.radius = radius
            solver.allocate(2)
        else:
            make_batch(2, radius=radius, zd=zd)
