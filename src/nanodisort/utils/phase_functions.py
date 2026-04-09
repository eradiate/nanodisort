# SPDX-FileCopyrightText: 2025 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module provides functions to generate Legendre expansion coefficients
for various phase functions, matching cdisort's ``c_getmom()`` function.
"""

import enum

import numpy as np
from numpy.typing import NDArray


class PhaseFunction(enum.IntEnum):
    """Phase function type constants (matching cdisort.h)."""

    ISOTROPIC = 1  #: Isotropic scattering
    RAYLEIGH = 2  #: Rayleigh scattering
    HENYEY_GREENSTEIN = 3  #: Henyey-Greenstein
    HAZE_GARCIA_SIEWERT = 4  #: Haze L (:cite:t:`Garcia1985BenchmarkResultsRT` table 10)
    CLOUD_GARCIA_SIEWERT = 5  #: Cloud C.1 (:cite:t:`Garcia1985BenchmarkResultsRT` table 17)


# Cloud C.1 Legendre moments (from cdisort.c)
CLDMOM = np.array(
    [
        2.544,
        3.883,
        4.568,
        5.235,
        5.887,
        6.457,
        7.177,
        7.859,
        8.494,
        9.286,
        9.856,
        10.615,
        11.229,
        11.851,
        12.503,
        13.058,
        13.626,
        14.209,
        14.660,
        15.231,
        15.641,
        16.126,
        16.539,
        16.934,
        17.325,
        17.673,
        17.999,
        18.329,
        18.588,
        18.885,
        19.103,
        19.345,
        19.537,
        19.721,
        19.884,
        20.024,
        20.145,
        20.251,
        20.330,
        20.401,
        20.444,
        20.477,
        20.489,
        20.483,
        20.467,
        20.427,
        20.382,
        20.310,
        20.236,
        20.136,
        20.036,
        19.909,
        19.785,
        19.632,
        19.486,
        19.311,
        19.145,
        18.949,
        18.764,
        18.551,
        18.348,
        18.119,
        17.901,
        17.659,
        17.428,
        17.174,
        16.931,
        16.668,
        16.415,
        16.144,
        15.883,
        15.606,
        15.338,
        15.058,
        14.784,
        14.501,
        14.225,
        13.941,
        13.662,
        13.378,
        13.098,
        12.816,
        12.536,
        12.257,
        11.978,
        11.703,
        11.427,
        11.156,
        10.884,
        10.618,
        10.350,
        10.090,
        9.827,
        9.574,
        9.318,
        9.072,
        8.822,
        8.584,
        8.340,
        8.110,
        7.874,
        7.652,
        7.424,
        7.211,
        6.990,
        6.785,
        6.573,
        6.377,
        6.173,
        5.986,
        5.790,
        5.612,
        5.424,
        5.255,
        5.075,
        4.915,
        4.744,
        4.592,
        4.429,
        4.285,
        4.130,
        3.994,
        3.847,
        3.719,
        3.580,
        3.459,
        3.327,
        3.214,
        3.090,
        2.983,
        2.866,
        2.766,
        2.656,
        2.562,
        2.459,
        2.372,
        2.274,
        2.193,
        2.102,
        2.025,
        1.940,
        1.869,
        1.790,
        1.723,
        1.649,
        1.588,
        1.518,
        1.461,
        1.397,
        1.344,
        1.284,
        1.235,
        1.179,
        1.134,
        1.082,
        1.040,
        0.992,
        0.954,
        0.909,
        0.873,
        0.832,
        0.799,
        0.762,
        0.731,
        0.696,
        0.668,
        0.636,
        0.610,
        0.581,
        0.557,
        0.530,
        0.508,
        0.483,
        0.463,
        0.440,
        0.422,
        0.401,
        0.384,
        0.364,
        0.349,
        0.331,
        0.317,
        0.301,
        0.288,
        0.273,
        0.262,
        0.248,
        0.238,
        0.225,
        0.215,
        0.204,
        0.195,
        0.185,
        0.177,
        0.167,
        0.160,
        0.151,
        0.145,
        0.137,
        0.131,
        0.124,
        0.118,
        0.112,
        0.107,
        0.101,
        0.097,
        0.091,
        0.087,
        0.082,
        0.079,
        0.074,
        0.071,
        0.067,
        0.064,
        0.060,
        0.057,
        0.054,
        0.052,
        0.049,
        0.047,
        0.044,
        0.042,
        0.039,
        0.038,
        0.035,
        0.034,
        0.032,
        0.030,
        0.029,
        0.027,
        0.026,
        0.024,
        0.023,
        0.022,
        0.021,
        0.020,
        0.018,
        0.018,
        0.017,
        0.016,
        0.015,
        0.014,
        0.013,
        0.013,
        0.012,
        0.011,
        0.011,
        0.010,
        0.009,
        0.009,
        0.008,
        0.008,
        0.008,
        0.007,
        0.007,
        0.006,
        0.006,
        0.006,
        0.005,
        0.005,
        0.005,
        0.005,
        0.004,
        0.004,
        0.004,
        0.004,
        0.003,
        0.003,
        0.003,
        0.003,
        0.003,
        0.003,
        0.002,
        0.002,
        0.002,
        0.002,
        0.002,
        0.002,
        0.002,
        0.002,
        0.002,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
    ]
)


# Haze-L Legendre moments (from cdisort.c)
HAZELM = np.array(
    [
        2.41260,
        3.23047,
        3.37296,
        3.23150,
        2.89350,
        2.49594,
        2.11361,
        1.74812,
        1.44692,
        1.17714,
        0.96643,
        0.78237,
        0.64114,
        0.51966,
        0.42563,
        0.34688,
        0.28351,
        0.23317,
        0.18963,
        0.15788,
        0.12739,
        0.10762,
        0.08597,
        0.07381,
        0.05828,
        0.05089,
        0.03971,
        0.03524,
        0.02720,
        0.02451,
        0.01874,
        0.01711,
        0.01298,
        0.01198,
        0.00904,
        0.00841,
        0.00634,
        0.00592,
        0.00446,
        0.00418,
        0.00316,
        0.00296,
        0.00225,
        0.00210,
        0.00160,
        0.00150,
        0.00115,
        0.00107,
        0.00082,
        0.00077,
        0.00059,
        0.00055,
        0.00043,
        0.00040,
        0.00031,
        0.00029,
        0.00023,
        0.00021,
        0.00017,
        0.00015,
        0.00012,
        0.00011,
        0.00009,
        0.00008,
        0.00006,
        0.00006,
        0.00005,
        0.00004,
        0.00004,
        0.00003,
        0.00003,
        0.00002,
        0.00002,
        0.00002,
        0.00001,
        0.00001,
        0.00001,
        0.00001,
        0.00001,
        0.00001,
        0.00001,
        0.00001,
    ]
)


def getmom(iphas: PhaseFunction, gg: float, nmom: int) -> NDArray[np.float64]:
    """
    Calculate phase function Legendre expansion coefficients.

    This function mimics the ``c_getmom()`` function from cdisort.c.

    Parameters
    ----------
    iphas : PhaseFunction
        Phase function type.

    gg : float
        Asymmetry factor for Henyey-Greenstein case (must be in (-1, 1)).

    nmom : int
        Index of highest Legendre coefficient needed.

    Returns
    -------
    pmom : ndarray
        Legendre expansion coefficients (shape: (nmom+1,))
    """
    if nmom < 0:
        raise ValueError("nmom must be non-negative")

    # Initialize all moments to zero
    pmom = np.zeros(nmom + 1, dtype=np.float64)
    pmom[0] = 1.0

    if iphas == PhaseFunction.ISOTROPIC:
        # Isotropic: all zeros except pmom[0]
        pass

    elif iphas == PhaseFunction.RAYLEIGH:
        # Rayleigh: pmom[2] = 0.1
        if nmom >= 2:
            pmom[2] = 0.1

    elif iphas == PhaseFunction.HENYEY_GREENSTEIN:
        # Henyey-Greenstein: pmom[k] = gg^k
        if gg <= -1.0 or gg >= 1.0:
            raise ValueError(f"gg must be in (-1, 1), got {gg}")
        for k in range(1, nmom + 1):
            pmom[k] = gg**k

    elif iphas == PhaseFunction.HAZE_GARCIA_SIEWERT:
        # Haze-L phase function
        n_available = min(len(HAZELM), nmom)
        for k in range(n_available):
            pmom[k + 1] = HAZELM[k] / (2 * (k + 1) + 1)

    elif iphas == PhaseFunction.CLOUD_GARCIA_SIEWERT:
        # Cloud C.1 phase function
        n_available = min(len(CLDMOM), nmom)
        for k in range(n_available):
            pmom[k + 1] = CLDMOM[k] / (2 * (k + 1) + 1)

    else:
        raise ValueError(f"Unknown phase function type: {iphas}")

    return pmom


def isotropic(nmom: int) -> NDArray[np.float64]:
    """Generate isotropic phase function moments."""
    return getmom(PhaseFunction.ISOTROPIC, 0.0, nmom)


def rayleigh(nmom: int) -> NDArray[np.float64]:
    """Generate Rayleigh phase function moments."""
    return getmom(PhaseFunction.RAYLEIGH, 0.0, nmom)


def henyey_greenstein(gg: float, nmom: int) -> NDArray[np.float64]:
    """Generate Henyey-Greenstein phase function moments."""
    return getmom(PhaseFunction.HENYEY_GREENSTEIN, gg, nmom)


def haze_l(nmom: int) -> NDArray[np.float64]:
    """Generate Haze-L phase function moments."""
    return getmom(PhaseFunction.HAZE_GARCIA_SIEWERT, 0.0, nmom)


def cloud_c1(nmom: int) -> NDArray[np.float64]:
    """Generate Cloud C.1 phase function moments."""
    return getmom(PhaseFunction.CLOUD_GARCIA_SIEWERT, 0.0, nmom)
