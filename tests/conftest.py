"""
Pytest configuration and shared fixtures for nanodisort tests.
"""

import pytest


@pytest.fixture
def basic_state():
    """
    Create a basic DisortState with common test dimensions.

    Returns
    -------
    DisortState
        A configured but unallocated state object.
    """
    import nanodisort as nd

    state = nd.DisortState()
    state.nstr = 8
    state.nlyr = 6
    state.nmom = 8
    state.ntau = 5
    state.numu = 4
    state.nphi = 3
    return state
