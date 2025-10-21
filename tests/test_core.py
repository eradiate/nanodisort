"""
Tests for core nanodisort bindings.

These tests verify the basic functionality of the DisortState class
and the underlying cdisort solver.
"""

import pytest

import nanodisort as nd


class TestDisortState:
    """Test DisortState class initialization and configuration."""

    def test_create_state(self):
        """Test creating a DisortState instance."""
        state = nd.DisortState()
        assert state is not None

    def test_set_dimensions(self):
        """Test setting solver dimensions."""
        state = nd.DisortState()
        state.nstr = 8
        state.nlyr = 6
        state.nmom = 8
        state.ntau = 5
        state.numu = 4
        state.nphi = 3

        assert state.nstr == 8
        assert state.nlyr == 6
        assert state.nmom == 8
        assert state.ntau == 5
        assert state.numu == 4
        assert state.nphi == 3

    def test_allocate(self):
        """Test memory allocation."""
        state = nd.DisortState()
        state.nstr = 8
        state.nlyr = 6
        state.nmom = 8
        state.ntau = 5
        state.numu = 4
        state.nphi = 3

        # Should not raise
        state.allocate()

    @pytest.mark.skip(reason="Solver integration not yet complete")
    def test_solve_minimal(self):
        """Test running solver with minimal configuration."""
        state = nd.DisortState()
        state.nstr = 8
        state.nlyr = 1
        state.nmom = 0
        state.ntau = 1
        state.numu = 1
        state.nphi = 1

        state.allocate()
        # TODO: Set required inputs before solving
        # state.solve()


class TestImport:
    """Test package import and version."""

    def test_import(self):
        """Test that package imports successfully."""
        assert nd.__version__ == "0.1.0"

    def test_public_api(self):
        """Test that expected names are exported."""
        assert "DisortState" in nd.__all__
        assert hasattr(nd, "DisortState")
