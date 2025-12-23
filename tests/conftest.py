"""Pytest fixtures for F1-MARS tests."""

import pytest
import numpy as np
from f1_mars.envs import F1Env, Car, Track, TrackGenerator, TyreSet, TyreCompound


@pytest.fixture
def car():
    """Create a car at origin facing right."""
    return Car(x=0, y=0, heading=0)


@pytest.fixture
def oval_track():
    """Create a simple oval track for testing."""
    track_data = TrackGenerator.generate_oval(
        length=500, 
        width=200, 
        track_width=12.0
    )
    track = Track.__new__(Track)
    track.load_from_dict(track_data)
    return track


@pytest.fixture
def soft_tyres():
    """Create fresh soft tyres."""
    return TyreSet(TyreCompound.SOFT)


@pytest.fixture
def medium_tyres():
    """Create fresh medium tyres."""
    return TyreSet(TyreCompound.MEDIUM)


@pytest.fixture
def hard_tyres():
    """Create fresh hard tyres."""
    return TyreSet(TyreCompound.HARD)


@pytest.fixture
def env():
    """Create F1 environment with default settings."""
    return F1Env(max_laps=3)


@pytest.fixture
def env_long_race():
    """Create F1 environment for longer races."""
    return F1Env(max_laps=10)
