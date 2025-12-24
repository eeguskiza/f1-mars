"""Tests for track loading and validation."""

import pytest
from tracks.track_loader import list_available_tracks, load_track, get_tracks_by_difficulty


def test_all_tracks_exist():
    """Verificar que los 4 circuitos existen."""
    tracks = list_available_tracks()
    assert "oval" in tracks
    assert "simple" in tracks
    assert "technical" in tracks
    assert "mixed" in tracks


def test_load_each_track():
    """Verificar que cada circuito carga correctamente."""
    for name in ["oval", "simple", "technical", "mixed"]:
        track = load_track(name)
        assert track is not None
        assert track.total_length > 0
        assert len(track.checkpoint_indices) >= 4


def test_difficulty_progression():
    """Verificar dificultades correctas."""
    assert "oval" in get_tracks_by_difficulty(0)
    assert "simple" in get_tracks_by_difficulty(1)
    assert "technical" in get_tracks_by_difficulty(2)
    assert "mixed" in get_tracks_by_difficulty(3)


def test_track_widths():
    """Verificar anchos de pista razonables."""
    for name in list_available_tracks():
        track = load_track(name)
        assert 10 <= track.widths[0] <= 15, f"{name} has invalid width: {track.widths[0]}"


def test_track_in_environment():
    """Verificar que los circuitos funcionan en F1Env."""
    from f1_mars.envs import F1Env
    from tracks import TRACKS_DIR

    for name in ["oval", "simple", "technical", "mixed"]:
        track_path = str(TRACKS_DIR / f"{name}.json")
        env = F1Env(track_path=track_path, max_laps=1)
        obs, info = env.reset()

        # Verificar que el entorno funciona
        assert obs.shape[0] == 26

        # Dar unos pasos
        for _ in range(100):
            action = [0.0, 0.5, 0.0]  # Acelerar recto
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

        env.close()


def test_track_not_found():
    """Verificar que se lanza error para circuito inexistente."""
    with pytest.raises(FileNotFoundError):
        load_track("nonexistent")


def test_track_has_valid_checkpoints():
    """Verificar que los checkpoints están dentro del rango válido."""
    for name in list_available_tracks():
        track = load_track(name)
        for checkpoint_idx in track.checkpoint_indices:
            assert 0 <= checkpoint_idx < track.num_control_points
