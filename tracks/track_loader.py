"""Utilidades para cargar y gestionar circuitos."""

import json
from pathlib import Path
from typing import Optional
from f1_mars.envs.track import Track


TRACKS_DIR = Path(__file__).parent


def list_available_tracks() -> list[str]:
    """Retorna nombres de circuitos disponibles (sin extensión)."""
    return [f.stem for f in TRACKS_DIR.glob("*.json")]


def load_track(name: str) -> Track:
    """
    Carga un circuito por nombre.

    Args:
        name: Nombre del circuito (sin .json)

    Returns:
        Track instance
    """
    filepath = TRACKS_DIR / f"{name}.json"
    if not filepath.exists():
        available = list_available_tracks()
        raise FileNotFoundError(
            f"Track '{name}' not found. Available: {available}"
        )

    with open(filepath) as f:
        data = json.load(f)

    # Create Track instance and load from dict
    track = Track.__new__(Track)
    track.load_from_dict(data)
    return track


def get_tracks_by_difficulty(difficulty: int) -> list[str]:
    """Retorna nombres de circuitos con la dificultad dada."""
    tracks = []
    for name in list_available_tracks():
        filepath = TRACKS_DIR / f"{name}.json"
        with open(filepath) as f:
            data = json.load(f)
        if data.get("difficulty") == difficulty:
            tracks.append(name)
    return tracks


def get_track_info(name: str) -> dict:
    """Retorna metadata de un circuito sin cargarlo completo."""
    filepath = TRACKS_DIR / f"{name}.json"
    with open(filepath) as f:
        data = json.load(f)
    return {
        "name": data["name"],
        "difficulty": data["difficulty"],
        "width": data["width"],
        "checkpoints": len(data["checkpoints"]),
        "reference_lap_time": data.get("reference_lap_time"),
        "metadata": data.get("metadata", {})
    }
