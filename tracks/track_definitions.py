"""
Generación de circuitos para F1-MARS.
Cada función retorna un dict compatible con Track.from_dict()
"""

import numpy as np
import json
from pathlib import Path


def generate_oval() -> dict:
    """
    Circuito oval simple (Dificultad 0).
    Ideal para aprender controles básicos.
    """
    points = []

    # Parámetros
    straight_length = 200
    curve_radius = 80
    num_curve_points = 25
    num_straight_points = 20

    # Recta superior (izquierda a derecha)
    for i in range(num_straight_points):
        x = -straight_length/2 + i * straight_length / (num_straight_points - 1)
        y = curve_radius
        points.append([x, y])

    # Curva derecha (semicírculo)
    for i in range(num_curve_points):
        angle = np.pi/2 - i * np.pi / (num_curve_points - 1)
        x = straight_length/2 + curve_radius * np.cos(angle)
        y = curve_radius * np.sin(angle)
        points.append([x, y])

    # Recta inferior (derecha a izquierda)
    for i in range(num_straight_points):
        x = straight_length/2 - i * straight_length / (num_straight_points - 1)
        y = -curve_radius
        points.append([x, y])

    # Curva izquierda (semicírculo)
    for i in range(num_curve_points - 1):  # -1 para no duplicar punto inicial
        angle = -np.pi/2 - i * np.pi / (num_curve_points - 1)
        x = -straight_length/2 + curve_radius * np.cos(angle)
        y = curve_radius * np.sin(angle)
        points.append([x, y])

    # Calcular longitud aproximada
    total_length = 2 * straight_length + 2 * np.pi * curve_radius

    return {
        "name": "Oval",
        "author": "F1-MARS",
        "difficulty": 0,
        "centerline": points,
        "width": 14.0,
        "pit_entry_index": 5,
        "pit_exit_index": 15,
        "start_position": [-straight_length/2 + 20, curve_radius, 0.0],
        "checkpoints": [0, len(points)//4, len(points)//2, 3*len(points)//4],
        "reference_lap_time": 20.0,
        "metadata": {
            "total_length_approx": total_length,
            "characteristics": ["beginner", "high_speed", "symmetrical"],
            "description": "Simple oval for learning basic controls"
        }
    }


def generate_simple() -> dict:
    """
    Circuito simple con una horquilla (Dificultad 1).
    Introduce frenadas fuertes y curvas variadas.
    """
    points = []

    # Recta de meta (hacia la derecha)
    for i in range(20):
        points.append([i * 15, 0])

    # Curva suave a la derecha (90°)
    cx, cy = 300, 50
    for i in range(15):
        angle = -np.pi/2 + i * (np.pi/2) / 14
        points.append([cx + 50 * np.cos(angle), cy + 50 * np.sin(angle)])

    # Recta corta hacia arriba
    for i in range(10):
        points.append([350, 50 + i * 10])

    # Horquilla cerrada (180°)
    cx, cy = 320, 150
    for i in range(20):
        angle = 0 - i * np.pi / 19
        points.append([cx + 30 * np.cos(angle), cy + 30 * np.sin(angle)])

    # Recta hacia abajo
    for i in range(10):
        points.append([290, 150 - i * 10])

    # Curva amplia de regreso
    cx, cy = 200, 50
    for i in range(20):
        angle = np.pi/2 - i * np.pi / 19
        points.append([cx + 90 * np.cos(angle), cy + 90 * np.sin(angle)])

    # Curva final hacia la recta de meta
    cx, cy = 50, -40
    for i in range(15):
        angle = np.pi/2 + i * (np.pi/2) / 14
        points.append([cx + 40 * np.cos(angle), cy + 40 * np.sin(angle)])

    return {
        "name": "Simple Circuit",
        "author": "F1-MARS",
        "difficulty": 1,
        "centerline": points,
        "width": 12.0,
        "pit_entry_index": 8,
        "pit_exit_index": 18,
        "start_position": [20, 0, 0.0],
        "checkpoints": [0, 25, 50, 80],
        "reference_lap_time": 28.0,
        "metadata": {
            "characteristics": ["hairpin", "mixed_speed"],
            "description": "Simple circuit with one tight hairpin"
        }
    }


def generate_technical() -> dict:
    """
    Circuito técnico con chicanes (Dificultad 2).
    Requiere precisión y buen control de velocidad.
    """
    points = []

    # Recta de meta
    for i in range(15):
        points.append([i * 10, 0])

    # Primera curva (derecha, 90°)
    cx, cy = 150, 40
    for i in range(12):
        angle = -np.pi/2 + i * (np.pi/2) / 11
        points.append([cx + 40 * np.cos(angle), cy + 40 * np.sin(angle)])

    # Chicane en S
    # Primera parte del S (izquierda)
    for i in range(8):
        points.append([190 + i * 5, 80 + i * 8])

    # Segunda parte del S (derecha)
    for i in range(8):
        points.append([230 + i * 5, 144 - i * 8])

    # Recta corta
    for i in range(8):
        points.append([270 + i * 8, 80])

    # Curva cerrada izquierda
    cx, cy = 334, 110
    for i in range(15):
        angle = -np.pi/2 - i * np.pi / 14
        points.append([cx + 30 * np.cos(angle), cy + 30 * np.sin(angle)])

    # Bajada
    for i in range(10):
        points.append([304, 110 - i * 12])

    # Curva de regreso (amplia)
    cx, cy = 200, -10
    for i in range(20):
        angle = 0 - i * np.pi / 19
        points.append([cx + 104 * np.cos(angle), cy + 60 * np.sin(angle)])

    # Última curva hacia meta
    cx, cy = 50, -10
    for i in range(12):
        angle = -np.pi + i * (np.pi/2) / 11
        points.append([cx + 50 * np.cos(angle), cy + 50 * np.sin(angle)])

    return {
        "name": "Technical Circuit",
        "author": "F1-MARS",
        "difficulty": 2,
        "centerline": points,
        "width": 11.0,
        "pit_entry_index": 5,
        "pit_exit_index": 12,
        "start_position": [20, 0, 0.0],
        "checkpoints": [0, 20, 40, 60, 80, 95],
        "reference_lap_time": 35.0,
        "metadata": {
            "characteristics": ["chicane", "technical", "narrow"],
            "description": "Technical circuit with chicane requiring precision"
        }
    }


def generate_mixed() -> dict:
    """
    Circuito mixto completo (Dificultad 3).
    Combina todos los elementos: rectas, chicanes, horquillas.
    """
    points = []

    # Recta principal larga
    for i in range(25):
        points.append([i * 12, 0])

    # Curva rápida de derechas
    cx, cy = 300, 60
    for i in range(15):
        angle = -np.pi/2 + i * (np.pi/3) / 14
        points.append([cx + 60 * np.cos(angle), cy + 60 * np.sin(angle)])

    # Recta en diagonal
    for i in range(12):
        points.append([340 + i * 6, 90 + i * 8])

    # Chicane rápida
    for i in range(6):
        points.append([412 + i * 4, 186 + i * 6])
    for i in range(6):
        points.append([436 + i * 4, 222 - i * 6])

    # Horquilla
    cx, cy = 430, 186
    for i in range(18):
        angle = np.pi/2 - i * np.pi / 17
        points.append([cx + 30 * np.cos(angle) + 30, cy + 30 * np.sin(angle)])

    # Recta de vuelta
    for i in range(15):
        points.append([430 - i * 10, 156])

    # Curvas enlazadas (eses lentas)
    cx, cy = 280, 130
    for i in range(10):
        angle = np.pi/2 + i * (np.pi/2) / 9
        points.append([cx + 26 * np.cos(angle), cy + 26 * np.sin(angle)])

    cx, cy = 254, 80
    for i in range(10):
        angle = np.pi - i * (np.pi/2) / 9
        points.append([cx + 24 * np.cos(angle), cy + 24 * np.sin(angle)])

    # Curva amplia de regreso
    cx, cy = 150, 56
    for i in range(20):
        angle = np.pi/2 - i * (2*np.pi/3) / 19
        points.append([cx + 80 * np.cos(angle), cy + 56 * np.sin(angle)])

    # Última curva hacia recta de meta
    cx, cy = 60, 0
    for i in range(12):
        angle = np.pi + i * (np.pi/2) / 11
        points.append([cx + 60 * np.cos(angle), cy + 40 * np.sin(angle)])

    return {
        "name": "Grand Prix Circuit",
        "author": "F1-MARS",
        "difficulty": 3,
        "centerline": points,
        "width": 12.0,
        "pit_entry_index": 10,
        "pit_exit_index": 22,
        "start_position": [30, 0, 0.0],
        "checkpoints": [0, 18, 36, 54, 72, 90, 108, 126],
        "reference_lap_time": 42.0,
        "metadata": {
            "characteristics": ["complete", "mixed_speed", "strategic"],
            "description": "Full GP circuit with all corner types"
        }
    }


def save_all_tracks(output_dir: str = "tracks"):
    """Genera y guarda todos los circuitos como JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    tracks = {
        "oval.json": generate_oval(),
        "simple.json": generate_simple(),
        "technical.json": generate_technical(),
        "mixed.json": generate_mixed()
    }

    for filename, track_data in tracks.items():
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            json.dump(track_data, f, indent=2)
        print(f"✓ Guardado: {filepath}")
        print(f"  - Nombre: {track_data['name']}")
        print(f"  - Dificultad: {track_data['difficulty']}")
        print(f"  - Puntos: {len(track_data['centerline'])}")
        print(f"  - Ancho: {track_data['width']}m")
        print()


if __name__ == "__main__":
    save_all_tracks()
