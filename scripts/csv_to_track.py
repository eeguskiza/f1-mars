#!/usr/bin/env python3
"""
Convierte archivos CSV de circuitos a formato JSON para F1-MARS.

Formato CSV esperado:
    # x_m,y_m,w_tr_right_m,w_tr_left_m
    x,y,ancho_derecha,ancho_izquierda
    ...

Uso:
    python scripts/csv_to_track.py tracks/csv/Budapest.csv
    python scripts/csv_to_track.py tracks/csv/Budapest.csv --output tracks/budapest.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np


def simplify_path(points: List[List[float]], tolerance: float = 2.0) -> List[List[float]]:
    """
    Simplifica un path usando el algoritmo Douglas-Peucker.

    Args:
        points: Lista de [x, y] coordenadas
        tolerance: Tolerancia en metros (mayor = más simplificación)

    Returns:
        Path simplificado
    """
    if len(points) < 3:
        return points

    def perpendicular_distance(point, line_start, line_end):
        """Calcula distancia perpendicular de un punto a una línea."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)

        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx**2 + dy**2)

    def douglas_peucker(points, tolerance):
        """Implementación recursiva de Douglas-Peucker."""
        if len(points) < 3:
            return points

        # Encontrar punto más alejado de la línea entre primero y último
        max_dist = 0
        max_index = 0

        for i in range(1, len(points) - 1):
            dist = perpendicular_distance(points[i], points[0], points[-1])
            if dist > max_dist:
                max_dist = dist
                max_index = i

        # Si el punto más alejado está más lejos que la tolerancia, dividir recursivamente
        if max_dist > tolerance:
            left = douglas_peucker(points[:max_index + 1], tolerance)
            right = douglas_peucker(points[max_index:], tolerance)
            return left[:-1] + right
        else:
            return [points[0], points[-1]]

    simplified = douglas_peucker(points, tolerance)
    return simplified


def parse_csv(csv_path: str, simplify: bool = True, tolerance: float = 2.0) -> Tuple[List[List[float]], List[float]]:
    """
    Parsea el CSV del circuito.

    Args:
        csv_path: Ruta al CSV
        simplify: Si True, simplifica el path con Douglas-Peucker
        tolerance: Tolerancia para simplificación en metros

    Returns:
        centerline: Lista de [x, y] coordenadas
        widths: Lista de anchos totales (right + left)
    """
    centerline = []
    widths = []

    with open(csv_path, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                parts = line.split(',')
                x = float(parts[0])
                y = float(parts[1])
                w_right = float(parts[2])
                w_left = float(parts[3])

                centerline.append([x, y])
                widths.append(w_right + w_left)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid line: {line}", file=sys.stderr)
                continue

    original_count = len(centerline)
    original_centerline = [p[:] for p in centerline]  # Copiar antes de simplificar
    original_widths = widths[:]

    # Simplificar path si está habilitado
    if simplify and len(centerline) > 100:
        print(f"  Simplifying path (tolerance={tolerance}m)...")
        centerline = simplify_path(centerline, tolerance)

        # Interpolar widths para los nuevos puntos
        new_widths = []
        for point in centerline:
            # Encontrar punto original más cercano
            min_dist = float('inf')
            closest_idx = 0
            for i, orig_point in enumerate(original_centerline):
                dx = point[0] - orig_point[0]
                dy = point[1] - orig_point[1]
                dist = dx*dx + dy*dy
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            new_widths.append(original_widths[min(closest_idx, len(original_widths) - 1)])

        widths = new_widths
        print(f"  Reduced from {original_count} to {len(centerline)} points ({100*(1-len(centerline)/original_count):.1f}% reduction)")

    return centerline, widths


def calculate_track_length(centerline: List[List[float]]) -> float:
    """Calcula la longitud total del circuito."""
    total_length = 0.0

    for i in range(len(centerline) - 1):
        x1, y1 = centerline[i]
        x2, y2 = centerline[i + 1]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_length += distance

    # Add closing segment
    if len(centerline) > 0:
        x1, y1 = centerline[-1]
        x2, y2 = centerline[0]
        total_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return total_length


def generate_checkpoints(num_points: int, num_checkpoints: int = 20) -> List[int]:
    """Genera índices de checkpoints espaciados uniformemente."""
    if num_points <= num_checkpoints:
        return list(range(num_points))

    step = num_points / num_checkpoints
    return [int(i * step) for i in range(num_checkpoints)]


def create_track_json(csv_path: str, output_path: str = None,
                      total_laps: int = 5,
                      reference_lap_time: float = None,
                      simplify: bool = True,
                      tolerance: float = 2.0):
    """
    Crea el archivo JSON del circuito a partir del CSV.

    Args:
        csv_path: Ruta al archivo CSV
        output_path: Ruta de salida (opcional, por defecto usa mismo nombre)
        total_laps: Número de vueltas por defecto
        reference_lap_time: Tiempo de referencia (None = auto-calculate)
    """
    # Parse CSV
    print(f"Reading {csv_path}...")
    centerline, widths = parse_csv(csv_path, simplify=simplify, tolerance=tolerance)

    if not centerline:
        print("Error: No valid data found in CSV", file=sys.stderr)
        sys.exit(1)

    # Calculate stats
    track_length = calculate_track_length(centerline)
    avg_width = sum(widths) / len(widths)

    print(f"  Points: {len(centerline)}")
    print(f"  Length: {track_length:.2f}m ({track_length/1000:.3f} km)")
    print(f"  Avg width: {avg_width:.2f}m")

    # Estimate lap time if not provided (assuming ~150 km/h average)
    if reference_lap_time is None:
        avg_speed_kmh = 150
        avg_speed_ms = avg_speed_kmh / 3.6
        reference_lap_time = track_length / avg_speed_ms

    # Generate checkpoints
    checkpoints = generate_checkpoints(len(centerline))

    # Determine circuit name from filename
    circuit_name = Path(csv_path).stem

    # Find pit lane area (typically in the main straight, first ~20% of track)
    pit_start_idx = int(len(centerline) * 0.05)
    pit_end_idx = int(len(centerline) * 0.15)

    pit_entry = centerline[pit_start_idx].copy()
    pit_exit = centerline[pit_end_idx].copy()

    # Offset pit lane to the right
    if pit_start_idx < len(centerline) - 1:
        dx = centerline[pit_start_idx + 1][0] - centerline[pit_start_idx][0]
        dy = centerline[pit_start_idx + 1][1] - centerline[pit_start_idx][1]
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            # Perpendicular to the right
            pit_offset = 8.0  # meters
            nx = -dy / length
            ny = dx / length
            pit_entry[0] += nx * pit_offset
            pit_entry[1] += ny * pit_offset
            pit_exit[0] += nx * pit_offset
            pit_exit[1] += ny * pit_offset

    # Create pit lane path
    num_pit_points = 10
    pit_lane = []
    for i in range(num_pit_points + 1):
        t = i / num_pit_points
        x = pit_entry[0] + t * (pit_exit[0] - pit_entry[0])
        y = pit_entry[1] + t * (pit_exit[1] - pit_entry[1])
        pit_lane.append([round(x, 2), round(y, 2)])

    # Start position debe estar EN la pista (primer punto de centerline)
    start_x, start_y = centerline[0]
    # Calcular heading inicial (dirección hacia el segundo punto)
    if len(centerline) > 1:
        dx = centerline[1][0] - centerline[0][0]
        dy = centerline[1][1] - centerline[0][1]
        start_heading = math.atan2(dy, dx)
    else:
        start_heading = 0.0

    # Create track JSON
    track_data = {
        "name": f"{circuit_name} - Real F1 Circuit",
        "description": f"Circuito real de F1. Longitud: {track_length/1000:.2f} km, Ancho medio: {avg_width:.1f}m",
        "total_laps": total_laps,
        "reference_lap_time": round(reference_lap_time, 1),
        "width": round(avg_width, 1),
        "centerline": [[round(x, 2), round(y, 2)] for x, y in centerline],
        "checkpoints": checkpoints,
        "start_position": [round(start_x, 2), round(start_y, 2), round(start_heading, 2)],
        "pit_entry": [round(pit_entry[0], 2), round(pit_entry[1], 2)],
        "pit_exit": [round(pit_exit[0], 2), round(pit_exit[1], 2)],
        "pit_lane": pit_lane
    }

    # Determine output path
    if output_path is None:
        output_path = f"tracks/{circuit_name.lower()}.json"

    # Save JSON
    print(f"\nSaving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(track_data, f, indent=2)

    print(f"✓ Circuit created successfully!")
    print(f"\nTo view the circuit, run:")
    print(f"  python scripts/watch_agent.py \\")
    print(f"    --model trained_models/best_model.zip \\")
    print(f"    --track {output_path} \\")
    print(f"    --laps {total_laps}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert F1 circuit CSV to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Budapest circuit
  python scripts/csv_to_track.py tracks/csv/Budapest.csv

  # Specify output path
  python scripts/csv_to_track.py tracks/csv/Budapest.csv -o tracks/budapest.json

  # Custom lap settings
  python scripts/csv_to_track.py tracks/csv/Budapest.csv --laps 10 --ref-time 95.5
        """
    )

    parser.add_argument('csv_file', help='Path to CSV file with circuit data')
    parser.add_argument('-o', '--output', help='Output JSON path (default: tracks/<name>.json)')
    parser.add_argument('--laps', type=int, default=5, help='Total laps (default: 5)')
    parser.add_argument('--ref-time', type=float, help='Reference lap time in seconds (default: auto-calculate)')
    parser.add_argument('--no-simplify', action='store_true', help='Disable path simplification (keep all points)')
    parser.add_argument('--tolerance', type=float, default=2.0, help='Simplification tolerance in meters (default: 2.0, higher=more reduction)')

    args = parser.parse_args()

    # Check if CSV exists
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    # Create track
    create_track_json(
        args.csv_file,
        args.output,
        args.laps,
        args.ref_time,
        simplify=not args.no_simplify,
        tolerance=args.tolerance
    )


if __name__ == '__main__':
    main()
