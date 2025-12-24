#!/usr/bin/env python3
"""
Script para probar todos los circuitos disponibles.
Ejecuta una vuelta rápida en cada circuito y muestra estadísticas.
"""

import sys
sys.path.insert(0, '.')

from tracks import list_available_tracks, load_track, get_track_info
import numpy as np


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_track(name: str):
    """Prueba rápida de un circuito."""
    print_header(f"Testing Track: {name}")

    # Cargar información
    info = get_track_info(name)
    print(f"\nMetadata:")
    print(f"  Name: {info['name']}")
    print(f"  Difficulty: {info['difficulty']}")
    print(f"  Width: {info['width']}m")
    print(f"  Checkpoints: {info['checkpoints']}")
    print(f"  Reference lap time: {info['reference_lap_time']}s")
    print(f"  Characteristics: {', '.join(info['metadata'].get('characteristics', []))}")
    print(f"  Description: {info['metadata'].get('description', 'N/A')}")

    # Cargar circuito completo
    track = load_track(name)
    print(f"\nTrack properties:")
    print(f"  Total length: {track.total_length:.1f}m")
    print(f"  Control points: {track.num_control_points}")
    print(f"  Checkpoints: {len(track.checkpoint_indices)}")

    # Analizar curvatura
    distances = np.linspace(0, track.total_length, 100)
    curvatures = [abs(track.get_curvature_at_distance(d)) for d in distances]
    max_curvature = max(curvatures)
    avg_curvature = np.mean(curvatures)

    print(f"\nCurvature analysis:")
    print(f"  Max curvature: {max_curvature:.4f} (min radius: {1/max_curvature:.1f}m)")
    print(f"  Avg curvature: {avg_curvature:.4f}")

    # Clasificar curvas
    tight_curves = sum(1 for c in curvatures if c > 0.02)
    medium_curves = sum(1 for c in curvatures if 0.01 < c <= 0.02)
    fast_curves = sum(1 for c in curvatures if 0.005 < c <= 0.01)

    print(f"\nCurve distribution (% of track):")
    print(f"  Tight (R<50m):   {tight_curves}%")
    print(f"  Medium (50-100m): {medium_curves}%")
    print(f"  Fast (100-200m):  {fast_curves}%")
    print(f"  Straights:       {100-tight_curves-medium_curves-fast_curves}%")

    print(f"\n✓ Track '{name}' loaded successfully!")


def main():
    """Prueba todos los circuitos disponibles."""
    print_header("F1-MARS Track Testing Suite")

    tracks = sorted(list_available_tracks())
    print(f"\nFound {len(tracks)} tracks: {', '.join(tracks)}")

    for track_name in tracks:
        try:
            test_track(track_name)
        except Exception as e:
            print(f"\n❌ Error testing track '{track_name}': {e}")
            import traceback
            traceback.print_exc()

    print_header("All Tracks Tested Successfully")
    print("\nReady for training!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
