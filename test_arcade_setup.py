#!/usr/bin/env python3
"""
Script de prueba para verificar la configuración de Arcade.

Uso:
    python test_arcade_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Prueba que todos los módulos se importan correctamente."""
    print("Testing imports...")

    try:
        import arcade
        print("✓ arcade imported")
    except ImportError as e:
        print(f"✗ arcade import failed: {e}")
        return False

    try:
        from f1_mars.rendering import (
            F1MarsWindow,
            GameState,
            RacingCamera,
            F1CarSprite,
            TrackRenderer,
            RacingHUD,
            EffectsManager
        )
        print("✓ f1_mars.rendering imported")
    except ImportError as e:
        print(f"✗ f1_mars.rendering import failed: {e}")
        return False

    try:
        from f1_mars.envs import F1Env
        print("✓ f1_mars.envs imported")
    except ImportError as e:
        print(f"✗ f1_mars.envs import failed: {e}")
        return False

    return True

def test_classes():
    """Prueba que las clases se pueden instanciar."""
    print("\nTesting class instantiation...")

    try:
        from f1_mars.rendering import GameState
        state = GameState()
        print(f"✓ GameState created: car_x={state.car_x}, car_y={state.car_y}")
    except Exception as e:
        print(f"✗ GameState instantiation failed: {e}")
        return False

    try:
        from f1_mars.rendering import RacingCamera
        # Note: RacingCamera requires an active arcade window
        print(f"✓ RacingCamera class available (requires active window for instantiation)")
    except Exception as e:
        print(f"✗ RacingCamera import failed: {e}")
        return False

    try:
        from f1_mars.rendering import F1CarSprite
        car = F1CarSprite()
        print(f"✓ F1CarSprite created: length={car.length}, width={car.width}")
    except Exception as e:
        print(f"✗ F1CarSprite instantiation failed: {e}")
        return False

    try:
        from f1_mars.rendering import TrackRenderer
        track = TrackRenderer()
        print(f"✓ TrackRenderer created")
    except Exception as e:
        print(f"✗ TrackRenderer instantiation failed: {e}")
        return False

    try:
        from f1_mars.rendering import RacingHUD
        hud = RacingHUD(1280, 720)
        print(f"✓ RacingHUD created")
    except Exception as e:
        print(f"✗ RacingHUD instantiation failed: {e}")
        return False

    try:
        from f1_mars.rendering import EffectsManager
        effects = EffectsManager()
        print(f"✓ EffectsManager created")
    except Exception as e:
        print(f"✗ EffectsManager instantiation failed: {e}")
        return False

    return True

def test_environment():
    """Prueba que el entorno se puede crear."""
    print("\nTesting F1Env creation...")

    try:
        from f1_mars.envs import F1Env
        env = F1Env(max_laps=1)
        print(f"✓ F1Env created")

        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        track_points = env.track.control_points if hasattr(env.track, 'control_points') else env.track.centerline
        print(f"  Track points: {len(track_points)}")

        # Get track width
        if hasattr(env.track, 'width'):
            track_width = env.track.width
        elif hasattr(env.track, 'widths'):
            import numpy as np
            track_width = float(np.mean(env.track.widths))
        else:
            track_width = 12.0  # Default
        print(f"  Track width: {track_width}")

        # Verify car attributes
        car_x = env.car.position[0]
        car_y = env.car.position[1]
        print(f"  Car position: ({car_x:.2f}, {car_y:.2f})")
        print(f"  Car heading: {env.car.heading:.2f}")
        print(f"  Car velocity: {env.car.velocity:.2f}")

        env.close()
        print(f"✓ Environment closed")

    except Exception as e:
        print(f"✗ F1Env test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def main():
    print("="*60)
    print("  F1-MARS ARCADE SETUP TEST")
    print("="*60)

    success = True

    if not test_imports():
        success = False

    if not test_classes():
        success = False

    if not test_environment():
        success = False

    print("\n" + "="*60)
    if success:
        print("  ✓ ALL TESTS PASSED")
        print("="*60)
        print("\nReady to use! Try:")
        print("  python scripts/watch_agent.py --model trained_models/PPO_default_final.zip")
        return 0
    else:
        print("  ✗ SOME TESTS FAILED")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
