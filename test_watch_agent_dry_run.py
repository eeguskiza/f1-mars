#!/usr/bin/env python3
"""
Dry run test del watch_agent sin abrir ventana.

Verifica que todo el código de inicialización funciona correctamente.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_watch_agent_initialization():
    """Test de inicialización sin abrir ventana."""
    print("Testing watch_agent initialization...")

    try:
        from stable_baselines3 import PPO
        from f1_mars.envs import F1Env
        import numpy as np
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    # Cargar modelo
    model_path = "trained_models/PPO_default_final.zip"
    if not Path(model_path).exists():
        print(f"⚠ Model not found: {model_path}")
        print("  Skipping model test")
        model = None
    else:
        try:
            model = PPO.load(model_path)
            print(f"✓ Model loaded: {model_path}")
        except Exception as e:
            print(f"✗ Model load failed: {e}")
            return False

    # Crear entorno
    try:
        env = F1Env(max_laps=3)
        print("✓ Environment created")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False

    # Reset
    try:
        obs, info = env.reset()
        print(f"✓ Environment reset (obs shape: {obs.shape})")
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        return False

    # Verificar datos del track
    try:
        if hasattr(env.track, 'centerline'):
            track_centerline = env.track.centerline.tolist()
        elif hasattr(env.track, 'control_points'):
            track_centerline = env.track.control_points.tolist() if hasattr(env.track.control_points, 'tolist') else env.track.control_points
        else:
            raise AttributeError("Track has no centerline or control_points")

        if hasattr(env.track, 'width'):
            track_width = env.track.width
        elif hasattr(env.track, 'widths'):
            track_width = float(np.mean(env.track.widths))
        else:
            track_width = 12.0

        print(f"✓ Track data extracted: {len(track_centerline)} points, width={track_width}")
    except Exception as e:
        print(f"✗ Track data extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verificar car attributes
    try:
        car_x = env.car.position[0]
        car_y = env.car.position[1]
        car_heading = env.car.heading
        car_velocity = env.car.velocity
        print(f"✓ Car state: pos=({car_x:.1f}, {car_y:.1f}), heading={car_heading:.2f}, vel={car_velocity:.2f}")
    except Exception as e:
        print(f"✗ Car state extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verificar tyres
    try:
        tyre_compound = env.tyres.compound.name
        tyre_wear = env.tyres.wear
        tyre_temp = env.tyres.temperature
        print(f"✓ Tyre state: compound={tyre_compound}, wear={tyre_wear:.1f}%, temp={tyre_temp:.1f}°C")
    except Exception as e:
        print(f"✗ Tyre state extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test de predicción del modelo
    if model is not None:
        try:
            action, _ = model.predict(obs, deterministic=True)
            print(f"✓ Model prediction works (action shape: {action.shape})")
        except Exception as e:
            print(f"✗ Model prediction failed: {e}")
            return False

    # Test de step
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step works (reward={reward:.2f})")
    except Exception as e:
        print(f"✗ Environment step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Cleanup
    env.close()
    print("✓ Environment closed")

    return True

def main():
    print("="*60)
    print("  WATCH_AGENT DRY RUN TEST")
    print("="*60)
    print()

    success = test_watch_agent_initialization()

    print()
    print("="*60)
    if success:
        print("  ✓ ALL INITIALIZATION TESTS PASSED")
        print("="*60)
        print()
        print("✅ watch_agent.py está listo para ejecutar!")
        print()
        print("📝 Próximo paso: Configurar display para WSL")
        print("   Ver: WSL_DISPLAY_SETUP.md")
        return 0
    else:
        print("  ✗ SOME TESTS FAILED")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
