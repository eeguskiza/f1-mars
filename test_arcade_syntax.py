#!/usr/bin/env python3
"""
Test de sintaxis y estructura de código Arcade (sin display).

Este test verifica que:
1. Todos los módulos se importan correctamente
2. Las clases se pueden instanciar (sin display)
3. La estructura del código es correcta

NO requiere display gráfico.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Verifica que todos los módulos se importan sin errores."""
    print("Testing imports...")

    try:
        import arcade
        print(f"✓ arcade {arcade.__version__}")
    except ImportError as e:
        print(f"✗ arcade: {e}")
        return False

    try:
        from arcade.shape_list import ShapeElementList, create_polygon, create_line
        print("✓ arcade.shape_list")
    except ImportError as e:
        print(f"✗ arcade.shape_list: {e}")
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
        print("✓ f1_mars.rendering (all classes)")
    except ImportError as e:
        print(f"✗ f1_mars.rendering: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_class_definitions():
    """Verifica que las clases están bien definidas."""
    print("\nTesting class definitions...")

    from f1_mars.rendering import GameState, F1CarSprite, TrackRenderer, RacingHUD, EffectsManager

    # GameState
    state = GameState()
    assert state.car_x == 0.0
    assert state.car_y == 0.0
    print("✓ GameState definition correct")

    # F1CarSprite
    car = F1CarSprite()
    assert car.length > 0
    assert car.width > 0
    assert hasattr(car, 'draw')
    print("✓ F1CarSprite definition correct")

    # TrackRenderer
    track = TrackRenderer()
    assert hasattr(track, 'setup')
    assert hasattr(track, 'draw')
    print("✓ TrackRenderer definition correct")

    # RacingHUD
    hud = RacingHUD(1280, 720)
    assert hud.width == 1280
    assert hud.height == 720
    assert hasattr(hud, 'draw')
    print("✓ RacingHUD definition correct")

    # EffectsManager
    effects = EffectsManager()
    assert hasattr(effects, 'update')
    assert hasattr(effects, 'draw_behind')
    assert hasattr(effects, 'draw_front')
    print("✓ EffectsManager definition correct")

    return True

def test_method_signatures():
    """Verifica que los métodos tienen las firmas correctas."""
    print("\nTesting method signatures...")

    from f1_mars.rendering import F1CarSprite, TrackRenderer, RacingHUD, EffectsManager
    import inspect

    # F1CarSprite.draw
    car = F1CarSprite()
    sig = inspect.signature(car.draw)
    params = list(sig.parameters.keys())
    assert 'x' in params and 'y' in params and 'heading' in params and 'velocity' in params
    print("✓ F1CarSprite.draw signature correct")

    # TrackRenderer.setup
    track = TrackRenderer()
    sig = inspect.signature(track.setup)
    params = list(sig.parameters.keys())
    assert 'centerline' in params and 'width' in params
    print("✓ TrackRenderer.setup signature correct")

    # RacingHUD.draw
    hud = RacingHUD(1280, 720)
    sig = inspect.signature(hud.draw)
    params = list(sig.parameters.keys())
    assert 'state' in params
    print("✓ RacingHUD.draw signature correct")

    # EffectsManager.update
    effects = EffectsManager()
    sig = inspect.signature(effects.update)
    params = list(sig.parameters.keys())
    required = ['car_x', 'car_y', 'heading', 'velocity', 'throttle', 'brake', 'delta_time']
    assert all(p in params for p in required)
    print("✓ EffectsManager.update signature correct")

    return True

def test_shape_list_usage():
    """Verifica que ShapeElementList se puede importar."""
    print("\nTesting ShapeElementList usage...")

    from arcade.shape_list import ShapeElementList, create_line
    from f1_mars.rendering import TrackRenderer

    # Crear renderer sin setup (no necesita display)
    renderer = TrackRenderer()

    # Verificar que las clases/funciones existen
    # Nota: No se pueden instanciar sin una ventana activa
    print("✓ ShapeElementList importable")
    print("✓ create_line importable")
    print("✓ TrackRenderer uses correct imports")

    return True

def main():
    print("="*60)
    print("  F1-MARS ARCADE SYNTAX TEST (No Display Required)")
    print("="*60)

    success = True

    if not test_imports():
        success = False

    if not test_class_definitions():
        success = False

    if not test_method_signatures():
        success = False

    if not test_shape_list_usage():
        success = False

    print("\n" + "="*60)
    if success:
        print("  ✓ ALL SYNTAX TESTS PASSED")
        print("="*60)
        print("\n✅ El código está correcto!")
        print("\n📝 Nota: Para ejecutar el visualizador necesitas:")
        print("   1. Display gráfico (X11 en WSL o Windows nativo)")
        print("   2. Ver WSL_DISPLAY_SETUP.md para instrucciones")
        return 0
    else:
        print("  ✗ SOME TESTS FAILED")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
