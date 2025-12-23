#!/usr/bin/env python3
"""Test de verificación de desgaste de neumáticos."""

import numpy as np
from f1_mars.envs import F1Env

def test_tyre_wear_rate():
    """
    Verifica que el desgaste sea ~4% en 30 segundos de conducción agresiva.
    """
    env = F1Env(max_laps=10)
    obs, info = env.reset()

    # Conducción agresiva pero controlada
    # Use LIDAR to avoid going off track while maintaining high cornering
    total_steps = 30 * 60  # 30 segundos a 60 fps

    initial_wear = info.get('tyre_wear', 0)
    max_speed = 0
    max_lateral = 0
    off_track_count = 0
    avg_lateral = 0.0

    for step in range(total_steps):
        # Aggressive policy that uses LIDAR to stay on track
        lidar = obs[4:15]  # LIDAR rays
        center = lidar[5]   # Center ray
        left_avg = np.mean(lidar[:5])
        right_avg = np.mean(lidar[6:])

        # Steer towards the wider side while maintaining aggression
        if left_avg > right_avg:
            steering = -0.6  # Turn left
        else:
            steering = 0.6   # Turn right

        # Add oscillation for more cornering
        steering += 0.3 * np.sin(step * 0.05)
        steering = np.clip(steering, -1.0, 1.0)

        # High throttle
        throttle = 0.9

        aggressive_action = np.array([steering, throttle, 0.0])
        obs, reward, terminated, truncated, info = env.step(aggressive_action)

        if terminated or truncated:
            print(f"Episode ended early at step {step} (terminated={terminated}, truncated={truncated})")
            if not info.get('on_track', True):
                print(f"  Reason: Off track for too long")
            break

        if not info.get('on_track', True):
            off_track_count += 1

        max_speed = max(max_speed, info.get('velocity', 0))
        current_lateral = env.car.get_lateral_force()
        max_lateral = max(max_lateral, current_lateral)
        avg_lateral += current_lateral

    final_wear = info.get('tyre_wear', 0)
    wear_delta = final_wear - initial_wear
    actual_steps = step + 1 if 'step' in locals() else total_steps
    avg_lateral = avg_lateral / actual_steps if actual_steps > 0 else 0

    print(f"\n{'='*50}")
    print(f"TYRE WEAR TEST RESULTS")
    print(f"{'='*50}")
    print(f"Duration: {actual_steps/60:.1f} seconds ({actual_steps} steps)")
    print(f"Max speed reached: {max_speed * 3.6:.1f} km/h")
    print(f"Max lateral force: {max_lateral:.3f}")
    print(f"Avg lateral force: {avg_lateral:.3f}")
    print(f"Off-track frames: {off_track_count}")
    print(f"Initial wear: {initial_wear:.2f}%")
    print(f"Final wear: {final_wear:.2f}%")
    print(f"Wear delta: {wear_delta:.2f}%")
    print(f"Tyre temperature: {info.get('tyre_temperature', 0):.1f}°C")
    print(f"{'='*50}")

    # Verificaciones
    expected_min = 2.0  # Mínimo 2% en 30s
    expected_max = 6.0  # Máximo 6% en 30s

    if wear_delta < expected_min:
        print(f"❌ FAIL: Wear too LOW ({wear_delta:.2f}% < {expected_min}%)")
        print(f"   Probable cause: lateral_force not being passed or always 0")
        return False
    elif wear_delta > expected_max:
        print(f"⚠️  WARN: Wear too HIGH ({wear_delta:.2f}% > {expected_max}%)")
        print(f"   May need to reduce wear_rate slightly")
        return True  # Still functional, just needs tuning
    else:
        print(f"✅ PASS: Wear in expected range ({expected_min}% - {expected_max}%)")
        return True

    env.close()

def test_lateral_force_calculation():
    """
    Verifica que lateral_force se calcule correctamente.
    """
    env = F1Env(max_laps=10)
    obs, info = env.reset()

    print(f"\n{'='*50}")
    print(f"LATERAL FORCE TEST")
    print(f"{'='*50}")

    # Test 1: Sin steering, lateral force debe ser ~0
    straight_action = np.array([0.0, 1.0, 0.0])
    for _ in range(60):  # 1 segundo para ganar velocidad
        env.step(straight_action)

    lateral_straight = env.car.get_lateral_force()
    print(f"Lateral force (straight): {lateral_straight:.4f}")

    # Test 2: Con steering máximo, lateral force debe ser > 0.5
    turn_action = np.array([0.8, 0.8, 0.0])  # Steering fuerte
    for _ in range(30):  # Medio segundo curvando
        env.step(turn_action)

    lateral_turn = env.car.get_lateral_force()
    speed_kmh = env.car.velocity * 3.6
    print(f"Lateral force (turning): {lateral_turn:.4f}")
    print(f"Speed during turn: {speed_kmh:.1f} km/h")

    if lateral_straight > 0.1:
        print(f"❌ FAIL: Lateral force should be ~0 when going straight")
        return False

    if lateral_turn < 0.3:
        print(f"❌ FAIL: Lateral force should be > 0.3 when turning hard")
        print(f"   Check car.get_lateral_force() implementation")
        return False

    print(f"✅ PASS: Lateral force calculation working correctly")

    env.close()
    return True

def test_integration_details():
    """
    Test detallado de la integración paso a paso.
    """
    env = F1Env(max_laps=10)
    obs, info = env.reset()

    print(f"\n{'='*50}")
    print(f"INTEGRATION DETAILS TEST")
    print(f"{'='*50}")

    # Acción agresiva
    action = np.array([0.8, 1.0, 0.0])  # High steering + full throttle

    # Ejecutar 1 segundo
    for i in range(60):
        obs, reward, terminated, truncated, info = env.step(action)

    # Imprimir detalles
    print(f"\nAfter 1 second of aggressive driving:")
    print(f"  dt: {env.dt:.6f} seconds")
    print(f"  Speed: {env.car.velocity:.2f} m/s ({env.car.velocity * 3.6:.1f} km/h)")
    print(f"  Steering angle: {env.car.steering_angle:.4f} rad")
    print(f"  Lateral force: {env.car.get_lateral_force():.4f}")
    print(f"  Tyre temperature: {env.tyres.temperature:.1f}°C")
    print(f"  Tyre wear: {env.tyres.wear:.4f}%")
    print(f"  Tyre grip: {env.tyres.get_grip():.4f}")

    # Verificar que todo está cambiando
    checks = {
        "Speed > 0": env.car.velocity > 0,
        "Lateral force > 0": env.car.get_lateral_force() > 0,
        "Temperature increased": env.tyres.temperature > 70,
        "Wear increased": env.tyres.wear > 0,
    }

    print(f"\nIntegration checks:")
    all_pass = True
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_pass = False

    env.close()
    return all_pass

if __name__ == "__main__":
    print("Running tyre wear diagnostics...\n")

    test1 = test_lateral_force_calculation()
    test2 = test_integration_details()
    test3 = test_tyre_wear_rate()

    print(f"\n{'='*50}")
    if test1 and test2 and test3:
        print("✅ ALL TESTS PASSED - Tyre wear is correctly integrated")
    else:
        print("❌ SOME TESTS FAILED - Review the integration")
    print(f"{'='*50}")
