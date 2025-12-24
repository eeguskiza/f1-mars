#!/usr/bin/env python3
"""
Demostración del curriculum learning con los 4 circuitos.
Ejecuta un agente simple en cada circuito mostrando progresión de dificultad.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from f1_mars.envs import F1Env
from tracks import list_available_tracks, get_tracks_by_difficulty, TRACKS_DIR


def simple_controller(obs, track_difficulty):
    """
    Controlador simple que se adapta a la dificultad del circuito.

    Args:
        obs: Observación del entorno (26 dims)
        track_difficulty: Nivel de dificultad del circuito (0-3)

    Returns:
        action: [steering, throttle, brake]
    """
    # Extraer datos relevantes
    lidar = obs[4:15]  # Sensores LIDAR
    velocity = obs[15]  # Velocidad (normalizada)

    # Calcular distancias
    left_distance = np.mean(lidar[:5])
    right_distance = np.mean(lidar[6:])
    center_distance = lidar[5]

    # Steering: seguir el centro de la pista
    if left_distance < right_distance:
        steering = 0.3  # Girar a la derecha
    else:
        steering = -0.3  # Girar a la izquierda

    # Ajustar agresividad según dificultad
    steering *= (1.0 + track_difficulty * 0.2)
    steering = np.clip(steering, -1.0, 1.0)

    # Throttle y brake: adaptar a distancia frontal y dificultad
    safety_threshold = 0.4 - (track_difficulty * 0.05)

    if center_distance < safety_threshold:
        # Zona peligrosa: frenar
        throttle = 0.2
        brake = 0.3 + (track_difficulty * 0.1)
    else:
        # Zona segura: acelerar
        throttle = 0.7 - (track_difficulty * 0.1)
        brake = 0.0

    return np.array([steering, throttle, brake], dtype=np.float32)


def test_track_curriculum(track_name: str, difficulty: int, max_steps: int = 500):
    """
    Prueba un circuito del curriculum.

    Args:
        track_name: Nombre del circuito
        difficulty: Nivel de dificultad
        max_steps: Máximo de pasos a ejecutar
    """
    print("\n" + "=" * 70)
    print(f"  CIRCUITO: {track_name.upper()} (Dificultad {difficulty})")
    print("=" * 70)

    # Cargar entorno
    track_path = str(TRACKS_DIR / f"{track_name}.json")
    env = F1Env(track_path=track_path, max_laps=1)

    print(f"\nInicializando entorno...")
    obs, info = env.reset()

    print(f"✓ Posición inicial: [{info['position_x']:.1f}, {info['position_y']:.1f}]")
    print(f"✓ Longitud del circuito: {info['track_length']:.1f}m")
    print(f"\nEjecutando {max_steps} pasos con controlador simple...")

    # Ejecutar episodio
    episode_reward = 0.0
    steps = 0
    completed = False

    while steps < max_steps:
        # Obtener acción del controlador
        action = simple_controller(obs, difficulty)

        # Ejecutar paso
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

        # Mostrar progreso cada 100 pasos
        if steps % 100 == 0:
            progress = info['lap_progress'] * 100
            speed_kmh = info['velocity_kmh']
            print(f"  Paso {steps:3d} | "
                  f"Progreso: {progress:5.1f}% | "
                  f"Velocidad: {speed_kmh:6.1f} km/h | "
                  f"Reward: {episode_reward:8.1f}")

        # Verificar si terminó
        if terminated or truncated:
            if info['lap_progress'] >= 0.95:
                completed = True
                print(f"\n🏁 ¡Vuelta completada en {steps} pasos!")
            else:
                print(f"\n❌ Episodio terminado en paso {steps}")
                if not info['on_track']:
                    print(f"   Razón: Fuera de pista")
                else:
                    print(f"   Razón: Otro (desgaste, etc.)")
            break

    # Mostrar resumen
    print(f"\nResumen:")
    print(f"  Pasos totales:     {steps}")
    print(f"  Reward total:      {episode_reward:.1f}")
    print(f"  Progreso final:    {info['lap_progress']*100:.1f}%")
    print(f"  Velocidad final:   {info['velocity_kmh']:.1f} km/h")
    print(f"  Desgaste neumáticos: {info['tyre_wear']:.1f}%")
    print(f"  Estado:            {'✓ COMPLETADO' if completed else '✗ INCOMPLETO'}")

    env.close()
    return completed, episode_reward, steps


def main():
    """Ejecuta el curriculum completo."""
    print("\n" + "=" * 70)
    print("  F1-MARS CURRICULUM LEARNING DEMO")
    print("=" * 70)

    print("\nEste demo ejecuta un controlador simple en los 4 circuitos")
    print("para demostrar la progresión de dificultad del curriculum.\n")

    # Obtener circuitos ordenados por dificultad
    results = []

    for difficulty in range(4):
        track_names = get_tracks_by_difficulty(difficulty)
        if not track_names:
            continue

        track_name = track_names[0]
        completed, reward, steps = test_track_curriculum(track_name, difficulty)
        results.append({
            'name': track_name,
            'difficulty': difficulty,
            'completed': completed,
            'reward': reward,
            'steps': steps
        })

    # Resumen final
    print("\n" + "=" * 70)
    print("  RESUMEN DEL CURRICULUM")
    print("=" * 70)

    print(f"\n{'Circuito':<15} {'Dif.':<5} {'Estado':<12} {'Reward':<10} {'Pasos'}")
    print("-" * 70)

    for r in results:
        status = "✓ Completado" if r['completed'] else "✗ Incompleto"
        print(f"{r['name']:<15} {r['difficulty']:<5} {status:<12} "
              f"{r['reward']:>8.1f}  {r['steps']:>6}")

    # Calcular tasa de éxito
    completed_count = sum(1 for r in results if r['completed'])
    success_rate = (completed_count / len(results)) * 100 if results else 0

    print("\n" + "=" * 70)
    print(f"Tasa de éxito: {completed_count}/{len(results)} ({success_rate:.0f}%)")
    print("=" * 70)
    print("\n✓ Demo completado. Listo para entrenamiento con RL!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
