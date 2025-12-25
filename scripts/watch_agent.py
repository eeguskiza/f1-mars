#!/usr/bin/env python3
"""
Visualizar un agente entrenado conduciendo.

Uso:
    python scripts/watch_agent.py --model trained_models/PPO_default_final.zip
    python scripts/watch_agent.py --model trained_models/best_model.zip --track tracks/oval.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, SAC, TD3
from f1_mars.envs import F1Env
from f1_mars.rendering import F1Renderer


def detect_algorithm(model_path: str) -> str:
    """Detecta el algoritmo usado en el modelo."""
    path = Path(model_path)
    name = path.stem.upper()

    if 'PPO' in name:
        return 'PPO'
    elif 'SAC' in name:
        return 'SAC'
    elif 'TD3' in name:
        return 'TD3'
    else:
        # Intentar cargar con cada uno
        for algo_name, algo_class in [('PPO', PPO), ('SAC', SAC), ('TD3', TD3)]:
            try:
                algo_class.load(model_path)
                return algo_name
            except:
                continue
        return 'PPO'  # Default


def main():
    parser = argparse.ArgumentParser(description='Visualizar agente entrenado')
    parser.add_argument('--model', type=str, required=True,
                       help='Ruta al modelo entrenado (.zip)')
    parser.add_argument('--track', type=str, default=None,
                       help='Ruta al circuito (.json). Default: oval generado')
    parser.add_argument('--laps', type=int, default=3,
                       help='Número de vueltas (default: 3)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frames por segundo (default: 60)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Ancho de ventana (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Alto de ventana (default: 720)')
    parser.add_argument('--zoom', type=float, default=2.0,
                       help='Zoom inicial (default: 2.0)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Usar política determinista (default: True)')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Número de episodios a visualizar (default: 1)')

    args = parser.parse_args()

    # Verificar que el modelo existe
    if not Path(args.model).exists():
        print(f"Error: Modelo no encontrado: {args.model}")
        sys.exit(1)

    # Detectar algoritmo
    algo_name = detect_algorithm(args.model)
    algo_class = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3}[algo_name]

    print(f"\n{'='*60}")
    print(f"  F1-MARS AGENT VIEWER")
    print(f"{'='*60}")
    print(f"Model:      {args.model}")
    print(f"Algorithm:  {algo_name}")
    print(f"Track:      {args.track or 'Default oval'}")
    print(f"Laps:       {args.laps}")
    print(f"{'='*60}\n")

    # Cargar modelo
    print("Loading model...")
    model = algo_class.load(args.model)
    print("✓ Model loaded")

    # Crear entorno
    print("Creating environment...")
    if args.track:
        env = F1Env(track_path=args.track, max_laps=args.laps)
    else:
        env = F1Env(max_laps=args.laps)
    print("✓ Environment created")

    # Crear renderizador
    print("Initializing renderer...")
    renderer = F1Renderer(
        width=args.width,
        height=args.height,
        title=f"F1-MARS - {Path(args.model).stem}"
    )
    renderer.zoom = args.zoom
    print("✓ Renderer initialized")

    print("\n" + "="*60)
    print("  CONTROLS")
    print("="*60)
    print("  SPACE  - Pause/Resume")
    print("  R      - Reset episode")
    print("  H      - Toggle HUD")
    print("  T      - Toggle trajectory")
    print("  +/-    - Zoom in/out")
    print("  ESC    - Quit")
    print("="*60 + "\n")

    # Loop de visualización
    for episode in range(args.episodes):
        print(f"\n--- Episode {episode + 1}/{args.episodes} ---")

        obs, info = env.reset()
        renderer.trajectory.clear()

        total_reward = 0
        step = 0

        while renderer.running:
            # Procesar eventos
            actions = renderer.handle_events()

            if actions['quit']:
                break
            if actions['reset']:
                obs, info = env.reset()
                renderer.trajectory.clear()
                total_reward = 0
                step = 0
                continue

            # Si está pausado, solo renderizar
            if renderer.paused:
                # Preparar datos para render
                car_state = {
                    'x': env.car.position[0],
                    'y': env.car.position[1],
                    'heading': env.car.heading,
                    'velocity': env.car.velocity
                }

                track_data = {
                    'centerline': env.track.control_points.tolist() if hasattr(env.track.control_points, 'tolist') else env.track.control_points,
                    'width': float(np.mean(env.track.widths)),
                }

                telemetry = {
                    'velocity': env.car.velocity,
                    'lap': info.get('lap', 1),
                    'total_laps': args.laps,
                    'lap_time': info.get('lap_time', 0),
                    'best_lap_time': info.get('best_lap_time'),
                    'tyre_compound': env.tyres.compound.name,
                    'tyre_wear': env.tyres.wear,
                    'tyre_temp': env.tyres.temperature,
                    'on_track': info.get('on_track', True),
                }

                renderer.render(car_state, track_data, telemetry)
                renderer.tick(args.fps)
                continue

            # Obtener acción del modelo
            action, _ = model.predict(obs, deterministic=args.deterministic)

            # Ejecutar step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Preparar datos para render
            car_state = {
                'x': env.car.position[0],
                'y': env.car.position[1],
                'heading': env.car.heading,
                'velocity': env.car.velocity
            }

            track_data = {
                'centerline': env.track.control_points.tolist() if hasattr(env.track.control_points, 'tolist') else env.track.control_points,
                'width': float(np.mean(env.track.widths)),
            }

            telemetry = {
                'velocity': env.car.velocity,
                'lap': info.get('lap', 1),
                'total_laps': args.laps,
                'lap_time': info.get('lap_time', 0),
                'best_lap_time': info.get('best_lap_time'),
                'tyre_compound': env.tyres.compound.name,
                'tyre_wear': env.tyres.wear,
                'tyre_temp': env.tyres.temperature,
                'on_track': info.get('on_track', True),
            }

            # Renderizar
            renderer.render(car_state, track_data, telemetry)
            renderer.tick(args.fps)

            # Verificar fin de episodio
            if terminated or truncated:
                print(f"\nEpisode finished!")
                print(f"  Steps: {step}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Laps completed: {info.get('laps_completed', 0)}")

                # Pausa breve antes de siguiente episodio
                if episode < args.episodes - 1:
                    renderer.show_engineer_message("EPISODE COMPLETE - Starting next...", 120)
                    for _ in range(120):
                        renderer.render(car_state, track_data, telemetry)
                        renderer.tick(args.fps)
                        if not renderer.running:
                            break
                break

        if not renderer.running:
            break

    # Cleanup
    renderer.close()
    env.close()
    print("\n✓ Viewer closed")


if __name__ == "__main__":
    main()
