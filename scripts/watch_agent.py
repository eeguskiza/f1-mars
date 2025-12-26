#!/usr/bin/env python3
"""
Visualizar agente entrenado con Arcade (GPU rendering).

Uso:
    python scripts/watch_agent.py                    # Modo interactivo
    python scripts/watch_agent.py --model MODEL.zip  # Modo directo
"""

import argparse
import sys
from pathlib import Path
import os

import arcade
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, SAC, TD3
from f1_mars.envs import F1Env
from f1_mars.rendering import F1MarsWindow


def detect_algorithm(model_path: str) -> str:
    """Detecta el algoritmo del modelo."""
    name = Path(model_path).stem.upper()
    if 'PPO' in name:
        return 'PPO'
    elif 'SAC' in name:
        return 'SAC'
    elif 'TD3' in name:
        return 'TD3'
    return 'PPO'


def find_models(models_dir: str = "trained_models") -> list:
    """Busca todos los modelos disponibles."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    models = list(models_path.glob("*.zip"))
    return sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)


def find_tracks(tracks_dir: str = "tracks") -> list:
    """Busca todos los circuitos disponibles."""
    tracks_path = Path(tracks_dir)
    if not tracks_path.exists():
        return []

    tracks = list(tracks_path.glob("*.json"))
    return sorted(tracks)


def select_model_interactive() -> Path:
    """Selección interactiva de modelo."""
    models = find_models()

    if not models:
        print("❌ No se encontraron modelos en 'trained_models/'")
        print("\nEntrena un modelo primero con:")
        print("  python scripts/train_agent.py")
        sys.exit(1)

    print("\n" + "="*60)
    print("  MODELOS DISPONIBLES")
    print("="*60)

    for i, model in enumerate(models, 1):
        size_mb = model.stat().st_size / (1024 * 1024)
        algo = detect_algorithm(str(model))
        print(f"  [{i}] {model.name:<40} ({size_mb:6.2f} MB, {algo})")

    print("="*60)

    while True:
        try:
            choice = input(f"\nSelecciona un modelo [1-{len(models)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print(f"❌ Número inválido. Escribe entre 1 y {len(models)}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Cancelado")
            sys.exit(0)


def select_track_interactive() -> Path | None:
    """Selección interactiva de circuito."""
    tracks = find_tracks()

    print("\n" + "="*60)
    print("  CIRCUITOS DISPONIBLES")
    print("="*60)

    print("  [0] Default (Oval generado)")

    for i, track in enumerate(tracks, 1):
        size_kb = track.stat().st_size / 1024
        print(f"  [{i}] {track.name:<40} ({size_kb:6.2f} KB)")

    print("="*60)

    while True:
        try:
            choice = input(f"\nSelecciona un circuito [0-{len(tracks)}]: ").strip()
            idx = int(choice)
            if idx == 0:
                return None  # Default
            if 1 <= idx <= len(tracks):
                return tracks[idx - 1]
            print(f"❌ Número inválido. Escribe entre 0 y {len(tracks)}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Cancelado")
            sys.exit(0)


def select_laps_interactive() -> int:
    """Selección interactiva de vueltas."""
    print("\n" + "="*60)
    print("  NÚMERO DE VUELTAS")
    print("="*60)

    while True:
        try:
            laps = input("Número de vueltas [1-10, default=3]: ").strip()
            if not laps:
                return 3
            laps_int = int(laps)
            if 1 <= laps_int <= 10:
                return laps_int
            print("❌ Escribe un número entre 1 y 10")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Cancelado")
            sys.exit(0)


class AgentViewer(F1MarsWindow):
    """Ventana que ejecuta un agente entrenado."""

    def __init__(self, model, env, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.env = env
        self.obs = None
        self.info = {}

        # Control
        self.deterministic = True
        self.episode_reward = 0
        self.episode_steps = 0

    def setup(self):
        """Configura el visor."""
        # Reset del entorno
        self.obs, self.info = self.env.reset()

        # Obtener datos del circuito
        if hasattr(self.env.track, 'centerline'):
            track_centerline = self.env.track.centerline.tolist()
        elif hasattr(self.env.track, 'control_points'):
            track_centerline = self.env.track.control_points.tolist() if hasattr(self.env.track.control_points, 'tolist') else self.env.track.control_points
        else:
            raise AttributeError("Track has no centerline or control_points")

        if hasattr(self.env.track, 'width'):
            track_width = self.env.track.width
        elif hasattr(self.env.track, 'widths'):
            track_width = float(np.mean(self.env.track.widths))
        else:
            track_width = 12.0  # Default

        # Setup base
        super().setup(track_centerline, track_width)

        # Actualizar estado inicial
        self._update_from_env()

    def on_update(self, delta_time: float):
        """Actualización: ejecuta step del agente."""
        if self.paused:
            return

        # Obtener acción del modelo (CPU)
        action, _ = self.model.predict(self.obs, deterministic=self.deterministic)

        # Ejecutar step en el entorno
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)

        self.episode_reward += reward
        self.episode_steps += 1

        # Actualizar estado de renderizado
        self._update_from_env()

        # Check fin de episodio
        if terminated or truncated:
            print(f"\n{'='*60}")
            print(f"  EPISODE FINISHED")
            print(f"{'='*60}")
            print(f"  Steps:        {self.episode_steps}")
            print(f"  Total reward: {self.episode_reward:.2f}")
            print(f"  Laps:         {self.info.get('laps_completed', 0)}")
            if self.info.get('best_lap_time'):
                print(f"  Best lap:     {self.info['best_lap_time']:.2f}s")
            print(f"{'='*60}\n")

            # Reset
            self.obs, self.info = self.env.reset()
            self.episode_reward = 0
            self.episode_steps = 0
            self._update_from_env()

        # Update base (cámara, efectos)
        super().on_update(delta_time)

    def _update_from_env(self):
        """Actualiza el estado de renderizado desde el entorno."""
        # Car position puede ser array o atributos directos
        car_x = self.env.car.position[0] if hasattr(self.env.car.position, '__getitem__') else self.env.car.position
        car_y = self.env.car.position[1] if hasattr(self.env.car.position, '__getitem__') else 0

        state = {
            'x': car_x,
            'y': car_y,
            'heading': self.env.car.heading,
            'velocity': self.env.car.velocity,
            'lap': self.info.get('lap', 1),
            'total_laps': self.env.max_laps,
            'lap_time': self.info.get('lap_time', 0),
            'best_lap_time': self.info.get('best_lap_time'),
            'tyre_compound': self.env.tyres.compound.name,
            'tyre_wear': self.env.tyres.wear,
            'tyre_temp': self.env.tyres.temperature,
            'on_track': self.info.get('on_track', True),
            'throttle': float(self.env.car.throttle) if hasattr(self.env.car, 'throttle') else 0,
            'brake': float(self.env.car.brake) if hasattr(self.env.car, 'brake') else 0,
        }
        self.update_state(state)

    def on_key_press(self, key, modifiers):
        """Maneja teclas."""
        super().on_key_press(key, modifiers)

        if key == arcade.key.R:
            # Reset episodio
            self.obs, self.info = self.env.reset()
            self.episode_reward = 0
            self.episode_steps = 0
            self._update_from_env()
            print("\n🔄 Episode reset!")


def main():
    parser = argparse.ArgumentParser(
        description='Visualizar agente con Arcade (modo interactivo)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/watch_agent.py                           # Modo interactivo
  python scripts/watch_agent.py --model MODEL.zip         # Modelo específico
  python scripts/watch_agent.py --model M.zip --track T   # Con circuito
        """
    )
    parser.add_argument('--model', type=str,
                       help='Ruta al modelo (.zip). Si no se especifica, modo interactivo.')
    parser.add_argument('--track', type=str,
                       help='Ruta al circuito (.json). Default: oval generado')
    parser.add_argument('--laps', type=int,
                       help='Número de vueltas (1-10, default: 3)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Ancho de ventana (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Alto de ventana (default: 720)')

    args = parser.parse_args()

    # Modo interactivo o directo
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Error: Modelo no encontrado: {args.model}")
            sys.exit(1)
    else:
        # Modo interactivo
        print("\n🏎️  F1-MARS VIEWER - MODO INTERACTIVO")
        model_path = select_model_interactive()

    if args.track:
        track_path = Path(args.track)
        if not track_path.exists():
            print(f"❌ Error: Circuito no encontrado: {args.track}")
            sys.exit(1)
    elif not args.model:  # Solo interactivo si no se especificó modelo
        track_path = select_track_interactive()
    else:
        track_path = None

    if args.laps:
        laps = args.laps
    elif not args.model:  # Solo interactivo si no se especificó modelo
        laps = select_laps_interactive()
    else:
        laps = 3

    # Cargar modelo
    algo_name = detect_algorithm(str(model_path))
    algo_class = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3}[algo_name]

    print(f"\n{'='*60}")
    print(f"  F1-MARS VIEWER (Arcade GPU Rendering)")
    print(f"{'='*60}")
    print(f"Model:      {model_path.name}")
    print(f"Algorithm:  {algo_name}")
    print(f"Track:      {track_path.name if track_path else 'Default oval'}")
    print(f"Laps:       {laps}")
    print(f"{'='*60}\n")

    print("Loading model...")
    model = algo_class.load(str(model_path))
    print("✓ Model loaded")

    print("Creating environment...")
    if track_path:
        env = F1Env(track_path=str(track_path), max_laps=laps)
    else:
        env = F1Env(max_laps=laps)
    print("✓ Environment created")

    print("Starting viewer...")
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset episode")
    print("  H     - Toggle HUD")
    print("  D     - Toggle debug")
    print("  +/-   - Zoom")
    print("  ESC   - Quit")
    print()

    # Crear y ejecutar ventana
    window = AgentViewer(
        model=model,
        env=env,
        width=args.width,
        height=args.height,
        title=f"F1-MARS - {model_path.stem}"
    )
    window.setup()

    arcade.run()

    # Cleanup
    env.close()
    print("\n✓ Viewer closed")


if __name__ == "__main__":
    main()
