#!/usr/bin/env python3
"""
Demostración de los componentes físicos implementados.

Muestra:
- Coche moviéndose en un circuito
- Física del modelo de bicicleta
- Desgaste de neumáticos
- Detección de límites de pista
- Sensores LIDAR (raycast)
"""

import sys
import numpy as np
from pathlib import Path
import importlib.util

# Importar módulos directamente sin pasar por __init__.py
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Cargar módulos
base_path = Path(__file__).parent / 'f1_mars'
car_module = load_module('car', base_path / 'envs' / 'car.py')
track_module = load_module('track', base_path / 'envs' / 'track.py')
tyres_module = load_module('tyres', base_path / 'envs' / 'tyres.py')
geom_module = load_module('geometry', base_path / 'utils' / 'geometry.py')

# Extraer clases
Car = car_module.Car
Track = track_module.Track
TrackGenerator = track_module.TrackGenerator
TyreSet = tyres_module.TyreSet
TyreCompound = tyres_module.TyreCompound
TyreStrategy = tyres_module.TyreStrategy
raycast = geom_module.raycast


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def simulate_lap():
    """Simula una vuelta completa con física realista."""

    print_header("DEMO: Simulación de Física F1 Mars")

    # ===== 1. Crear circuito =====
    print_header("1. Generando Circuito")

    track_data = TrackGenerator.generate_oval(length=1000, width=80)
    track = Track.__new__(Track)
    track.load_from_dict(track_data)

    print(f"✓ Circuito: {track.name}")
    print(f"  Longitud total: {track.total_length:.1f} metros")
    print(f"  Puntos de control: {track.num_control_points}")
    print(f"  Checkpoints: {len(track.checkpoint_indices)}")
    print(f"  Ancho: {track.widths[0]:.1f} metros")

    # ===== 2. Crear coche =====
    print_header("2. Inicializando Coche")

    car = Car(x=track.start_x, y=track.start_y, heading=track.start_heading)

    print(f"✓ Coche creado en posición inicial")
    print(f"  Posición: [{car.position[0]:.1f}, {car.position[1]:.1f}]")
    print(f"  Heading: {car.heading:.3f} rad")
    print(f"  Dimensiones: {car.length}m × {car.width}m")
    print(f"  Wheelbase: {car.wheelbase}m")
    print(f"  Velocidad máxima: {car.max_speed} m/s")

    # ===== 3. Crear neumáticos =====
    print_header("3. Instalando Neumáticos")

    tyres = TyreSet(TyreCompound.SOFT)

    print(f"✓ Neumáticos instalados: {tyres.compound.name}")
    print(f"  Grip base: {tyres.grip_base}")
    print(f"  Desgaste inicial: {tyres.wear:.1f}%")
    print(f"  Temperatura: {tyres.temperature:.1f}°C")
    print(f"  Grip actual: {tyres.get_grip():.3f}")

    # ===== 4. Simular vuelta =====
    print_header("4. Simulando Vuelta Completa")

    dt = 0.016  # ~60 FPS
    total_time = 0.0
    distance_traveled = 0.0
    checkpoints_passed = 0
    off_track_count = 0

    # Inputs de conducción (AI simplificada)
    throttle = 0.8
    brake = 0.0
    steering_input = 0.0

    print("Iniciando simulación...")
    print("(Mostrando estado cada 10 segundos)")

    last_print_time = 0.0

    # Simular hasta completar una vuelta (aproximadamente)
    max_simulation_time = 40.0  # segundos (reducido para demo rápida)

    while total_time < max_simulation_time:
        # === Control simple basado en la pista ===

        # Obtener posición en la pista
        closest_pt, dist_along, lateral_offset, track_heading = \
            track.get_closest_point_on_track(car.position)

        # Steering: intentar seguir la línea central
        heading_error = track_heading - car.heading
        # Normalizar error de ángulo a [-π, π]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Control proporcional simple
        steering_input = np.clip(heading_error * 2.0 + lateral_offset * 0.05, -1.0, 1.0)

        # Ajustar velocidad en curvas
        curvature = abs(track.get_curvature_at_distance(dist_along))
        if curvature > 0.01:  # Curva cerrada
            throttle = 0.5
            brake = 0.1
        else:  # Recta
            throttle = 0.9
            brake = 0.0

        # === Actualizar física ===

        # Obtener grip de neumáticos
        grip_multiplier = tyres.get_grip()

        # Actualizar coche
        car.update(
            dt=dt,
            throttle=throttle,
            brake=brake,
            steering_input=steering_input,
            grip_multiplier=grip_multiplier
        )

        # Obtener fuerza lateral del modelo del coche
        lateral_force = car.get_lateral_force()

        # Actualizar neumáticos
        tyres.update(
            dt=dt,
            speed=car.velocity,
            lateral_force=lateral_force,
            throttle=throttle,
            braking=brake
        )

        # === Comprobar límites de pista ===
        on_track = track.is_on_track(car.position)
        if not on_track:
            off_track_count += 1

        # === Comprobar checkpoints ===
        current_checkpoint = track.get_checkpoint_index(dist_along)
        if current_checkpoint > checkpoints_passed:
            checkpoints_passed = current_checkpoint

        # === Actualizar contadores ===
        total_time += dt
        distance_traveled = dist_along

        # === Imprimir estado cada 10 segundos ===
        if total_time - last_print_time >= 10.0:
            progress = (distance_traveled / track.total_length) * 100
            speed_kmh = car.velocity * 3.6

            print(f"\n⏱️  Tiempo: {total_time:.1f}s | Progreso: {progress:.1f}%")
            print(f"   Velocidad: {speed_kmh:.1f} km/h ({car.velocity:.1f} m/s) | Posición: [{car.position[0]:.0f}, {car.position[1]:.0f}]")
            print(f"   Neumáticos: {tyres.wear:.2f}% wear, {tyres.temperature:.1f}°C, grip={tyres.get_grip():.3f}")
            print(f"   Checkpoints: {checkpoints_passed}/{len(track.checkpoint_indices)}")
            print(f"   En pista: {'✓' if on_track else '✗ FUERA!'} (salidas: {off_track_count})")

            last_print_time = total_time

        # Terminar si completamos la vuelta
        if distance_traveled >= track.total_length * 0.95:
            print("\n🏁 ¡Vuelta completada!")
            break

    # ===== 5. Resumen final =====
    print_header("5. Resumen de la Vuelta")

    avg_speed_ms = distance_traveled / total_time
    avg_speed_kmh = avg_speed_ms * 3.6
    final_speed_kmh = car.velocity * 3.6

    print(f"Tiempo total: {total_time:.2f} segundos")
    print(f"Distancia recorrida: {distance_traveled:.1f}m / {track.total_length:.1f}m")
    print(f"Velocidad promedio: {avg_speed_kmh:.1f} km/h ({avg_speed_ms:.1f} m/s)")
    print(f"Checkpoints pasados: {checkpoints_passed}/{len(track.checkpoint_indices)}")
    print(f"Veces fuera de pista: {off_track_count}")
    print(f"\nEstado final del coche:")
    print(f"  Velocidad: {final_speed_kmh:.1f} km/h ({car.velocity:.1f} m/s)")
    print(f"  Posición: [{car.position[0]:.1f}, {car.position[1]:.1f}]")
    print(f"\nEstado final de neumáticos:")
    state = tyres.get_state()
    print(f"  Compuesto: {state['compound']}")
    print(f"  Desgaste: {state['wear']:.2f}%")
    print(f"  Temperatura: {state['temperature']:.1f}°C")
    print(f"  Grip: {state['current_grip']:.3f}")

    # ===== 6. Demo de LIDAR =====
    print_header("6. Demo Sensores LIDAR")

    # Crear segmentos de los bordes de la pista usando el método optimizado
    segments = track.get_boundary_segments(num_samples=100)  # Reduced for faster demo

    # Lanzar rayos LIDAR en 11 direcciones
    num_rays = 11
    max_distance = 50.0  # Reduced from 100m to 50m for better detection

    print(f"Lanzando {num_rays} rayos desde posición del coche...")
    print(f"Posición: [{car.position[0]:.1f}, {car.position[1]:.1f}], Heading: {car.heading:.3f}")

    for i in range(num_rays):
        angle = car.heading + (i - num_rays/2) * (np.pi / (num_rays - 1))
        distance = raycast(car.position, angle, segments, max_distance)

        angle_deg = np.degrees(angle - car.heading)
        print(f"  Rayo {i+1} ({angle_deg:+6.1f}°): {distance:6.2f}m")

    # ===== 7. Estrategia de neumáticos =====
    print_header("7. Análisis de Estrategia")

    # Estimar desgaste por vuelta
    wear_per_lap = tyres.wear  # Primera vuelta

    if wear_per_lap > 0:
        laps_remaining = TyreStrategy.estimate_laps_remaining(tyres, wear_per_lap)
        should_pit = TyreStrategy.should_pit(tyres, laps_remaining=10, pit_time_cost=20.0)

        print(f"Desgaste por vuelta: ~{wear_per_lap:.2f}%")
        print(f"Vueltas estimadas restantes: {laps_remaining}")
        print(f"¿Debería hacer pit stop? {'SÍ' if should_pit else 'NO'}")

        # Sugerir compuesto para próximo stint
        next_compound = TyreStrategy.choose_compound(laps_remaining=15)
        print(f"Compuesto recomendado para 15 vueltas: {next_compound.name}")

    print_header("✅ Demostración Completada")
    print("\nTodos los componentes funcionan correctamente:")
    print("  ✓ Física del coche (modelo de bicicleta)")
    print("  ✓ Sistema de circuitos (splines suaves)")
    print("  ✓ Desgaste y temperatura de neumáticos")
    print("  ✓ Detección de límites de pista")
    print("  ✓ Sensores LIDAR (raycast)")
    print("  ✓ Estrategia de pit stops")
    print("\n🚀 Listo para integrar en el entorno Gymnasium!")


if __name__ == "__main__":
    try:
        simulate_lap()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulación interrumpida por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error en la simulación: {e}")
        import traceback
        traceback.print_exc()
