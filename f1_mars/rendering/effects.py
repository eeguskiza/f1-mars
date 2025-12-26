"""Efectos visuales: partículas, estelas, etc."""

import arcade
import math
import random
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Particle:
    """Partícula individual."""
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: Tuple[int, int, int, int]


class EffectsManager:
    """
    Gestor de efectos visuales.

    Efectos implementados:
    - Partículas de humo/polvo
    - Chispas en frenadas fuertes
    - Speed lines en los laterales
    - Marcas de neumático (futuro)
    """

    def __init__(self):
        self.particles: List[Particle] = []
        self.speed_lines: List[dict] = []
        self.max_particles = 100
        self.max_speed_lines = 30

        self.pixels_per_meter = 8.0

        # Colores
        self.smoke_color = (150, 150, 150)
        self.spark_color = (255, 200, 50)
        self.dust_color = (139, 119, 101)

    def update(self, car_x: float, car_y: float, heading: float,
               velocity: float, throttle: float, brake: float,
               delta_time: float):
        """Actualiza todos los efectos."""
        self._update_particles(delta_time)
        self._update_speed_lines(velocity, delta_time)

        # Spawn nuevas partículas según condiciones
        self._spawn_effects(car_x, car_y, heading, velocity, throttle, brake)

    def _update_particles(self, delta_time: float):
        """Actualiza partículas existentes."""
        alive_particles = []

        for p in self.particles:
            p.x += p.vx * delta_time
            p.y += p.vy * delta_time
            p.life -= delta_time

            # Física simple: fricción y gravedad
            p.vx *= 0.98
            p.vy *= 0.98

            if p.life > 0:
                alive_particles.append(p)

        self.particles = alive_particles[-self.max_particles:]

    def _update_speed_lines(self, velocity: float, delta_time: float):
        """Actualiza líneas de velocidad."""
        # Spawn basado en velocidad
        if velocity > 50 and random.random() < velocity / 150:
            side = random.choice(['left', 'right'])
            self.speed_lines.append({
                'x': -200 if side == 'left' else 200,
                'y': random.randint(-300, 300),
                'length': random.randint(30, 80),
                'speed': velocity * 0.3,
                'alpha': 200,
                'side': side
            })

        # Actualizar existentes
        alive_lines = []
        for line in self.speed_lines:
            if line['side'] == 'left':
                line['x'] += line['speed'] * delta_time * 60
            else:
                line['x'] -= line['speed'] * delta_time * 60

            line['alpha'] -= 5

            if line['alpha'] > 0 and abs(line['x']) < 400:
                alive_lines.append(line)

        self.speed_lines = alive_lines[-self.max_speed_lines:]

    def _spawn_effects(self, car_x: float, car_y: float, heading: float,
                       velocity: float, throttle: float, brake: float):
        """Genera nuevos efectos."""
        px = car_x * self.pixels_per_meter
        py = car_y * self.pixels_per_meter

        # Humo en aceleración fuerte
        if throttle > 0.9 and velocity > 10:
            self._spawn_smoke(px, py, heading, velocity)

        # Chispas en frenada fuerte
        if brake > 0.8 and velocity > 30:
            self._spawn_sparks(px, py, heading, velocity)

    def _spawn_smoke(self, px: float, py: float, heading: float, velocity: float):
        """Genera partículas de humo."""
        if random.random() > 0.3:
            return

        # Posición trasera del coche
        back_x = px - math.cos(heading) * 25
        back_y = py - math.sin(heading) * 25

        # Velocidad opuesta al movimiento + dispersión
        spread = 0.5
        vx = -math.cos(heading) * velocity * 0.1 + random.uniform(-spread, spread)
        vy = -math.sin(heading) * velocity * 0.1 + random.uniform(-spread, spread)

        particle = Particle(
            x=back_x + random.uniform(-5, 5),
            y=back_y + random.uniform(-5, 5),
            vx=vx * 10,
            vy=vy * 10,
            life=0.5,
            max_life=0.5,
            size=random.uniform(3, 8),
            color=(*self.smoke_color, 150)
        )
        self.particles.append(particle)

    def _spawn_sparks(self, px: float, py: float, heading: float, velocity: float):
        """Genera chispas de frenada."""
        if random.random() > 0.5:
            return

        # Posición delantera (frenos)
        front_x = px + math.cos(heading) * 15
        front_y = py + math.sin(heading) * 15

        for _ in range(3):
            angle = heading + random.uniform(-0.5, 0.5)
            speed = random.uniform(50, 150)

            particle = Particle(
                x=front_x + random.uniform(-10, 10),
                y=front_y + random.uniform(-5, 5),
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=0.2,
                max_life=0.2,
                size=2,
                color=(*self.spark_color, 255)
            )
            self.particles.append(particle)

    def draw_behind(self, camera):
        """Dibuja efectos que van detrás del coche."""
        # Humo
        for p in self.particles:
            if p.color[:3] == self.smoke_color:
                alpha = int(255 * (p.life / p.max_life))
                color = (*p.color[:3], alpha)
                size = p.size * (1 + (1 - p.life / p.max_life))
                arcade.draw_circle_filled(p.x, p.y, size, color)

    def draw_front(self, camera):
        """Dibuja efectos que van delante del coche."""
        # Chispas
        for p in self.particles:
            if p.color[:3] == self.spark_color:
                alpha = int(255 * (p.life / p.max_life))
                color = (*p.color[:3], alpha)
                arcade.draw_circle_filled(p.x, p.y, p.size, color)

        # Speed lines (relativo a la cámara, se dibuja en GUI)
        # Nota: Las speed lines se dibujan mejor en coordenadas de GUI
