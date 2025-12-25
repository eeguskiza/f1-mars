"""Sprites y elementos visuales mejorados."""

import pygame
import numpy as np
import math
from typing import Tuple, List
from .colors import *


class CarSprite:
    """Sprite de coche F1 con efectos visuales."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.base_width = int(50 * scale)
        self.base_height = int(22 * scale)

        # Crear sprite base
        self.sprite = self._create_f1_car()

        # Cache de rotaciones
        self._rotation_cache = {}

        # Efectos
        self.trail_positions: List[Tuple[float, float, float]] = []  # x, y, alpha
        self.max_trail = 15

    def _create_f1_car(self) -> pygame.Surface:
        """Crea un sprite de F1 más detallado."""
        w, h = self.base_width, self.base_height
        surface = pygame.Surface((w, h), pygame.SRCALPHA)

        # Colores del coche (estilo Ferrari rojo)
        main_color = (220, 20, 20)      # Rojo Ferrari
        accent_color = (255, 80, 80)    # Rojo brillante
        black = (20, 20, 25)
        white = (240, 240, 240)
        red = (255, 50, 50)

        # === CUERPO PRINCIPAL ===
        # Forma aerodinámica de F1
        body_points = [
            (8, h//2),           # Trasera centro
            (6, h//2 - 6),       # Trasera superior
            (15, h//2 - 7),      # Sidepod izq
            (30, h//2 - 5),      # Cockpit izq
            (w - 8, h//2 - 3),   # Morro izq
            (w - 3, h//2),       # Punta morro
            (w - 8, h//2 + 3),   # Morro der
            (30, h//2 + 5),      # Cockpit der
            (15, h//2 + 7),      # Sidepod der
            (6, h//2 + 6),       # Trasera inferior
        ]
        pygame.draw.polygon(surface, main_color, body_points)

        # Línea de acento central
        pygame.draw.line(surface, accent_color, (10, h//2), (w-10, h//2), 2)

        # === ALERÓN TRASERO ===
        wing_rect = pygame.Rect(3, h//2 - 9, 6, 18)
        pygame.draw.rect(surface, black, wing_rect)
        # Endplates
        pygame.draw.rect(surface, accent_color, (2, h//2 - 10, 2, 20))
        pygame.draw.rect(surface, accent_color, (8, h//2 - 10, 2, 20))

        # === ALERÓN DELANTERO ===
        # Elemento principal
        pygame.draw.rect(surface, black, (w-6, h//2 - 8, 4, 16))
        # Endplates delanteros
        pygame.draw.line(surface, accent_color, (w-4, h//2 - 9), (w-4, h//2 + 9), 2)

        # === COCKPIT (HALO) ===
        cockpit_rect = pygame.Rect(25, h//2 - 4, 12, 8)
        pygame.draw.ellipse(surface, black, cockpit_rect)
        # Halo
        pygame.draw.arc(surface, (60, 60, 60),
                       pygame.Rect(23, h//2 - 5, 14, 10),
                       math.radians(0), math.radians(180), 2)

        # === RUEDAS ===
        wheel_color = (30, 30, 35)
        tyre_color = (50, 50, 55)

        # Ruedas traseras (más grandes)
        # Izquierda
        pygame.draw.ellipse(surface, tyre_color, (5, 0, 10, 5))
        pygame.draw.ellipse(surface, wheel_color, (6, 1, 8, 3))
        # Derecha
        pygame.draw.ellipse(surface, tyre_color, (5, h-5, 10, 5))
        pygame.draw.ellipse(surface, wheel_color, (6, h-4, 8, 3))

        # Ruedas delanteras (más pequeñas)
        # Izquierda
        pygame.draw.ellipse(surface, tyre_color, (w-14, 1, 8, 4))
        pygame.draw.ellipse(surface, wheel_color, (w-13, 2, 6, 2))
        # Derecha
        pygame.draw.ellipse(surface, tyre_color, (w-14, h-5, 8, 4))
        pygame.draw.ellipse(surface, wheel_color, (w-13, h-4, 6, 2))

        # === DETALLES ===
        # Entrada de aire
        pygame.draw.rect(surface, black, (18, h//2 - 2, 4, 4))

        # Luces traseras
        pygame.draw.rect(surface, red, (4, h//2 - 3, 2, 6))

        # Número (círculo blanco)
        pygame.draw.circle(surface, white, (20, h//2), 4)

        return surface

    def get_rotated(self, angle_rad: float) -> pygame.Surface:
        """Obtiene sprite rotado (con cache)."""
        # Redondear ángulo para cache
        angle_deg = int(np.degrees(-angle_rad)) % 360

        if angle_deg not in self._rotation_cache:
            rotated = pygame.transform.rotate(self.sprite, angle_deg)
            self._rotation_cache[angle_deg] = rotated

        return self._rotation_cache[angle_deg]

    def update_trail(self, x: float, y: float, velocity: float):
        """Actualiza la estela del coche."""
        if velocity > 20:  # Solo si va rápido
            self.trail_positions.append((x, y, min(255, velocity * 2)))

        # Limitar longitud
        while len(self.trail_positions) > self.max_trail:
            self.trail_positions.pop(0)

        # Fade out
        self.trail_positions = [
            (x, y, max(0, a - 15))
            for x, y, a in self.trail_positions
            if a > 0
        ]

    def draw_trail(self, surface: pygame.Surface,
                   world_to_screen: callable,
                   color: Tuple[int, int, int] = (0, 200, 180)):
        """Dibuja la estela de velocidad."""
        if len(self.trail_positions) < 2:
            return

        for i, (x, y, alpha) in enumerate(self.trail_positions[:-1]):
            if alpha <= 0:
                continue

            sx, sy = world_to_screen(x, y)
            nx, ny = world_to_screen(*self.trail_positions[i + 1][:2])

            # Línea con alpha
            trail_color = (*color, int(alpha * 0.5))
            thickness = max(1, int(alpha / 50))

            # Crear superficie temporal para alpha
            trail_surf = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
            pygame.draw.line(trail_surf, trail_color, (sx, sy), (nx, ny), thickness)
            surface.blit(trail_surf, (0, 0))


class SpeedLines:
    """Líneas de velocidad en los laterales."""

    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height
        self.lines: List[dict] = []
        self.max_lines = 30

    def update(self, velocity: float):
        """Actualiza líneas según velocidad."""
        # Spawn rate basado en velocidad
        spawn_chance = min(0.8, velocity / 100)

        if np.random.random() < spawn_chance:
            # Crear nueva línea
            side = np.random.choice(['left', 'right'])
            y = np.random.randint(100, self.height - 100)

            self.lines.append({
                'x': 0 if side == 'left' else self.width,
                'y': y,
                'length': np.random.randint(20, 60 + int(velocity / 3)),
                'speed': 10 + velocity / 5,
                'alpha': 200,
                'side': side
            })

        # Actualizar existentes
        new_lines = []
        for line in self.lines:
            if line['side'] == 'left':
                line['x'] += line['speed']
                if line['x'] < self.width // 3:
                    line['alpha'] -= 10
                    if line['alpha'] > 0:
                        new_lines.append(line)
            else:
                line['x'] -= line['speed']
                if line['x'] > 2 * self.width // 3:
                    line['alpha'] -= 10
                    if line['alpha'] > 0:
                        new_lines.append(line)

        self.lines = new_lines[-self.max_lines:]

    def draw(self, surface: pygame.Surface):
        """Dibuja las líneas de velocidad."""
        for line in self.lines:
            color = (200, 200, 200, line['alpha'])

            if line['side'] == 'left':
                start = (line['x'], line['y'])
                end = (line['x'] - line['length'], line['y'])
            else:
                start = (line['x'], line['y'])
                end = (line['x'] + line['length'], line['y'])

            # Superficie con alpha
            line_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.line(line_surf, color, start, end, 2)
            surface.blit(line_surf, (0, 0))


class TrackRenderer:
    """Renderizado mejorado del circuito."""

    def __init__(self):
        self.kerb_pattern_size = 8

    def draw_track(self, surface: pygame.Surface,
                   centerline: List[Tuple[float, float]],
                   width: float,
                   world_to_screen: callable,
                   zoom: float):
        """Dibuja el circuito con kerbs y detalles."""
        if len(centerline) < 2:
            return

        screen_points = [world_to_screen(p[0], p[1]) for p in centerline]
        track_width = int(width * zoom)

        if track_width < 1:
            return

        # 1. Hierba de fondo (ya está en el clear)

        # 2. Borde exterior (grava/kerb área)
        self._draw_track_border(surface, screen_points, track_width + 20, (100, 90, 80))

        # 3. Kerbs
        self._draw_kerbs(surface, screen_points, track_width, zoom)

        # 4. Asfalto principal
        self._draw_asphalt(surface, screen_points, track_width)

        # 5. Línea central (racing line hint)
        if len(screen_points) >= 2:
            pygame.draw.lines(surface, (60, 60, 65), True, screen_points, 1)

        # 6. Línea de meta
        if screen_points:
            self._draw_start_finish(surface, screen_points[0],
                                   screen_points[1] if len(screen_points) > 1 else screen_points[0],
                                   track_width)

    def _draw_track_border(self, surface: pygame.Surface,
                           points: List[Tuple[int, int]],
                           width: int,
                           color: Tuple[int, int, int]):
        """Dibuja borde del circuito."""
        if len(points) >= 2 and width > 0:
            pygame.draw.lines(surface, color, True, points, width)

    def _draw_asphalt(self, surface: pygame.Surface,
                      points: List[Tuple[int, int]],
                      width: int):
        """Dibuja el asfalto con textura sutil."""
        if len(points) >= 2 and width > 0:
            # Color base
            pygame.draw.lines(surface, (45, 45, 50), True, points, width)
            # Línea de borde más clara
            pygame.draw.lines(surface, (70, 70, 75), True, points, width)
            # Centro más oscuro
            pygame.draw.lines(surface, (40, 40, 45), True, points, max(1, width - 4))

    def _draw_kerbs(self, surface: pygame.Surface,
                    points: List[Tuple[int, int]],
                    track_width: int,
                    zoom: float):
        """Dibuja kerbs rojo/blanco en los bordes."""
        if len(points) < 2:
            return

        kerb_width = max(4, int(6 * zoom))

        # Simplificado: dibujar bordes alternados
        # Interior
        pygame.draw.lines(surface, (255, 50, 50), True, points,
                         track_width + kerb_width)
        pygame.draw.lines(surface, (45, 45, 50), True, points, track_width)

    def _draw_start_finish(self, surface: pygame.Surface,
                           p1: Tuple[int, int],
                           p2: Tuple[int, int],
                           track_width: int):
        """Dibuja línea de meta con patrón de cuadros."""
        # Calcular perpendicular
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(1, np.sqrt(dx*dx + dy*dy))

        # Normal
        nx = -dy / length
        ny = dx / length

        # Dibujar línea de cuadros
        half_width = track_width // 2

        start = (int(p1[0] - nx * half_width), int(p1[1] - ny * half_width))
        end = (int(p1[0] + nx * half_width), int(p1[1] + ny * half_width))

        # Patrón de cuadros
        pygame.draw.line(surface, (255, 255, 255), start, end, 6)
        pygame.draw.line(surface, (20, 20, 20), start, end, 4)
        pygame.draw.line(surface, (255, 255, 255), start, end, 2)


def create_tyre_indicator(compound: str, wear: float, size: int = 40) -> pygame.Surface:
    """Crea indicador visual de neumático mejorado."""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)

    # Colores según compuesto
    colors = {
        'SOFT': (255, 55, 55),
        'MEDIUM': (255, 210, 0),
        'HARD': (240, 240, 240)
    }
    tyre_color = colors.get(compound.upper(), (128, 128, 128))

    center = size // 2
    radius = size // 2 - 4

    # Sombra
    pygame.draw.circle(surface, (0, 0, 0, 100), (center + 2, center + 2), radius)

    # Neumático exterior
    pygame.draw.circle(surface, tyre_color, (center, center), radius)

    # Degradado interior
    pygame.draw.circle(surface, tuple(max(0, c - 30) for c in tyre_color),
                      (center, center), radius - 2)

    # Llanta
    pygame.draw.circle(surface, (60, 60, 65), (center, center), radius // 2)
    pygame.draw.circle(surface, (80, 80, 85), (center, center), radius // 2 - 2)

    # Centro de la llanta
    pygame.draw.circle(surface, (40, 40, 45), (center, center), radius // 4)

    # Arco de desgaste
    if wear > 0:
        wear_angle = (wear / 100) * 2 * math.pi
        rect = pygame.Rect(4, 4, size - 8, size - 8)
        pygame.draw.arc(surface, (255, 0, 0, 200), rect,
                       math.pi / 2, math.pi / 2 - wear_angle, 4)

    return surface


# Mantener funciones legacy para compatibilidad
def create_car_sprite(width: int = 40, height: int = 20) -> pygame.Surface:
    """Legacy: Crea sprite simple del coche."""
    car = CarSprite(scale=1.0)
    return car.sprite


def draw_start_finish_line(surface: pygame.Surface,
                           position: Tuple[float, float],
                           direction: float,
                           width: float,
                           camera_offset: Tuple[float, float],
                           scale: float):
    """Legacy: Dibuja línea de meta."""
    x = (position[0] - camera_offset[0]) * scale + surface.get_width() // 2
    y = (position[1] - camera_offset[1]) * scale + surface.get_height() // 2

    # Calcular perpendicular a la dirección
    perp = direction + np.pi / 2
    half_width = (width * scale) / 2

    start = (x - np.cos(perp) * half_width, y - np.sin(perp) * half_width)
    end = (x + np.cos(perp) * half_width, y + np.sin(perp) * half_width)

    # Dibujar patrón de cuadros
    pygame.draw.line(surface, WHITE, start, end, 4)
