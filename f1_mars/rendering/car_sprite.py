"""Sprite del coche F1 con efectos visuales."""

import arcade
from arcade.types import XYWH
import math
from typing import Tuple


# Colores del coche (estilo Aston Martin / Mercedes)
CAR_MAIN = (0, 131, 131)        # Teal oscuro
CAR_ACCENT = (0, 210, 190)      # Teal brillante
CAR_BLACK = (20, 20, 25)
CAR_WHITE = (240, 240, 240)
CAR_RED = (255, 50, 50)


class F1CarSprite:
    """
    Sprite de F1 renderizado con primitivas de Arcade.

    Usa ShapeElementList para renderizado eficiente en GPU.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.pixels_per_meter = 8.0

        # Dimensiones del coche (en píxeles)
        self.length = 50 * scale
        self.width = 22 * scale

        # Cache de formas para diferentes ángulos
        self._shape_cache = {}

        # Trail de velocidad
        self.trail_points = []
        self.max_trail_points = 20

    def draw(self, x: float, y: float, heading: float, velocity: float):
        """
        Dibuja el coche F1.

        Args:
            x, y: Posición en metros
            heading: Ángulo en radianes
            velocity: Velocidad en m/s
        """
        # Convertir a píxeles
        px = x * self.pixels_per_meter
        py = y * self.pixels_per_meter
        angle_deg = math.degrees(heading)

        # Actualizar trail
        self._update_trail(px, py, velocity)

        # Dibujar trail
        self._draw_trail()

        # Dibujar el coche usando primitivas
        self._draw_car_body(px, py, angle_deg)

    def _update_trail(self, px: float, py: float, velocity: float):
        """Actualiza los puntos del trail."""
        if velocity > 15:  # Solo si va a más de ~54 km/h
            alpha = min(255, int(velocity * 2))
            self.trail_points.append((px, py, alpha))

        # Limitar longitud y fade out
        new_trail = []
        for tx, ty, alpha in self.trail_points:
            new_alpha = alpha - 12
            if new_alpha > 0:
                new_trail.append((tx, ty, new_alpha))

        self.trail_points = new_trail[-self.max_trail_points:]

    def _draw_trail(self):
        """Dibuja la estela de velocidad."""
        if len(self.trail_points) < 2:
            return

        for i in range(len(self.trail_points) - 1):
            x1, y1, a1 = self.trail_points[i]
            x2, y2, a2 = self.trail_points[i + 1]

            # Color con alpha
            alpha = (a1 + a2) // 2
            color = (*CAR_ACCENT, alpha)

            # Grosor basado en alpha
            width = max(1, alpha // 40)

            arcade.draw_line(x1, y1, x2, y2, color, width)

    def _draw_car_body(self, px: float, py: float, angle_deg: float):
        """Dibuja el cuerpo del coche."""
        # Usar transformación para rotar todo el coche

        # === CUERPO PRINCIPAL ===
        # Forma aerodinámica simplificada
        half_length = self.length / 2
        half_width = self.width / 2

        # Puntos del cuerpo (relativo al centro)
        body_points = [
            (-half_length + 5, 0),           # Cola centro
            (-half_length + 3, -half_width + 3),  # Cola izq
            (-half_length + 15, -half_width + 2), # Sidepod izq
            (half_length - 10, -half_width + 5),  # Morro izq
            (half_length - 3, 0),             # Punta
            (half_length - 10, half_width - 5),   # Morro der
            (-half_length + 15, half_width - 2),  # Sidepod der
            (-half_length + 3, half_width - 3),   # Cola der
        ]

        # Rotar y trasladar puntos
        rotated_body = self._rotate_points(body_points, angle_deg, px, py)

        # Dibujar cuerpo
        arcade.draw_polygon_filled(rotated_body, CAR_MAIN)
        arcade.draw_polygon_outline(rotated_body, CAR_ACCENT, 2)

        # === ALERÓN TRASERO ===
        wing_points = [
            (-half_length + 2, -half_width - 2),
            (-half_length + 8, -half_width - 2),
            (-half_length + 8, half_width + 2),
            (-half_length + 2, half_width + 2),
        ]
        rotated_wing = self._rotate_points(wing_points, angle_deg, px, py)
        arcade.draw_polygon_filled(rotated_wing, CAR_BLACK)

        # === ALERÓN DELANTERO ===
        front_wing = [
            (half_length - 5, -half_width - 1),
            (half_length - 2, -half_width - 1),
            (half_length - 2, half_width + 1),
            (half_length - 5, half_width + 1),
        ]
        rotated_front = self._rotate_points(front_wing, angle_deg, px, py)
        arcade.draw_polygon_filled(rotated_front, CAR_BLACK)

        # === RUEDAS ===
        wheel_positions = [
            (-half_length + 8, -half_width - 2),   # Trasera izq
            (-half_length + 8, half_width + 2),    # Trasera der
            (half_length - 12, -half_width - 1),   # Delantera izq
            (half_length - 12, half_width + 1),    # Delantera der
        ]

        for wx, wy in wheel_positions:
            rotated = self._rotate_point(wx, wy, angle_deg, px, py)
            arcade.draw_ellipse_filled(rotated[0], rotated[1],
                                      8 * self.scale, 4 * self.scale,
                                      CAR_BLACK, angle_deg)

        # === COCKPIT ===
        cockpit_center = self._rotate_point(-5, 0, angle_deg, px, py)
        arcade.draw_ellipse_filled(cockpit_center[0], cockpit_center[1],
                                  10 * self.scale, 6 * self.scale,
                                  CAR_BLACK, angle_deg)

        # === HALO ===
        halo_points = [
            (-8, -3),
            (2, -2),
            (2, 2),
            (-8, 3),
        ]
        rotated_halo = self._rotate_points(halo_points, angle_deg, px, py)
        arcade.draw_polygon_outline(rotated_halo, (80, 80, 85), 2)

        # === LUCES TRASERAS ===
        light_pos = self._rotate_point(-half_length + 4, 0, angle_deg, px, py)
        light_width = 3 * self.scale
        light_height = 6 * self.scale
        arcade.draw_rect_filled(
            XYWH(light_pos[0] - light_width/2, light_pos[1] - light_height/2,
                 light_width, light_height),
            CAR_RED, angle_deg
        )

        # === LÍNEA DE ACENTO ===
        line_start = self._rotate_point(-half_length + 10, 0, angle_deg, px, py)
        line_end = self._rotate_point(half_length - 10, 0, angle_deg, px, py)
        arcade.draw_line(line_start[0], line_start[1],
                        line_end[0], line_end[1],
                        CAR_ACCENT, 2)

    def _rotate_point(self, x: float, y: float,
                      angle_deg: float,
                      cx: float, cy: float) -> Tuple[float, float]:
        """Rota un punto alrededor de un centro."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy

        return rx, ry

    def _rotate_points(self, points: list, angle_deg: float,
                       cx: float, cy: float) -> list:
        """Rota una lista de puntos."""
        return [self._rotate_point(x, y, angle_deg, cx, cy)
                for x, y in points]
