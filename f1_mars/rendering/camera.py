"""Cámara dinámica para carreras."""

import arcade
import math


class RacingCamera:
    """
    Cámara que sigue al coche con efectos dinámicos.

    Features:
    - Suavizado de movimiento
    - Zoom dinámico según velocidad
    - Look-ahead (mira hacia donde va el coche)
    - Shake en colisiones (futuro)
    """

    def __init__(self, width: int, height: int):
        self.camera = arcade.Camera2D()
        self.width = width
        self.height = height

        # Posición actual
        self.x = 0.0
        self.y = 0.0

        # Zoom
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.min_zoom = 0.3
        self.max_zoom = 3.0
        self.base_zoom = 1.5  # Zoom base sin velocidad

        # Suavizado
        self.position_smoothing = 0.08
        self.zoom_smoothing = 0.05

        # Look-ahead
        self.look_ahead_factor = 0.5  # Cuánto mira hacia adelante

        # Escala metros -> píxeles
        self.pixels_per_meter = 8.0

    def update(self, target_x: float, target_y: float,
               velocity: float, delta_time: float,
               heading: float = 0.0):
        """
        Actualiza la cámara.

        Args:
            target_x, target_y: Posición del coche en metros
            velocity: Velocidad en m/s
            delta_time: Tiempo desde último frame
            heading: Dirección del coche en radianes
        """
        # Convertir a píxeles
        target_px = target_x * self.pixels_per_meter
        target_py = target_y * self.pixels_per_meter

        # Look-ahead: desplazar cámara en dirección del movimiento
        look_ahead_distance = velocity * self.look_ahead_factor * self.pixels_per_meter
        target_px += math.cos(heading) * look_ahead_distance
        target_py += math.sin(heading) * look_ahead_distance

        # Suavizar posición
        self.x += (target_px - self.x) * self.position_smoothing
        self.y += (target_py - self.y) * self.position_smoothing

        # Zoom dinámico: alejar cuando va rápido
        speed_factor = min(1.0, velocity / 100.0)  # Normalizar a 100 m/s
        self.target_zoom = self.base_zoom - (speed_factor * 0.7)
        self.target_zoom = max(self.min_zoom, min(self.max_zoom, self.target_zoom))

        # Suavizar zoom
        self.zoom += (self.target_zoom - self.zoom) * self.zoom_smoothing

        # Aplicar a la cámara de Arcade
        # En Arcade 3.x, la posición de la cámara es el centro del viewport
        self.camera.position = (self.x, self.y)
        self.camera.zoom = self.zoom

    def use(self):
        """Activa esta cámara para renderizado."""
        self.camera.use()

    def resize(self, width: int, height: int):
        """Actualiza dimensiones de la cámara."""
        self.width = width
        self.height = height

    def zoom_in(self):
        """Aumenta zoom manualmente."""
        self.base_zoom = min(self.max_zoom, self.base_zoom * 1.2)

    def zoom_out(self):
        """Reduce zoom manualmente."""
        self.base_zoom = max(self.min_zoom, self.base_zoom / 1.2)

    def world_to_screen(self, world_x: float, world_y: float) -> tuple:
        """Convierte coordenadas del mundo a pantalla."""
        px = world_x * self.pixels_per_meter
        py = world_y * self.pixels_per_meter

        screen_x = (px - self.x) * self.zoom + self.width / 2
        screen_y = (py - self.y) * self.zoom + self.height / 2

        return screen_x, screen_y

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple:
        """Convierte coordenadas de pantalla a mundo."""
        px = (screen_x - self.width / 2) / self.zoom + self.x
        py = (screen_y - self.height / 2) / self.zoom + self.y

        world_x = px / self.pixels_per_meter
        world_y = py / self.pixels_per_meter

        return world_x, world_y
