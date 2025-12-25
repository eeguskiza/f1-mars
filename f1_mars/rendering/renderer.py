"""PyGame renderer for F1 Mars simulator."""

import pygame
import numpy as np
from typing import Optional

from f1_mars.envs.track import Track
from f1_mars.envs.car import Car
from f1_mars.rendering.hud import HUD
from f1_mars.utils.config import *


class Renderer:
    """
    Handles PyGame-based rendering of the racing environment.

    Renders track, car, and HUD overlay.
    """

    def __init__(self, track: Track, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        """
        Initialize the renderer.

        Args:
            track: Track object to render
            width: Screen width in pixels
            height: Screen height in pixels
        """
        self.width = width
        self.height = height
        self.track = track

        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("F1 Mars - 2D Racing Simulator")
        self.clock = pygame.time.Clock()

        # Initialize HUD
        self.hud = HUD(width, height)

        # Camera settings
        self.camera_pos = np.array([0.0, 0.0])
        self.zoom = 1.0

        # Pre-render track surface for efficiency
        self.track_surface = self._render_track_surface()

    def _render_track_surface(self) -> pygame.Surface:
        """
        Pre-render the track to a surface for efficient drawing.

        Returns:
            Surface with track rendered
        """
        # Calculate track bounds
        all_points = self.track.inner_boundary + self.track.outer_boundary
        if not all_points:
            # Fallback if track not loaded properly
            surface = pygame.Surface((self.width, self.height))
            surface.fill(COLOR_GRASS)
            return surface

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        track_width = max_x - min_x + 200  # padding
        track_height = max_y - min_y + 200

        # Create surface
        surface = pygame.Surface((int(track_width), int(track_height)))
        surface.fill(COLOR_GRASS)

        # Offset for padding
        offset_x = -min_x + 100
        offset_y = -min_y + 100

        # Draw track
        if len(self.track.inner_boundary) > 2 and len(self.track.outer_boundary) > 2:
            # Convert to screen coordinates
            inner_points = [(p[0] + offset_x, p[1] + offset_y)
                           for p in self.track.inner_boundary]
            outer_points = [(p[0] + offset_x, p[1] + offset_y)
                           for p in self.track.outer_boundary]

            # Draw track surface (combine inner and outer boundaries)
            track_polygon = inner_points + outer_points[::-1]
            pygame.draw.polygon(surface, COLOR_TRACK, track_polygon)

            # Draw centerline (optional, for debugging)
            # centerline_points = [(p[0] + offset_x, p[1] + offset_y)
            #                     for p in self.track.centerline]
            # pygame.draw.lines(surface, (255, 255, 0), True, centerline_points, 2)

        self.track_offset = (offset_x, offset_y)

        return surface

    def render(self, car: Car, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the current frame.

        Args:
            car: Car object to render
            mode: Render mode ("human" or "rgb_array")

        Returns:
            RGB array if mode is "rgb_array", None otherwise
        """
        # Clear screen
        self.screen.fill(COLOR_GRASS)

        # Update camera to follow car
        self._update_camera(car.position)

        # Draw track
        self._draw_track()

        # Draw car
        self._draw_car(car)

        # Draw HUD
        self.hud.draw(self.screen, car)

        # Handle rendering mode
        if mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
            return None
        elif mode == "rgb_array":
            # Return RGB array for video recording
            return np.transpose(
                pygame.surfarray.array3d(self.screen),
                axes=(1, 0, 2)
            )

    def _update_camera(self, target_pos: np.ndarray):
        """
        Update camera position to follow target.

        Args:
            target_pos: Position to center camera on
        """
        # Smooth camera follow
        target_camera = target_pos - np.array([self.width / 2, self.height / 2])
        self.camera_pos += (target_camera - self.camera_pos) * 0.1

    def _draw_track(self):
        """Draw the pre-rendered track surface."""
        # Calculate position to blit track surface
        blit_x = self.track_offset[0] - self.camera_pos[0]
        blit_y = self.track_offset[1] - self.camera_pos[1]

        self.screen.blit(self.track_surface, (blit_x, blit_y))

        # Draw checkpoints
        for i, checkpoint in enumerate(self.track.checkpoints):
            screen_pos = self._world_to_screen(checkpoint)
            pygame.draw.circle(
                self.screen,
                COLOR_CHECKPOINT,
                screen_pos,
                5
            )

    def _draw_car(self, car: Car):
        """
        Draw the car.

        Args:
            car: Car object to render
        """
        # Get car corners in world space
        corners = car.get_corners()

        # Convert to screen space
        screen_corners = [self._world_to_screen(tuple(corner)) for corner in corners]

        # Draw car body
        pygame.draw.polygon(self.screen, COLOR_CAR_PLAYER, screen_corners)

        # Draw outline
        pygame.draw.polygon(self.screen, COLOR_WHITE, screen_corners, 2)

        # Draw direction indicator
        front_pos = car.get_front_position()
        screen_front = self._world_to_screen(tuple(front_pos))
        screen_center = self._world_to_screen(tuple(car.position))

        pygame.draw.line(
            self.screen,
            COLOR_WHITE,
            screen_center,
            screen_front,
            3
        )

    def _world_to_screen(self, world_pos: tuple) -> tuple:
        """
        Convert world coordinates to screen coordinates.

        Args:
            world_pos: (x, y) in world space

        Returns:
            (x, y) in screen space
        """
        screen_x = int((world_pos[0] - self.camera_pos[0]) * self.zoom)
        screen_y = int((world_pos[1] - self.camera_pos[1]) * self.zoom)
        return (screen_x, screen_y)

    def close(self):
        """Clean up PyGame resources."""
        pygame.quit()


class F1Renderer:
    """
    Renderizador principal con PyGame (Phase 3 enhanced version).

    Features:
    - Cámara que sigue al coche
    - Renderizado del circuito
    - HUD con telemetría
    - Zoom ajustable
    """

    def __init__(self,
                 width: int = 1280,
                 height: int = 720,
                 title: str = "F1-MARS Racing Simulator"):
        """
        Inicializa el renderizador.

        Args:
            width: Ancho de la ventana
            height: Alto de la ventana
            title: Título de la ventana
        """
        pygame.init()
        pygame.display.set_caption(title)

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Cámara
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 5.0  # Píxeles por metro (más cerca del coche)
        self.camera_smoothing = 0.1  # Suavizado de cámara

        # Sprites y efectos mejorados
        from .sprites import CarSprite, SpeedLines, TrackRenderer
        self.car_sprite = CarSprite(scale=0.4)  # Coche más pequeño para proporciones realistas (F1 real ~5m largo)
        self.speed_lines = SpeedLines(width, height)
        self.track_renderer = TrackRenderer()

        # HUD
        self.hud = HUD(width, height)

        # Estado
        self.running = True
        self.paused = False
        self.show_hud = True
        self.show_trajectory = False

        # Historial de posiciones (para dibujar trayectoria)
        self.trajectory: list = []
        self.max_trajectory_points = 500

    def handle_events(self) -> dict:
        """
        Procesa eventos de PyGame.

        Returns:
            Dict con acciones del usuario:
                - quit: bool
                - pause: bool
                - zoom_in: bool
                - zoom_out: bool
                - reset: bool
        """
        actions = {
            'quit': False,
            'pause': False,
            'zoom_in': False,
            'zoom_out': False,
            'reset': False,
            'toggle_hud': False,
            'toggle_trajectory': False
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions['quit'] = True
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    actions['quit'] = True
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    actions['pause'] = True
                    self.paused = not self.paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    actions['zoom_in'] = True
                    self.zoom = min(10.0, self.zoom * 1.2)
                elif event.key == pygame.K_MINUS:
                    actions['zoom_out'] = True
                    self.zoom = max(0.5, self.zoom / 1.2)
                elif event.key == pygame.K_r:
                    actions['reset'] = True
                elif event.key == pygame.K_h:
                    actions['toggle_hud'] = True
                    self.show_hud = not self.show_hud
                elif event.key == pygame.K_t:
                    actions['toggle_trajectory'] = True
                    self.show_trajectory = not self.show_trajectory

        return actions

    def update_camera(self, target_x: float, target_y: float, velocity: float = 0):
        """
        Cámara dinámica que responde a velocidad.

        Args:
            target_x: Posición X del objetivo
            target_y: Posición Y del objetivo
            velocity: Velocidad del coche (m/s)
        """
        # Suavizado base
        smooth = 0.1

        # Zoom FIJO - no cambia automáticamente, solo con +/-
        # (el zoom se mantiene en self.zoom, modificable solo por el usuario)

        # Actualizar posición
        self.camera_x += (target_x - self.camera_x) * smooth
        self.camera_y += (target_y - self.camera_y) * smooth

    def world_to_screen(self, x: float, y: float) -> tuple:
        """Convierte coordenadas del mundo a coordenadas de pantalla."""
        screen_x = int((x - self.camera_x) * self.zoom + self.width // 2)
        screen_y = int((y - self.camera_y) * self.zoom + self.height // 2)
        return screen_x, screen_y

    def render(self,
               car_state: dict,
               track_data: dict,
               telemetry: dict):
        """
        Renderiza un frame completo con efectos visuales.

        Args:
            car_state: Estado del coche
                - x, y: Posición
                - heading: Ángulo
                - velocity: Velocidad
            track_data: Datos del circuito
                - centerline: Lista de puntos
                - width: Ancho de pista
                - checkpoints: Índices de checkpoints
            telemetry: Datos para el HUD
        """
        from .colors import TRACK_GRASS

        # 1. Limpiar pantalla (hierba)
        self.screen.fill(TRACK_GRASS)

        # Obtener velocidad
        velocity = car_state.get('velocity', 0)

        # 2. Actualizar cámara dinámica
        self.update_camera(car_state['x'], car_state['y'], velocity)

        # 3. Dibujar circuito con kerbs y detalles
        self.track_renderer.draw_track(
            self.screen,
            track_data['centerline'],
            track_data['width'],
            self.world_to_screen,
            self.zoom
        )

        # 4. Speed lines (detrás del coche)
        if velocity > 50:
            self.speed_lines.update(velocity)
            self.speed_lines.draw(self.screen)

        # 5. Trail del coche
        self.car_sprite.update_trail(car_state['x'], car_state['y'], velocity)
        self.car_sprite.draw_trail(self.screen, self.world_to_screen)

        # 6. Dibujar trayectoria (si está activada)
        if self.show_trajectory:
            self._draw_trajectory()

        # 7. Dibujar coche mejorado
        screen_x, screen_y = self.world_to_screen(car_state['x'], car_state['y'])
        rotated_car = self.car_sprite.get_rotated(car_state['heading'])
        rect = rotated_car.get_rect(center=(screen_x, screen_y))
        self.screen.blit(rotated_car, rect)

        # 8. Dibujar HUD
        if self.show_hud:
            self.hud.draw(self.screen, telemetry)

        # 9. Indicador de pausa
        if self.paused:
            self._draw_pause_overlay()

        # 10. Actualizar display
        pygame.display.flip()

    def _draw_track(self, track_data: dict):
        """Dibuja el circuito."""
        from .colors import TRACK_ASPHALT, TRACK_BORDER, GRAY, WHITE
        from .sprites import draw_start_finish_line

        centerline = track_data.get('centerline', [])
        width = track_data.get('width', 12.0)

        if len(centerline) < 2:
            return

        # Convertir puntos a coordenadas de pantalla
        screen_points = [self.world_to_screen(p[0], p[1]) for p in centerline]

        # Dibujar asfalto (línea gruesa)
        track_width = int(width * self.zoom)
        if track_width > 0 and len(screen_points) >= 2:
            # Dibujar borde exterior
            pygame.draw.lines(self.screen, TRACK_BORDER, True, screen_points,
                            track_width + 4)
            # Dibujar asfalto
            pygame.draw.lines(self.screen, TRACK_ASPHALT, True, screen_points,
                            track_width)
            # Dibujar línea central (guía)
            pygame.draw.lines(self.screen, GRAY, True, screen_points, 1)

        # Línea de meta
        if centerline:
            start_pos = centerline[0]
            # Calcular dirección
            if len(centerline) > 1:
                dx = centerline[1][0] - centerline[0][0]
                dy = centerline[1][1] - centerline[0][1]
                direction = np.arctan2(dy, dx)
            else:
                direction = 0

            draw_start_finish_line(
                self.screen, start_pos, direction, width,
                (self.camera_x, self.camera_y), self.zoom
            )

    def _draw_car(self, car_state: dict):
        """Dibuja el coche."""
        x, y = car_state['x'], car_state['y']
        heading = car_state['heading']

        # Convertir a coordenadas de pantalla
        screen_x, screen_y = self.world_to_screen(x, y)

        # Rotar sprite
        # PyGame usa grados, heading está en radianes
        angle_degrees = -np.degrees(heading)  # Negativo porque PyGame Y está invertido
        rotated_car = pygame.transform.rotate(self.car_sprite, angle_degrees)

        # Centrar sprite
        rect = rotated_car.get_rect(center=(screen_x, screen_y))

        # Dibujar
        self.screen.blit(rotated_car, rect)

    def _draw_trajectory(self):
        """Dibuja la trayectoria recorrida."""
        from .colors import F1_RED

        if len(self.trajectory) < 2:
            return

        screen_points = [self.world_to_screen(p[0], p[1]) for p in self.trajectory]
        pygame.draw.lines(self.screen, (*F1_RED, 128), False, screen_points, 2)

    def _draw_pause_overlay(self):
        """Dibuja overlay de pausa."""
        from .colors import WHITE, GRAY

        # Semi-transparente
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        # Texto
        font = pygame.font.Font(None, 72)
        text = font.render("PAUSED", True, WHITE)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text, text_rect)

        # Instrucciones
        font_small = pygame.font.Font(None, 36)
        instructions = [
            "SPACE - Resume",
            "R - Reset",
            "H - Toggle HUD",
            "T - Toggle Trajectory",
            "+/- - Zoom",
            "ESC - Quit"
        ]
        for i, instruction in enumerate(instructions):
            inst_text = font_small.render(instruction, True, GRAY)
            inst_rect = inst_text.get_rect(center=(self.width // 2,
                                                   self.height // 2 + 60 + i * 30))
            self.screen.blit(inst_text, inst_rect)

    def tick(self, fps: int = 60) -> float:
        """
        Limita FPS y retorna delta time.

        Args:
            fps: Frames por segundo objetivo

        Returns:
            Delta time en segundos
        """
        return self.clock.tick(fps) / 1000.0

    def close(self):
        """Cierra PyGame."""
        pygame.quit()

    def show_engineer_message(self, message: str, duration: int = 180):
        """Muestra mensaje del ingeniero en el HUD."""
        self.hud.show_engineer_message(message, duration)
