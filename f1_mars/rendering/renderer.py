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
