"""HUD (Heads-Up Display) for F1 Mars simulator."""

import pygame
from typing import Optional

from f1_mars.envs.car import Car
from f1_mars.utils.config import *


class HUD:
    """
    Renders the heads-up display showing car telemetry and race information.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the HUD.

        Args:
            screen_width: Width of the screen
            screen_height: Height of the screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Initialize fonts
        pygame.font.init()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # HUD layout parameters
        self.margin = 20
        self.line_spacing = 30

    def draw(self, screen: pygame.Surface, car: Car, additional_info: Optional[dict] = None):
        """
        Draw the HUD overlay.

        Args:
            screen: PyGame surface to draw on
            car: Car object to display telemetry for
            additional_info: Optional dictionary with extra info (lap, position, etc.)
        """
        # Draw semi-transparent background panels
        self._draw_panel(screen, 10, 10, 300, 200)  # Top-left panel
        self._draw_panel(screen, self.screen_width - 310, 10, 300, 150)  # Top-right

        # Draw telemetry
        self._draw_speed(screen, car)
        self._draw_gear(screen, car)
        self._draw_fuel(screen, car)
        self._draw_tire_info(screen, car)

        # Draw race info if provided
        if additional_info:
            self._draw_race_info(screen, additional_info)

    def _draw_panel(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        width: int,
        height: int
    ):
        """
        Draw a semi-transparent background panel.

        Args:
            screen: Surface to draw on
            x, y: Top-left corner position
            width, height: Panel dimensions
        """
        panel = pygame.Surface((width, height))
        panel.set_alpha(200)
        panel.fill(COLOR_HUD_BG[:3])  # RGB only, alpha set separately
        screen.blit(panel, (x, y))

    def _draw_speed(self, screen: pygame.Surface, car: Car):
        """
        Draw speedometer.

        Args:
            screen: Surface to draw on
            car: Car object
        """
        speed_kmh = car.speed * 3.6  # Convert to km/h equivalent
        speed_text = self.font_large.render(
            f"{int(speed_kmh)}",
            True,
            COLOR_WHITE
        )
        screen.blit(speed_text, (self.margin + 20, self.margin + 20))

        # Speed label
        label_text = self.font_small.render("km/h", True, COLOR_WHITE)
        screen.blit(label_text, (self.margin + 20, self.margin + 70))

    def _draw_gear(self, screen: pygame.Surface, car: Car):
        """
        Draw current gear (simulated based on speed).

        Args:
            screen: Surface to draw on
            car: Car object
        """
        # Simulate gear based on speed
        if car.speed < 20:
            gear = 1
        elif car.speed < 50:
            gear = 2
        elif car.speed < 80:
            gear = 3
        elif car.speed < 120:
            gear = 4
        elif car.speed < 180:
            gear = 5
        else:
            gear = 6

        gear_text = self.font_large.render(str(gear), True, COLOR_WHITE)
        screen.blit(gear_text, (self.margin + 150, self.margin + 20))

        # Gear label
        label_text = self.font_small.render("GEAR", True, COLOR_WHITE)
        screen.blit(label_text, (self.margin + 150, self.margin + 70))

    def _draw_fuel(self, screen: pygame.Surface, car: Car):
        """
        Draw fuel gauge.

        Args:
            screen: Surface to draw on
            car: Car object
        """
        y_offset = 120

        # Fuel label
        label_text = self.font_small.render("FUEL", True, COLOR_WHITE)
        screen.blit(label_text, (self.margin + 20, self.margin + y_offset))

        # Fuel bar
        bar_x = self.margin + 80
        bar_y = self.margin + y_offset + 5
        bar_width = 180
        bar_height = 20

        # Background
        pygame.draw.rect(
            screen,
            COLOR_BLACK,
            (bar_x, bar_y, bar_width, bar_height),
            2
        )

        # Fuel level
        fuel_width = int((car.fuel / 100.0) * bar_width)
        fuel_color = self._get_fuel_color(car.fuel)
        pygame.draw.rect(
            screen,
            fuel_color,
            (bar_x + 2, bar_y + 2, fuel_width - 4, bar_height - 4)
        )

        # Percentage text
        fuel_pct_text = self.font_small.render(
            f"{int(car.fuel)}%",
            True,
            COLOR_WHITE
        )
        screen.blit(fuel_pct_text, (bar_x + bar_width + 10, bar_y))

    def _draw_tire_info(self, screen: pygame.Surface, car: Car):
        """
        Draw tire wear indicators.

        Args:
            screen: Surface to draw on
            car: Car object
        """
        y_offset = 160

        # This requires access to TyreSystem - simplified for now
        # In actual implementation, pass TyreSystem separately or via car

        label_text = self.font_small.render("TIRES", True, COLOR_WHITE)
        screen.blit(label_text, (self.margin + 20, self.margin + y_offset))

        # Placeholder tire indicators
        # TODO: Access actual tire wear data
        tire_size = 15
        tire_spacing = 40
        base_x = self.margin + 100
        base_y = self.margin + y_offset

        # Draw 4 tire indicators (simplified)
        positions = [
            (base_x, base_y),  # FL
            (base_x + tire_spacing, base_y),  # FR
            (base_x, base_y + 30),  # RL
            (base_x + tire_spacing, base_y + 30),  # RR
        ]

        for pos in positions:
            pygame.draw.circle(screen, (100, 200, 100), pos, tire_size)
            pygame.draw.circle(screen, COLOR_WHITE, pos, tire_size, 2)

    def _draw_race_info(self, screen: pygame.Surface, info: dict):
        """
        Draw race information (lap, position, etc.).

        Args:
            screen: Surface to draw on
            info: Dictionary with race info
        """
        x = self.screen_width - 290
        y = self.margin + 20

        # Lap counter
        if "lap" in info:
            lap_text = self.font_medium.render(
                f"LAP: {info['lap']}",
                True,
                COLOR_WHITE
            )
            screen.blit(lap_text, (x, y))
            y += self.line_spacing

        # Position
        if "position" in info:
            pos_text = self.font_medium.render(
                f"P{info['position']}",
                True,
                COLOR_WHITE
            )
            screen.blit(pos_text, (x, y))
            y += self.line_spacing

        # Time
        if "time" in info:
            time_text = self.font_small.render(
                f"Time: {info['time']:.2f}s",
                True,
                COLOR_WHITE
            )
            screen.blit(time_text, (x, y))

    def _get_fuel_color(self, fuel_level: float) -> tuple:
        """
        Get color for fuel bar based on level.

        Args:
            fuel_level: Fuel percentage [0, 100]

        Returns:
            RGB color tuple
        """
        if fuel_level > 50:
            return (0, 255, 0)  # Green
        elif fuel_level > 25:
            return (255, 255, 0)  # Yellow
        else:
            return (255, 0, 0)  # Red
