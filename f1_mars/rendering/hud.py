"""HUD estilo F1 TV con Arcade GUI."""

import arcade
from arcade.types import XYWH
import math
from typing import Optional


class RacingHUD:
    """
    HUD estilo broadcast F1.

    Usa Arcade para renderizado eficiente.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Colores F1
        self.bg_color = (15, 15, 20, 230)
        self.accent_color = (225, 6, 0)  # F1 Red
        self.text_color = arcade.color.WHITE
        self.secondary_color = (180, 180, 180)

        # Colores de compuestos
        self.compound_colors = {
            'SOFT': (255, 55, 55),
            'MEDIUM': (255, 210, 0),
            'HARD': (240, 240, 240),
        }

        # Mensaje del ingeniero
        self.engineer_message = ""
        self.engineer_timer = 0

    def resize(self, width: int, height: int):
        """Actualiza dimensiones."""
        self.width = width
        self.height = height

    def draw(self, state):
        """Dibuja el HUD completo."""
        self._draw_speed_panel(state.velocity)
        self._draw_lap_panel(state.lap, state.total_laps,
                           state.lap_time, state.best_lap_time)
        self._draw_tyre_panel(state.tyre_compound, state.tyre_wear, state.tyre_temp)
        self._draw_inputs(state.throttle, state.brake)
        self._draw_minimap(state)

        if self.engineer_timer > 0:
            self._draw_engineer_message()
            self.engineer_timer -= 1

        if not state.on_track:
            self._draw_track_warning()

    def _draw_rounded_rect(self, x: float, y: float,
                           width: float, height: float,
                           color: tuple):
        """Dibuja rectángulo redondeado (centrado en x, y)."""
        arcade.draw_rect_filled(
            XYWH(x - width/2, y - height/2, width, height),
            color
        )

    def _draw_speed_panel(self, velocity: float):
        """Panel de velocidad."""
        speed_kmh = int(velocity * 3.6)

        # Posición
        panel_w, panel_h = 180, 100
        x = self.width - panel_w // 2 - 20
        y = self.height - panel_h // 2 - 20

        # Fondo
        self._draw_rounded_rect(x, y, panel_w, panel_h, self.bg_color)

        # Línea de acento
        accent_y = y + panel_h // 2 - 2
        arcade.draw_rect_filled(
            XYWH(x - panel_w/2, accent_y - 2, panel_w, 4),
            self.accent_color
        )

        # Velocidad
        arcade.draw_text(f"{speed_kmh}", x, y + 5,
                        self.text_color, 54,
                        anchor_x="center", anchor_y="center",
                        font_name="Arial")

        # Unidad
        arcade.draw_text("KM/H", x, y - 30,
                        self.secondary_color, 14,
                        anchor_x="center", anchor_y="center")

        # Barra de velocidad
        bar_width = panel_w - 20
        bar_height = 6
        bar_x = x
        bar_y = y - panel_h // 2 + 15

        # Fondo de barra
        arcade.draw_rect_filled(
            XYWH(bar_x - bar_width/2, bar_y - bar_height/2, bar_width, bar_height),
            (40, 40, 45)
        )

        # Relleno
        fill_ratio = min(speed_kmh / 350, 1.0)
        fill_width = bar_width * fill_ratio

        if speed_kmh > 300:
            bar_color = (255, 50, 50)
        elif speed_kmh > 200:
            bar_color = (255, 200, 0)
        else:
            bar_color = (0, 200, 100)

        if fill_width > 0:
            arcade.draw_rect_filled(
                XYWH(bar_x - bar_width/2, bar_y - bar_height/2, fill_width, bar_height),
                bar_color
            )

    def _draw_lap_panel(self, lap: int, total: int,
                        lap_time: float, best_time: Optional[float]):
        """Panel de vuelta y tiempo."""
        panel_w, panel_h = 200, 80
        x = self.width // 2
        y = self.height - panel_h // 2 - 20

        # Fondo
        self._draw_rounded_rect(x, y, panel_w, panel_h, self.bg_color)

        # LAP X/Y
        arcade.draw_text(f"LAP {lap}/{total}", x, y + 15,
                        self.text_color, 22,
                        anchor_x="center", anchor_y="center")

        # Tiempo
        time_str = self._format_time(lap_time)

        # Color según delta
        time_color = self.text_color
        if best_time and lap_time > 0:
            if lap_time < best_time:
                time_color = arcade.color.GREEN
            elif lap_time > best_time * 1.05:
                time_color = (255, 50, 50)

        arcade.draw_text(time_str, x, y - 15,
                        time_color, 28,
                        anchor_x="center", anchor_y="center")

    def _draw_tyre_panel(self, compound: str, wear: float, temp: float):
        """Panel de neumáticos."""
        panel_w, panel_h = 180, 100
        x = panel_w // 2 + 20
        y = panel_h // 2 + 20

        # Fondo
        self._draw_rounded_rect(x, y, panel_w, panel_h, self.bg_color)

        # Color del compuesto
        comp_color = self.compound_colors.get(compound.upper(), (128, 128, 128))

        # Indicador circular
        arcade.draw_circle_filled(x - 60, y + 20, 12, comp_color)
        arcade.draw_circle_filled(x - 60, y + 20, 6, (40, 40, 45))

        # Nombre
        arcade.draw_text(compound.upper(), x - 35, y + 20,
                        comp_color, 20,
                        anchor_x="left", anchor_y="center")

        # Barra de vida
        bar_w = panel_w - 30
        bar_h = 12
        bar_x = x
        bar_y = y - 10

        remaining = max(0, 100 - wear)

        # Fondo
        arcade.draw_rect_filled(
            XYWH(bar_x - bar_w/2, bar_y - bar_h/2, bar_w, bar_h),
            (40, 40, 45)
        )

        # Relleno
        fill_w = bar_w * (remaining / 100)
        if remaining > 50:
            fill_color = (0, 200, 100)
        elif remaining > 30:
            fill_color = (255, 200, 0)
        else:
            fill_color = (255, 50, 50)

        if fill_w > 0:
            arcade.draw_rect_filled(
                XYWH(bar_x - bar_w/2, bar_y - bar_h/2, fill_w, bar_h),
                fill_color
            )

        # Porcentaje
        arcade.draw_text(f"{remaining:.0f}%", x + bar_w // 2 + 15, bar_y,
                        self.text_color, 14,
                        anchor_x="left", anchor_y="center")

        # Temperatura
        if temp > 105:
            temp_color = (255, 50, 50)
        elif temp < 80:
            temp_color = (255, 200, 0)
        else:
            temp_color = (0, 200, 100)

        arcade.draw_text(f"{temp:.0f}°C", x - 70, y - 35,
                        temp_color, 14,
                        anchor_x="left", anchor_y="center")

    def _draw_inputs(self, throttle: float, brake: float):
        """Visualización de throttle/brake."""
        x = 30
        y = 150
        bar_h = 60
        bar_w = 15

        # Throttle - fondo
        arcade.draw_rect_filled(
            XYWH(x - bar_w/2, y - bar_h/2, bar_w, bar_h),
            (40, 40, 45)
        )
        # Throttle - relleno
        fill_h = bar_h * throttle
        if fill_h > 0:
            arcade.draw_rect_filled(
                XYWH(x - bar_w/2, y - bar_h/2, bar_w, fill_h),
                (0, 200, 100)
            )

        # Brake - fondo
        arcade.draw_rect_filled(
            XYWH(x + 25 - bar_w/2, y - bar_h/2, bar_w, bar_h),
            (40, 40, 45)
        )
        # Brake - relleno
        fill_h = bar_h * brake
        if fill_h > 0:
            arcade.draw_rect_filled(
                XYWH(x + 25 - bar_w/2, y - bar_h/2, bar_w, fill_h),
                (255, 50, 50)
            )

    def _draw_minimap(self, state):
        """Minimapa del circuito."""
        size = 100
        x = self.width - size // 2 - 20
        y = size // 2 + 20

        # Fondo
        self._draw_rounded_rect(x, y, size, size, (15, 15, 20, 180))

        # Punto del coche (circular alrededor del centro)
        # Simplificado: usa progress para posicionar
        progress = getattr(state, 'progress', 0) or 0
        angle = progress * 2 * math.pi

        dot_x = x + 30 * math.cos(angle)
        dot_y = y + 30 * math.sin(angle)

        arcade.draw_circle_filled(dot_x, dot_y, 5, self.accent_color)
        arcade.draw_circle_filled(dot_x, dot_y, 3, arcade.color.WHITE)

    def _draw_engineer_message(self):
        """Mensaje del ingeniero."""
        panel_w = 400
        panel_h = 50
        x = self.width // 2
        y = 80

        # Fondo rojo F1
        self._draw_rounded_rect(x, y, panel_w, panel_h,
                               (*self.accent_color, 240))

        arcade.draw_text(self.engineer_message, x, y,
                        arcade.color.WHITE, 22,
                        anchor_x="center", anchor_y="center")

    def _draw_track_warning(self):
        """Warning de límites de pista."""
        alpha = int(150 + 50 * math.sin(arcade.get_time() * 10))

        arcade.draw_rect_filled(
            XYWH(self.width//2 - 150, self.height//2 + 75, 300, 50),
            (255, 200, 0, alpha)
        )
        arcade.draw_text("⚠ TRACK LIMITS", self.width // 2, self.height // 2 + 100,
                        arcade.color.BLACK, 22,
                        anchor_x="center", anchor_y="center")

    def _format_time(self, seconds: float) -> str:
        """Formatea tiempo como M:SS.ms"""
        if seconds <= 0:
            return "0:00.00"
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"

    def show_engineer_message(self, message: str, duration: int = 180):
        """Muestra mensaje del ingeniero."""
        self.engineer_message = message
        self.engineer_timer = duration
