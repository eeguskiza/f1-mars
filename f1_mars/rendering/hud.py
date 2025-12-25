"""HUD estilo F1 TV Graphics."""

import pygame
import math
from typing import Dict, Any, Optional, Tuple
from .colors import *


class HUD:
    """
    HUD estilo broadcast F1 moderno.
    """

    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height

        pygame.font.init()

        # Intentar cargar fuente monospace para números
        try:
            self.font_speed = pygame.font.Font(None, 84)
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
        except:
            self.font_speed = pygame.font.SysFont('arial', 72)
            self.font_large = pygame.font.SysFont('arial', 42)
            self.font_medium = pygame.font.SysFont('arial', 28)
            self.font_small = pygame.font.SysFont('arial', 20)

        # Colores F1 TV
        self.bg_color = (15, 15, 20, 230)
        self.accent = (225, 6, 0)  # F1 Red
        self.text_primary = (255, 255, 255)
        self.text_secondary = (180, 180, 180)

        # Mensajes del ingeniero
        self.engineer_message = ""
        self.engineer_timer = 0

        # Delta tracking
        self.last_lap_time = None
        self.best_lap_time = None

    def draw(self, surface: pygame.Surface, telemetry: Dict[str, Any]):
        """Dibuja HUD completo."""
        # Panel de velocidad (derecha)
        self._draw_speed_panel(surface, telemetry.get('velocity', 0))

        # Panel de vuelta (centro-arriba)
        self._draw_lap_panel(surface,
                            telemetry.get('lap', 1),
                            telemetry.get('total_laps', 3),
                            telemetry.get('lap_time', 0),
                            telemetry.get('best_lap_time'))

        # Panel de neumáticos (izquierda-abajo)
        self._draw_tyre_panel(surface,
                             telemetry.get('tyre_compound', 'MEDIUM'),
                             telemetry.get('tyre_wear', 0),
                             telemetry.get('tyre_temp', 90))

        # Minimapa (esquina)
        self._draw_minimap(surface, telemetry)

        # Mensaje del ingeniero
        self._draw_engineer_message(surface)

        # Warning de límites de pista
        if not telemetry.get('on_track', True):
            self._draw_track_warning(surface)

    def _draw_rounded_rect(self, surface: pygame.Surface,
                           rect: pygame.Rect,
                           color: Tuple,
                           radius: int = 8):
        """Dibuja rectángulo redondeado con transparencia."""
        shape = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shape, color, (0, 0, rect.width, rect.height),
                        border_radius=radius)
        surface.blit(shape, rect.topleft)

    def _draw_speed_panel(self, surface: pygame.Surface, velocity: float):
        """Panel de velocidad estilo F1 TV."""
        speed_kmh = int(velocity * 3.6)

        # Panel
        panel_w, panel_h = 180, 100
        panel_x = self.width - panel_w - 20
        panel_y = 20

        self._draw_rounded_rect(surface,
                               pygame.Rect(panel_x, panel_y, panel_w, panel_h),
                               self.bg_color)

        # Línea de acento superior
        pygame.draw.rect(surface, self.accent,
                        (panel_x, panel_y, panel_w, 4),
                        border_radius=2)

        # Número de velocidad
        speed_text = self.font_speed.render(f"{speed_kmh}", True, self.text_primary)
        speed_rect = speed_text.get_rect(center=(panel_x + panel_w//2, panel_y + 50))
        surface.blit(speed_text, speed_rect)

        # Unidad
        unit_text = self.font_small.render("KM/H", True, self.text_secondary)
        unit_rect = unit_text.get_rect(center=(panel_x + panel_w//2, panel_y + 85))
        surface.blit(unit_text, unit_rect)

        # Barra de velocidad
        bar_y = panel_y + panel_h - 8
        bar_width = int((min(speed_kmh, 350) / 350) * (panel_w - 20))
        pygame.draw.rect(surface, (40, 40, 45),
                        (panel_x + 10, bar_y, panel_w - 20, 4),
                        border_radius=2)
        if bar_width > 0:
            # Color según velocidad
            if speed_kmh > 300:
                bar_color = (255, 50, 50)
            elif speed_kmh > 200:
                bar_color = (255, 200, 0)
            else:
                bar_color = (0, 200, 100)
            pygame.draw.rect(surface, bar_color,
                           (panel_x + 10, bar_y, bar_width, 4),
                           border_radius=2)

    def _draw_lap_panel(self, surface: pygame.Surface,
                        lap: int, total: int,
                        lap_time: float,
                        best_time: Optional[float]):
        """Panel de vuelta y tiempo."""
        panel_w, panel_h = 200, 80
        panel_x = self.width // 2 - panel_w // 2
        panel_y = 20

        self._draw_rounded_rect(surface,
                               pygame.Rect(panel_x, panel_y, panel_w, panel_h),
                               self.bg_color)

        # LAP X/Y
        lap_text = self.font_medium.render(f"LAP {lap}/{total}", True, self.text_primary)
        lap_rect = lap_text.get_rect(center=(panel_x + panel_w//2, panel_y + 25))
        surface.blit(lap_text, lap_rect)

        # Tiempo actual
        time_str = self._format_time(lap_time)

        # Color según delta con mejor tiempo
        time_color = self.text_primary
        if best_time and lap_time > 0:
            if lap_time < best_time:
                time_color = (0, 255, 0)  # Verde - mejorando
            elif lap_time > best_time * 1.05:
                time_color = (255, 50, 50)  # Rojo - perdiendo

        time_text = self.font_large.render(time_str, True, time_color)
        time_rect = time_text.get_rect(center=(panel_x + panel_w//2, panel_y + 55))
        surface.blit(time_text, time_rect)

    def _draw_tyre_panel(self, surface: pygame.Surface,
                         compound: str, wear: float, temp: float):
        """Panel de neumáticos."""
        panel_w, panel_h = 180, 100
        panel_x = 20
        panel_y = self.height - panel_h - 20

        self._draw_rounded_rect(surface,
                               pygame.Rect(panel_x, panel_y, panel_w, panel_h),
                               self.bg_color)

        # Color del compuesto
        compound_colors = {
            'SOFT': (255, 55, 55),
            'MEDIUM': (255, 210, 0),
            'HARD': (240, 240, 240)
        }
        comp_color = compound_colors.get(compound.upper(), (128, 128, 128))

        # Indicador de compuesto (círculo)
        pygame.draw.circle(surface, comp_color, (panel_x + 25, panel_y + 30), 12)
        pygame.draw.circle(surface, (40, 40, 45), (panel_x + 25, panel_y + 30), 6)

        # Nombre
        name_text = self.font_medium.render(compound.upper(), True, comp_color)
        surface.blit(name_text, (panel_x + 45, panel_y + 18))

        # Barra de vida
        bar_x = panel_x + 15
        bar_y = panel_y + 55
        bar_w = panel_w - 30
        bar_h = 12

        remaining = max(0, 100 - wear)
        fill_w = int((remaining / 100) * bar_w)

        # Fondo
        pygame.draw.rect(surface, (40, 40, 45),
                        (bar_x, bar_y, bar_w, bar_h),
                        border_radius=4)

        # Relleno
        if fill_w > 0:
            if remaining > 50:
                fill_color = (0, 200, 100)
            elif remaining > 30:
                fill_color = (255, 200, 0)
            else:
                fill_color = (255, 50, 50)
            pygame.draw.rect(surface, fill_color,
                           (bar_x, bar_y, fill_w, bar_h),
                           border_radius=4)

        # Porcentaje
        pct_text = self.font_small.render(f"{remaining:.0f}%", True, self.text_primary)
        surface.blit(pct_text, (bar_x + bar_w + 8, bar_y - 2))

        # Temperatura
        temp_color = (0, 200, 100) if 80 <= temp <= 105 else (255, 200, 0) if temp < 80 else (255, 50, 50)
        temp_text = self.font_small.render(f"{temp:.0f}°C", True, temp_color)
        surface.blit(temp_text, (panel_x + 15, panel_y + 75))

    def _draw_minimap(self, surface: pygame.Surface, telemetry: dict):
        """Minimapa del circuito."""
        # Simplificado - solo posición
        map_size = 100
        map_x = self.width - map_size - 20
        map_y = self.height - map_size - 20

        self._draw_rounded_rect(surface,
                               pygame.Rect(map_x, map_y, map_size, map_size),
                               (15, 15, 20, 180))

        # Punto del coche
        progress = telemetry.get('lap_time', 0) % 60 / 60  # Aproximación
        angle = progress * 2 * math.pi

        dot_x = map_x + map_size//2 + int(30 * math.cos(angle))
        dot_y = map_y + map_size//2 + int(30 * math.sin(angle))

        pygame.draw.circle(surface, self.accent, (dot_x, dot_y), 5)
        pygame.draw.circle(surface, (255, 255, 255), (dot_x, dot_y), 3)

    def _draw_engineer_message(self, surface: pygame.Surface):
        """Mensaje del ingeniero."""
        if self.engineer_timer <= 0:
            return

        self.engineer_timer -= 1

        # Panel central
        panel_w = 400
        panel_h = 60
        panel_x = self.width // 2 - panel_w // 2
        panel_y = self.height - 100

        # Fondo rojo F1
        self._draw_rounded_rect(surface,
                               pygame.Rect(panel_x, panel_y, panel_w, panel_h),
                               (*self.accent, 240))

        # Texto
        msg_text = self.font_medium.render(self.engineer_message, True, (255, 255, 255))
        msg_rect = msg_text.get_rect(center=(panel_x + panel_w//2, panel_y + panel_h//2))
        surface.blit(msg_text, msg_rect)

    def _draw_track_warning(self, surface: pygame.Surface):
        """Warning de límites de pista."""
        # Flash effect
        alpha = 150 + int(50 * math.sin(pygame.time.get_ticks() / 100))

        warning_surf = pygame.Surface((300, 50), pygame.SRCALPHA)
        pygame.draw.rect(warning_surf, (255, 200, 0, alpha), (0, 0, 300, 50),
                        border_radius=8)

        text = self.font_medium.render("⚠ TRACK LIMITS", True, (0, 0, 0))
        text_rect = text.get_rect(center=(150, 25))
        warning_surf.blit(text, text_rect)

        surface.blit(warning_surf, (self.width//2 - 150, self.height//2 - 100))

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

    def show_pit_signal(self, compound: str, laps_until_pit: int):
        """Muestra señal de pit stop (para Fase 5)."""
        if laps_until_pit <= 0:
            self.show_engineer_message(f"BOX BOX BOX! → {compound}", 300)
        elif laps_until_pit <= 2:
            self.show_engineer_message(f"PIT IN {laps_until_pit} LAPS → {compound}", 180)
