"""Ventana principal de Arcade para F1-MARS."""

import arcade
from arcade.types import XYWH
import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

# Configuración de pantalla
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "F1-MARS Racing Simulator"


@dataclass
class GameState:
    """Estado del juego para renderizado."""
    car_x: float = 0.0
    car_y: float = 0.0
    car_heading: float = 0.0
    velocity: float = 0.0
    lap: int = 1
    total_laps: int = 3
    lap_time: float = 0.0
    best_lap_time: Optional[float] = None
    tyre_compound: str = "MEDIUM"
    tyre_wear: float = 0.0
    tyre_temp: float = 90.0
    on_track: bool = True
    throttle: float = 0.0
    brake: float = 0.0
    steering: float = 0.0


class F1MarsWindow(arcade.Window):
    """
    Ventana principal del simulador F1-MARS.

    Usa OpenGL para renderizado eficiente en GPU.
    """

    def __init__(self, width: int = SCREEN_WIDTH,
                 height: int = SCREEN_HEIGHT,
                 title: str = SCREEN_TITLE):
        super().__init__(width, height, title, resizable=True)

        arcade.set_background_color((34, 85, 34))  # Verde hierba

        # Estado
        self.game_state = GameState()
        self.paused = False
        self.show_hud = True
        self.show_debug = False

        # Componentes (se inicializan en setup)
        self.track_renderer = None
        self.car_sprite = None
        self.hud = None
        self.effects = None
        self.camera = None
        self.gui_camera = None

        # Track data
        self.track_centerline = []
        self.track_width = 12.0

        # Callback para obtener acciones (conecta con el agente)
        self.get_action_callback: Optional[Callable] = None

        # Métricas
        self.fps_history = []

    def setup(self, track_centerline: list, track_width: float = 12.0):
        """
        Configura el juego.

        Args:
            track_centerline: Lista de puntos (x, y) del circuito
            track_width: Ancho de la pista en metros
        """
        from .track_renderer import TrackRenderer
        from .car_sprite import F1CarSprite
        from .hud import RacingHUD
        from .effects import EffectsManager
        from .camera import RacingCamera

        self.track_centerline = track_centerline
        self.track_width = track_width

        # Cámara del mundo (sigue al coche)
        self.camera = RacingCamera(self.width, self.height)

        # Cámara del GUI (fija)
        self.gui_camera = arcade.Camera2D()

        # Renderizador del circuito
        self.track_renderer = TrackRenderer()
        self.track_renderer.setup(track_centerline, track_width)

        # Sprite del coche
        self.car_sprite = F1CarSprite()

        # HUD
        self.hud = RacingHUD(self.width, self.height)

        # Efectos visuales
        self.effects = EffectsManager()

    def update_state(self, state: Dict[str, Any]):
        """Actualiza el estado desde el entorno/agente."""
        self.game_state.car_x = state.get('x', 0)
        self.game_state.car_y = state.get('y', 0)
        self.game_state.car_heading = state.get('heading', 0)
        self.game_state.velocity = state.get('velocity', 0)
        self.game_state.lap = state.get('lap', 1)
        self.game_state.lap_time = state.get('lap_time', 0)
        self.game_state.best_lap_time = state.get('best_lap_time')
        self.game_state.tyre_compound = state.get('tyre_compound', 'MEDIUM')
        self.game_state.tyre_wear = state.get('tyre_wear', 0)
        self.game_state.tyre_temp = state.get('tyre_temp', 90)
        self.game_state.on_track = state.get('on_track', True)
        self.game_state.throttle = state.get('throttle', 0)
        self.game_state.brake = state.get('brake', 0)
        self.game_state.steering = state.get('steering', 0)

    def on_update(self, delta_time: float):
        """Actualización del juego (llamado cada frame)."""
        if self.paused:
            return

        # Actualizar cámara
        self.camera.update(
            self.game_state.car_x,
            self.game_state.car_y,
            self.game_state.velocity,
            delta_time,
            self.game_state.car_heading
        )

        # Actualizar efectos
        if self.effects:
            self.effects.update(
                self.game_state.car_x,
                self.game_state.car_y,
                self.game_state.car_heading,
                self.game_state.velocity,
                self.game_state.throttle,
                self.game_state.brake,
                delta_time
            )

        # FPS tracking
        if delta_time > 0:
            self.fps_history.append(1.0 / delta_time)
            if len(self.fps_history) > 60:
                self.fps_history.pop(0)

    def on_draw(self):
        """Renderiza el frame."""
        self.clear()

        # === MUNDO (con cámara que sigue al coche) ===
        self.camera.use()

        # Dibujar circuito
        if self.track_renderer:
            self.track_renderer.draw(self.camera)

        # Dibujar efectos detrás del coche
        if self.effects:
            self.effects.draw_behind(self.camera)

        # Dibujar coche
        if self.car_sprite:
            self.car_sprite.draw(
                self.game_state.car_x,
                self.game_state.car_y,
                self.game_state.car_heading,
                self.game_state.velocity
            )

        # Dibujar efectos delante del coche
        if self.effects:
            self.effects.draw_front(self.camera)

        # === GUI (cámara fija) ===
        self.gui_camera.use()

        if self.show_hud and self.hud:
            self.hud.draw(self.game_state)

        # Debug info
        if self.show_debug:
            self._draw_debug()

        # Pausa overlay
        if self.paused:
            self._draw_pause_overlay()

    def _draw_debug(self):
        """Dibuja información de debug."""
        avg_fps = sum(self.fps_history) / max(1, len(self.fps_history))
        debug_text = f"FPS: {avg_fps:.1f} | Pos: ({self.game_state.car_x:.1f}, {self.game_state.car_y:.1f})"
        arcade.draw_text(debug_text, 10, self.height - 30,
                        arcade.color.WHITE, 14)

    def _draw_pause_overlay(self):
        """Dibuja overlay de pausa."""
        # Fondo semi-transparente
        arcade.draw_rect_filled(
            XYWH(0, 0, self.width, self.height),
            (0, 0, 0, 150)
        )

        # Texto
        arcade.draw_text("PAUSED", self.width // 2, self.height // 2 + 50,
                        arcade.color.WHITE, 48, anchor_x="center")

        controls = [
            "SPACE - Resume",
            "R - Reset",
            "H - Toggle HUD",
            "D - Toggle Debug",
            "+/- - Zoom",
            "ESC - Quit"
        ]
        for i, text in enumerate(controls):
            arcade.draw_text(text, self.width // 2, self.height // 2 - 20 - i * 25,
                           arcade.color.GRAY, 18, anchor_x="center")

    def on_key_press(self, key, modifiers):
        """Maneja teclas presionadas."""
        if key == arcade.key.ESCAPE:
            arcade.close_window()
        elif key == arcade.key.SPACE:
            self.paused = not self.paused
        elif key == arcade.key.H:
            self.show_hud = not self.show_hud
        elif key == arcade.key.D:
            self.show_debug = not self.show_debug
        elif key == arcade.key.R:
            # Reset signal
            pass
        elif key == arcade.key.EQUAL or key == arcade.key.PLUS:
            self.camera.zoom_in()
        elif key == arcade.key.MINUS:
            self.camera.zoom_out()

    def on_resize(self, width, height):
        """Maneja redimensionado de ventana."""
        super().on_resize(width, height)
        self.camera.resize(width, height)
        if self.hud:
            self.hud.resize(width, height)
