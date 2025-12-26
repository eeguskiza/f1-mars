"""Renderizado del circuito con Arcade."""

import arcade
from arcade.shape_list import ShapeElementList, create_polygon, create_line
import math
from typing import List, Tuple


class TrackRenderer:
    """
    Renderiza el circuito usando ShapeElementLists de Arcade.

    Las formas se pre-calculan y se renderizan en batch en GPU.
    """

    def __init__(self):
        self.centerline = []
        self.width = 12.0
        self.pixels_per_meter = 8.0

        # Shape lists (renderizado batch en GPU)
        self.track_shapes = None
        self.kerb_shapes = None
        self.detail_shapes = None

        # Colores
        self.asphalt_color = (45, 45, 50)
        self.kerb_red = (255, 50, 50)
        self.kerb_white = (240, 240, 240)
        self.border_color = (100, 90, 80)  # Grava
        self.line_color = (200, 200, 200)

    def setup(self, centerline: list, width: float = 12.0):
        """
        Configura el circuito.

        Args:
            centerline: Lista de puntos (x, y) en metros
            width: Ancho de la pista en metros
        """
        self.centerline = centerline
        self.width = width

        # Crear shape lists
        self._build_shapes()

    def _build_shapes(self):
        """Construye las formas del circuito (se hace una vez)."""
        if not self.centerline or len(self.centerline) < 2:
            return

        # Convertir a píxeles
        points_px = [(x * self.pixels_per_meter, y * self.pixels_per_meter)
                     for x, y in self.centerline]

        width_px = self.width * self.pixels_per_meter

        # === TRACK SHAPES ===
        self.track_shapes = ShapeElementList()

        # Crear polígono del asfalto
        # Calcular bordes izquierdo y derecho
        left_edge = []
        right_edge = []

        for i, (px, py) in enumerate(points_px):
            # Calcular normal
            if i == 0:
                dx = points_px[1][0] - px
                dy = points_px[1][1] - py
            elif i == len(points_px) - 1:
                dx = px - points_px[i-1][0]
                dy = py - points_px[i-1][1]
            else:
                dx = points_px[i+1][0] - points_px[i-1][0]
                dy = points_px[i+1][1] - points_px[i-1][1]

            # Normalizar
            length = max(0.001, math.sqrt(dx*dx + dy*dy))
            nx = -dy / length
            ny = dx / length

            # Puntos de los bordes
            half_width = width_px / 2
            left_edge.append((px + nx * half_width, py + ny * half_width))
            right_edge.append((px - nx * half_width, py - ny * half_width))

        # Crear polígono cerrado para el asfalto
        track_polygon = left_edge + list(reversed(right_edge))

        if len(track_polygon) >= 3:
            # Asfalto
            asphalt = create_polygon(track_polygon, self.asphalt_color)
            self.track_shapes.append(asphalt)

        # === KERB SHAPES ===
        self.kerb_shapes = ShapeElementList()

        # Crear kerbs en las curvas
        kerb_width = 4 * self.pixels_per_meter / self.pixels_per_meter  # ~4 metros
        kerb_width_px = kerb_width * self.pixels_per_meter

        # Kerb interior (simplificado - línea roja en el borde)
        for i in range(len(points_px) - 1):
            # Alternar colores cada ciertos segmentos
            color = self.kerb_red if (i // 3) % 2 == 0 else self.kerb_white

            line = create_line(
                left_edge[i][0], left_edge[i][1],
                left_edge[i+1][0], left_edge[i+1][1],
                color, 4
            )
            self.kerb_shapes.append(line)

            line = create_line(
                right_edge[i][0], right_edge[i][1],
                right_edge[i+1][0], right_edge[i+1][1],
                color, 4
            )
            self.kerb_shapes.append(line)

        # === DETAIL SHAPES ===
        self.detail_shapes = ShapeElementList()

        # Línea central
        for i in range(len(points_px) - 1):
            if i % 4 < 2:  # Línea discontinua
                line = create_line(
                    points_px[i][0], points_px[i][1],
                    points_px[i+1][0], points_px[i+1][1],
                    self.line_color, 1
                )
                self.detail_shapes.append(line)

        # Línea de meta
        self._add_start_finish_line(points_px, width_px)

    def _add_start_finish_line(self, points_px: list, width_px: float):
        """Añade la línea de meta."""
        if len(points_px) < 2:
            return

        # Posición de la línea de meta (primer punto)
        px, py = points_px[0]

        # Calcular dirección
        dx = points_px[1][0] - px
        dy = points_px[1][1] - py
        length = max(0.001, math.sqrt(dx*dx + dy*dy))

        # Normal (perpendicular)
        nx = -dy / length
        ny = dx / length

        half_width = width_px / 2

        # Dibujar patrón de cuadros (simplificado como línea)
        start = (px + nx * half_width, py + ny * half_width)
        end = (px - nx * half_width, py - ny * half_width)

        # Línea blanca
        line = create_line(start[0], start[1], end[0], end[1],
                          arcade.color.WHITE, 6)
        self.detail_shapes.append(line)

        # Línea negra interior
        line = create_line(start[0], start[1], end[0], end[1],
                          arcade.color.BLACK, 3)
        self.detail_shapes.append(line)

    def draw(self, camera):
        """
        Dibuja el circuito.

        Args:
            camera: Cámara activa para culling (futuro)
        """
        # Dibujar en orden: fondo -> asfalto -> kerbs -> detalles
        if self.track_shapes:
            self.track_shapes.draw()

        if self.kerb_shapes:
            self.kerb_shapes.draw()

        if self.detail_shapes:
            self.detail_shapes.draw()
