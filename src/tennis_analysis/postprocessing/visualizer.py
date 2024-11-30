"""
Módulo para visualización de tracking en video.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple, Any

from ..tracking.trackers import BallTracker, PlayerTracker

class TrackingVisualizer:
    """Visualizador para tracking de tenis."""
    
    def __init__(
        self,
        width: int,
        height: int,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ):
        """
        Args:
            width: Ancho del frame
            height: Alto del frame
            colors: Diccionario de colores para visualización
        """
        self.width = width
        self.height = height
        self.colors = colors or {
            'ball': (0, 0, 255),      # Rojo
            'ball_predicted': (0, 255, 255),  # Amarillo
            'trajectory': (0, 165, 255),      # Naranja
            'player1': (0, 255, 0),    # Verde
            'player2': (255, 0, 0),    # Azul
            'overlay_bg': (0, 0, 0),   # Negro
            'overlay_text': (255, 255, 255)  # Blanco
        }
        
        # Configuración de texto
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_scale_overlay = 0.4
        self.thickness = 2
        self.thickness_overlay = 1
        
    def draw_tracking(
        self,
        frame: np.ndarray,
        ball_tracker: BallTracker,
        player_tracker: PlayerTracker,
        frame_number: int,
        show_stats: bool = True
    ) -> np.ndarray:
        """
        Dibuja el tracking sobre el frame.
        
        Args:
            frame: Frame a procesar
            ball_tracker: Instancia de BallTracker
            player_tracker: Instancia de PlayerTracker
            frame_number: Número de frame actual
            show_stats: Si se debe mostrar estadísticas en overlay
            
        Returns:
            Frame con visualización
        """
        # Dibujar jugadores
        frame = self._draw_players(frame, player_tracker, frame_number)
        
        # Dibujar trayectoria de la pelota
        frame = self._draw_ball_trajectory(frame, ball_tracker)
        
        # Dibujar pelota (detectada o predicha)
        frame = self._draw_ball(frame, ball_tracker)
        
        # Añadir estadísticas si se solicita
        if show_stats:
            stats = {
                'frame': frame_number,
                'ball_detected': len(ball_tracker.positions) > 0,
                'ball_predicted': ball_tracker.last_predicted is not None,
                'players': len(player_tracker.get_player_positions(frame_number))
            }
            frame = self.add_overlay(frame, stats)
        
        return frame
    
    def _draw_players(
        self,
        frame: np.ndarray,
        player_tracker: PlayerTracker,
        frame_number: int
    ) -> np.ndarray:
        """Dibuja las detecciones de jugadores."""
        # Obtener posiciones actuales
        positions = player_tracker.get_player_positions(frame_number)
        
        for current_frame in player_tracker.history:
            if current_frame.frame_number == frame_number:
                for detection in current_frame.detections:
                    # Obtener color según jugador
                    color = self.colors[f'player{detection.player_class + 1}']
                    
                    # Dibujar bounding box
                    x1, y1, x2, y2 = detection.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Añadir etiqueta
                    label = f'Player {detection.player_class + 1}: {detection.confidence:.2f}'
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        self.font,
                        self.font_scale,
                        color,
                        self.thickness
                    )
        
        return frame
    
    def _draw_ball_trajectory(
        self,
        frame: np.ndarray,
        ball_tracker: BallTracker
    ) -> np.ndarray:
        """Dibuja la trayectoria de la pelota."""
        trajectory_points = ball_tracker.get_trajectory()
        
        for pt1, pt2 in trajectory_points:
            cv2.line(
                frame,
                tuple(map(int, pt1)),
                tuple(map(int, pt2)),
                self.colors['trajectory'],
                1
            )
        
        return frame
    
    def _draw_ball(
        self,
        frame: np.ndarray,
        ball_tracker: BallTracker
    ) -> np.ndarray:
        """Dibuja la pelota (detectada o predicha)."""
        # Dibujar última posición detectada
        if len(ball_tracker.positions) > 0:
            last_pos = ball_tracker.positions[-1]
            x, y = map(int, last_pos)
            cv2.circle(frame, (x, y), 5, self.colors['ball'], -1)
            cv2.putText(
                frame,
                'Ball',
                (x - 20, y - 10),
                self.font,
                self.font_scale,
                self.colors['ball'],
                self.thickness
            )
        
        # Dibujar posición predicha
        elif ball_tracker.last_predicted is not None:
            x, y = map(int, ball_tracker.last_predicted)
            cv2.circle(frame, (x, y), 5, self.colors['ball_predicted'], -1)
            cv2.putText(
                frame,
                'Predicted',
                (x - 30, y - 10),
                self.font,
                self.font_scale,
                self.colors['ball_predicted'],
                self.thickness
            )
        
        return frame
    
    def add_overlay(
        self,
        frame: np.ndarray,
        stats: Dict[str, Any],
        position: str = 'top-left'
    ) -> np.ndarray:
        """
        Añade un overlay con estadísticas al frame.
        
        Args:
            frame: Frame base
            stats: Diccionario con estadísticas
            position: Posición del overlay ('top-left', 'top-right', etc.)
            
        Returns:
            Frame con overlay
        """
        # Preparar textos
        texts = [
            f"Frame: {stats['frame']}",
            f"Ball: {'Detected' if stats['ball_detected'] else 'Predicted' if stats['ball_predicted'] else 'Not Found'}",
            f"Players: {stats['players']}"
        ]
        
        # Calcular dimensiones del overlay
        padding = 10
        line_height = 20
        overlay_width = 150
        overlay_height = (len(texts) + 1) * line_height
        
        # Determinar posición
        if position == 'top-left':
            x, y = padding, padding
        elif position == 'top-right':
            x, y = self.width - overlay_width - padding, padding
        else:
            x, y = padding, padding
        
        # Crear overlay semi-transparente
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + overlay_width, y + overlay_height),
            self.colors['overlay_bg'],
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Añadir textos
        for i, text in enumerate(texts):
            cv2.putText(
                frame,
                text,
                (x + 5, y + (i + 1) * line_height),
                self.font,
                self.font_scale_overlay,
                self.colors['overlay_text'],
                self.thickness_overlay
            )
        
        return frame