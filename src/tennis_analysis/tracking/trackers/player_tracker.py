"""
Módulo para el tracking de jugadores.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PlayerDetection:
    """Representa una detección de jugador."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    player_class: int  # 0: player1, 1: player2
    position: np.ndarray  # [x, y] centro del bbox

@dataclass
class TrackingFrame:
    """Representa el estado de tracking en un frame."""
    frame_number: int
    detections: List[PlayerDetection]
    timestamp: float

class PlayerTracker:
    """
    Tracker para los jugadores de tenis.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: Umbral mínimo de confianza para considerar detecciones
        """
        self.confidence_threshold = confidence_threshold
        self.history: List[TrackingFrame] = []
        self.classes = {0: "player1", 1: "player2"}

    def update(self, boxes_df: pd.DataFrame, frame_number: int, timestamp: float):
        """
        Actualiza el tracking con nuevas detecciones.
        
        Args:
            boxes_df: DataFrame con las detecciones [x, y, w, h, conf, class]
            frame_number: Número del frame actual
            timestamp: Timestamp del frame
        """
        # Filtrar por confianza y clase
        boxes_df = boxes_df[
            (boxes_df['conf'] >= self.confidence_threshold) &
            (boxes_df['class'] != 2)  # Excluir otras clases que no sean jugadores
        ]
        
        # Ordenar por confianza y eliminar duplicados de clase
        boxes_df = boxes_df.sort_values('conf', ascending=False).drop_duplicates('class')
        
        # Convertir a PlayerDetection
        detections = []
        for _, box in boxes_df.iterrows():
            x1 = int(box['x'] - box['w'] / 2)
            y1 = int(box['y'] - box['h'] / 2)
            x2 = int(box['x'] + box['w'] / 2)
            y2 = int(box['y'] + box['h'] / 2)
            
            detection = PlayerDetection(
                bbox=np.array([x1, y1, x2, y2]),
                confidence=float(box['conf']),
                player_class=int(box['class']),
                position=np.array([box['x'], box['y']])
            )
            detections.append(detection)
        
        # Guardar frame
        frame = TrackingFrame(
            frame_number=frame_number,
            detections=detections,
            timestamp=timestamp
        )
        self.history.append(frame)

    def get_player_positions(self, frame_number: int) -> Dict[str, Optional[np.ndarray]]:
        """
        Obtiene las posiciones de los jugadores en un frame específico.
        
        Args:
            frame_number: Número de frame
            
        Returns:
            Diccionario con posiciones de cada jugador
        """
        positions = {
            "player1": None,
            "player2": None
        }
        
        # Buscar el frame
        frame = next(
            (f for f in self.history if f.frame_number == frame_number),
            None
        )
        
        if frame:
            for detection in frame.detections:
                player_name = self.classes[detection.player_class]
                positions[player_name] = detection.position
                
        return positions

    def get_player_trajectories(self) -> Dict[str, List[np.ndarray]]:
        """
        Obtiene las trayectorias completas de los jugadores.
        
        Returns:
            Diccionario con lista de posiciones para cada jugador
        """
        trajectories = {
            "player1": [],
            "player2": []
        }
        
        for frame in self.history:
            positions = self.get_player_positions(frame.frame_number)
            for player, pos in positions.items():
                if pos is not None:
                    trajectories[player].append(pos)
                    
        return trajectories

    def reset(self):
        """Reinicia el estado del tracker."""
        self.history.clear()