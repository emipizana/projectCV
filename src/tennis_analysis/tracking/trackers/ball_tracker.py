"""
Módulo para el tracking de la pelota.
"""

from collections import deque
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple, List

class BallTracker:
    """
    Tracker para la pelota de tenis con predicción de posición.
    """
    
    def __init__(
        self,
        buffer_size: int = 10,
        max_frames_to_predict: int = 5,
        min_points_for_prediction: int = 3
    ):
        """
        Args:
            buffer_size: Tamaño del buffer para almacenar posiciones anteriores
            max_frames_to_predict: Máximo número de frames para usar predicción
            min_points_for_prediction: Mínimo número de puntos necesarios para predicción
        """
        self.positions = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.last_predicted = None
        self.frames_without_detection = 0
        self.max_frames_to_predict = max_frames_to_predict
        self.min_points_for_prediction = min_points_for_prediction

    def add_detection(self, position: np.ndarray, timestamp: int):
        """
        Añade una nueva detección de la pelota.
        
        Args:
            position: Array [x, y] con la posición
            timestamp: Timestamp (número de frame) de la detección
        """
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.frames_without_detection = 0
        self.last_predicted = None

    def predict_position(self, current_timestamp: int) -> Optional[np.ndarray]:
        """
        Predice la posición de la pelota basada en posiciones anteriores.
        
        Args:
            current_timestamp: Timestamp actual (número de frame)
            
        Returns:
            Array [x, y] con la posición predicha o None si no se puede predecir
        """
        if len(self.positions) < self.min_points_for_prediction:
            return None

        try:
            positions = np.array(self.positions)
            timestamps = np.array(self.timestamps)

            # Interpolación cuadrática para x e y
            fx = interp1d(timestamps, positions[:, 0], kind='quadratic', 
                         fill_value='extrapolate')
            fy = interp1d(timestamps, positions[:, 1], kind='quadratic', 
                         fill_value='extrapolate')

            predicted_x = fx(current_timestamp)
            predicted_y = fy(current_timestamp)

            return np.array([predicted_x, predicted_y])

        except ValueError:
            return None

    def get_trajectory(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Obtiene los puntos de la trayectoria para visualización.
        
        Returns:
            Lista de tuplas (punto_inicio, punto_fin) para dibujar líneas
        """
        if len(self.positions) < 2:
            return []
            
        positions = np.array(self.positions)
        return [(positions[i], positions[i + 1]) 
                for i in range(len(positions) - 1)]

    def reset(self):
        """Reinicia el estado del tracker."""
        self.positions.clear()
        self.timestamps.clear()
        self.last_predicted = None
        self.frames_without_detection = 0