"""
Módulo para el tracking de la pelota con trayectoria suavizada y detección de outliers.
"""

from collections import deque
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple, List

class BallTracker:
    """
    Tracker para la pelota de tenis con predicción de posición, filtrado de outliers
    y trayectoria suavizada.
    """
    
    def __init__(
        self,
        buffer_size: int = 30,
        max_frames_to_predict: int = 5,
        min_points_for_prediction: int = 3,
        max_speed: float = 1000.0,  # Velocidad máxima permitida para filtrar outliers
        trajectory_points: int = 50
    ):
        """
        Args:
            buffer_size: Tamaño del buffer para almacenar posiciones anteriores
            max_frames_to_predict: Máximo número de frames para usar predicción
            min_points_for_prediction: Mínimo número de puntos necesarios para predicción
            max_speed: Velocidad máxima permitida (pixels/segundo) para filtrar outliers
            trajectory_points: Número de puntos a usar en la interpolación
        """
        self.positions = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.confidences = deque(maxlen=buffer_size)
        self.last_predicted = None
        self.frames_without_detection = 0
        self.max_frames_to_predict = max_frames_to_predict
        self.min_points_for_prediction = min_points_for_prediction
        self.max_speed = max_speed
        self.trajectory_points = trajectory_points
        
        # Para almacenar posiciones filtradas
        self.filtered_positions = []
        self.filtered_timestamps = []
        self.outlier_positions = []
        self.outlier_timestamps = []

    def add_detection(self, position: np.ndarray, timestamp: float, confidence: float = 1.0):
        """
        Añade una nueva detección de la pelota.
        
        Args:
            position: Array [x, y] con la posición
            timestamp: Timestamp de la detección
            confidence: Confianza de la detección (0-1)
        """
        position = np.asarray(position).flatten()
        timestamp = float(timestamp)
        
        # Filtrar outliers basado en velocidad
        if len(self.filtered_positions) > 0:
            dt = timestamp - self.filtered_timestamps[-1]
            if dt > 0:  # Evitar división por cero
                dx = position[0] - self.filtered_positions[-1][0]
                dy = position[1] - self.filtered_positions[-1][1]
                speed = np.sqrt((dx/dt)**2 + (dy/dt)**2)
                
                if speed <= self.max_speed:
                    self.filtered_positions.append(position)
                    self.filtered_timestamps.append(timestamp)
                else:
                    self.outlier_positions.append(position)
                    self.outlier_timestamps.append(timestamp)
                    # No agregamos al buffer principal si es outlier
                    return
        else:
            # Siempre aceptamos el primer punto
            self.filtered_positions.append(position)
            self.filtered_timestamps.append(timestamp)
        
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.confidences.append(confidence)
        self.frames_without_detection = 0
        self.last_predicted = None

    def predict_position(self, current_timestamp: float) -> Optional[np.ndarray]:
        """
        Predice la posición de la pelota basada en posiciones anteriores filtradas.
        
        Args:
            current_timestamp: Timestamp actual
            
        Returns:
            Array [x, y] con la posición predicha o None si no se puede predecir
        """
        if len(self.filtered_positions) < self.min_points_for_prediction:
            return None

        try:
            positions = np.array(self.filtered_positions)
            timestamps = np.array(self.filtered_timestamps)

            # Interpolación cuadrática para x e y
            fx = interp1d(timestamps, positions[:, 0], kind='quadratic', 
                         fill_value='extrapolate')
            fy = interp1d(timestamps, positions[:, 1], kind='quadratic', 
                         fill_value='extrapolate')

            predicted_x = float(fx(current_timestamp))
            predicted_y = float(fy(current_timestamp))

            self.last_predicted = np.array([predicted_x, predicted_y])
            return self.last_predicted

        except (ValueError, IndexError):
            return None

    def get_smooth_trajectory(self, trajectory_length: int = 7) -> np.ndarray:
        """
        Genera una trayectoria suavizada usando solo los últimos N puntos filtrados.
        
        Args:
            trajectory_length: Número de puntos anteriores a usar para la trayectoria
            
        Returns:
            Array numpy de puntos [x, y] que forman la trayectoria suavizada
        """
        if len(self.filtered_positions) < 2:
            return np.array([])
            
        try:
            # Tomar solo los últimos N puntos
            positions = np.array(self.filtered_positions[-trajectory_length:])
            timestamps = np.array(self.filtered_timestamps[-trajectory_length:])
            
            # Crear timestamps interpolados
            t_smooth = np.linspace(
                timestamps[0], 
                timestamps[-1], 
                self.trajectory_points
            )
            
            # Interpolar coordenadas x e y
            fx = interp1d(timestamps, positions[:, 0], kind='quadratic')
            fy = interp1d(timestamps, positions[:, 1], kind='quadratic')
            
            # Generar puntos suavizados
            x_smooth = fx(t_smooth)
            y_smooth = fy(t_smooth)
            
            return np.column_stack((x_smooth, y_smooth))
            
        except (ValueError, IndexError):
            # Si hay error en interpolación, devolver los puntos filtrados
            return np.array(self.filtered_positions[-trajectory_length:])

    def get_trajectory_segments(self, trajectory_length: int = 7) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Obtiene segmentos de línea para dibujar la trayectoria suavizada.
        
        Args:
            trajectory_length: Número de puntos anteriores a usar para la trayectoria
            
        Returns:
            Lista de tuplas (punto_inicio, punto_fin) para dibujar líneas
        """
        trajectory_points = self.get_smooth_trajectory(trajectory_length)
        
        if len(trajectory_points) < 2:
            return []
            
        segments = []
        for i in range(len(trajectory_points) - 1):
            start = trajectory_points[i].copy()
            end = trajectory_points[i + 1].copy()
            segments.append((start, end))
            
        return segments

    def reset(self):
        """Reinicia el estado del tracker."""
        self.positions.clear()
        self.timestamps.clear()
        self.confidences.clear()
        self.filtered_positions = []
        self.filtered_timestamps = []
        self.outlier_positions = []
        self.outlier_timestamps = []
        self.last_predicted = None
        self.frames_without_detection = 0