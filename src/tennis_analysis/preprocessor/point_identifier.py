"""
Módulo para identificar puntos individuales de tenis.
"""

import cv2
import numpy as np
from typing import List
from .data_structures import TennisPoint
from .video_loader import VideoLoader

class PointIdentifier:
    """Identifica puntos individuales de tenis en el video."""
    
    def __init__(
        self,
        video_loader: VideoLoader,
        min_point_duration: float = 3.0,  # segundos
        max_point_duration: float = 60.0,  # segundos
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            video_loader: Instancia de VideoLoader
            min_point_duration: Duración mínima de un punto en segundos
            max_point_duration: Duración máxima de un punto en segundos
            confidence_threshold: Umbral mínimo de confianza para validar puntos
        """
        self.loader = video_loader
        self.min_frames = int(min_point_duration * video_loader.fps)
        self.max_frames = int(max_point_duration * video_loader.fps)
        self.confidence_threshold = confidence_threshold
    
    def identify_points(
        self,
        scene_changes: List[int],
        change_scores: List[float]
    ) -> List[TennisPoint]:
        """
        Identifica puntos de tenis basados en los cambios de escena detectados.
        
        Args:
            scene_changes: Lista de frames donde se detectaron cambios
            change_scores: Puntuaciones de cambio para cada frame
            
        Returns:
            Lista de puntos de tenis identificados
        """
        if not scene_changes or not change_scores:
            return []
            
        points = []
        score_offset = scene_changes[0]
        
        for i in range(len(scene_changes) - 1):
            start_frame = scene_changes[i]
            end_frame = scene_changes[i + 1]
            duration = end_frame - start_frame
            
            if self.min_frames <= duration <= self.max_frames:
                score_index = start_frame - score_offset
                if score_index < 0 or score_index >= len(change_scores):
                    continue
                
                # Calcular confianza y verificar marcador
                confidence = self._calculate_confidence(duration, start_frame)
                score_shown = self._check_scoreboard(start_frame)
                
                point = TennisPoint(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=round(start_frame / self.loader.fps, 4),
                    end_time=round(end_frame / self.loader.fps, 4),
                    score_shown=str(score_shown),
                    confidence=round(confidence, 4),
                    scene_change_score=round(change_scores[score_index], 4)
                )
                points.append(point)
        
        return self._validate_points(points)
    
    def _calculate_confidence(self, duration: int, start_frame: int) -> float:
        """
        Calcula la confianza del punto basado en diversos factores.
        """
        confidence = 1.0
        duration_secs = duration / self.loader.fps
        
        # Factor de duración
        if 10 <= duration_secs <= 30:
            confidence *= 1.0
        elif 5 <= duration_secs <= 40:
            confidence *= 0.8
        else:
            confidence *= 0.5
        
        # Factor de marcador
        if self._check_scoreboard(start_frame):
            confidence *= 1.2
        
        return min(confidence, 1.0)
    
    def _check_scoreboard(self, frame_num: int) -> bool:
        """
        Detecta la presencia de un marcador en el frame.
        """
        frame = self.loader.read_frame(frame_num)
        if frame is None:
            return False
        
        # Analizar solo la parte superior del frame
        top_portion = frame[:int(frame.shape[0]*0.2), :, :]
        gray = cv2.cvtColor(top_portion, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.mean(edges) > 50
    
    def _validate_points(self, points: List[TennisPoint]) -> List[TennisPoint]:
        """
        Filtra puntos según criterios de validación.
        """
        return [
            point for point in points
            if (self.min_frames <= (point.end_frame - point.start_frame) <= self.max_frames)
            and point.confidence >= self.confidence_threshold
        ]