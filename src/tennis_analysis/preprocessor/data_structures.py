"""
Estructuras de datos para el preprocesamiento de videos.
"""

from dataclasses import dataclass

@dataclass
class TennisPoint:
    """Representa un punto individual de tenis."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    score_shown: bool
    confidence: float
    scene_change_score: float

    @property
    def duration(self) -> float:
        """Duración del punto en segundos."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Convierte el punto a diccionario para exportación."""
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'score_shown': self.score_shown,
            'confidence': self.confidence,
            'scene_change_score': self.scene_change_score
        }