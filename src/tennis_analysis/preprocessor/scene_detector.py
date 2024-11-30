"""
Módulo para detectar cambios de escena en videos de tenis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from .video_loader import VideoLoader

class SceneDetector:
    """Detector de cambios de escena en videos."""
    
    def __init__(
        self,
        video_loader: VideoLoader,
        scene_threshold: float = 45.0,
        resize_width: int = 320,
        resize_height: int = 180
    ):
        """
        Inicializa el detector de escenas.
        
        Args:
            video_loader: Instancia de VideoLoader
            scene_threshold: Umbral para detectar cambios
            resize_width: Ancho para redimensionar frames
            resize_height: Alto para redimensionar frames
        """
        self.loader = video_loader
        self.scene_threshold = scene_threshold
        self.resize_dims = (resize_width, resize_height)
        
    def detect_scenes(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        example_frame: Optional[np.ndarray] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Detecta cambios de escena en el video.
        
        Args:
            start_frame: Frame inicial para procesar
            end_frame: Frame final (None para procesar hasta el final)
            example_frame: Frame de ejemplo para comparación
            
        Returns:
            Tuple con lista de frames donde hay cambios y sus puntuaciones
        """
        scene_changes = []
        change_scores = []
        
        # Determinar frame final
        if end_frame is None:
            end_frame = self.loader.frame_count
            
        # Leer primer frame
        prev_frame = self.loader.read_frame(start_frame)
        if prev_frame is None:
            return [], []
            
        # Redimensionar frame previo
        if example_frame is not None:
            small_prev = cv2.resize(example_frame, self.resize_dims)
        else:
            small_prev = cv2.resize(prev_frame, self.resize_dims)
            
        # Procesar frames
        for frame_num in range(start_frame + 1, end_frame):
            # Leer y procesar frame actual
            frame = self.loader.read_frame(frame_num)
            if frame is None:
                break
                
            small_frame = cv2.resize(frame, self.resize_dims)
            
            # Calcular diferencia
            diff = cv2.absdiff(small_frame, small_prev)
            diff_score = np.mean(diff)
            change_scores.append(diff_score)
            
            # Detectar cambio de escena
            if diff_score > self.scene_threshold:
                scene_changes.append(frame_num)
            
            # Actualizar frame previo
            if example_frame is None:
                small_prev = small_frame
                
            # Reportar progreso
            if (frame_num - start_frame) % 100 == 0:
                print(f"Procesando frame {frame_num}/{end_frame}")
                
        return scene_changes, change_scores
    
    def optimize_threshold(
        self,
        sample_size: int = 1000,
        start_frame: int = 0
    ) -> float:
        """
        Optimiza el umbral de detección usando una muestra de frames.
        
        Args:
            sample_size: Número de frames para la muestra
            start_frame: Frame inicial para la muestra
            
        Returns:
            Umbral optimizado
        """
        # Implementar optimización del umbral
        pass