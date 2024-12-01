"""
Módulo para detectar cambios de escena en videos de tenis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Callable
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
        self.loader = video_loader
        self.scene_threshold = scene_threshold
        self.resize_dims = (resize_width, resize_height)
        
    def detect_scenes(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        example_frame: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Detecta cambios de escena en el video usando lectura secuencial.
        """
        scene_changes = []
        change_scores = []
        
        # Determinar frame final
        if end_frame is None:
            end_frame = self.loader.frame_count
            
        total_frames = end_frame - start_frame
        UPDATE_INTERVAL = max(1, total_frames // 1000)
        
        # Posicionar el video en el frame inicial
        self.loader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Leer el primer frame
        ret, prev_frame = self.loader.cap.read()
        if not ret:
            return [], []
            
        # Redimensionar frame previo
        if example_frame is not None:
            small_prev = cv2.resize(example_frame, self.resize_dims)
        else:
            small_prev = cv2.resize(prev_frame, self.resize_dims)
        
        current_frame = start_frame + 1
        
        # Procesar frames secuencialmente
        while current_frame < end_frame:
            # Leer siguiente frame
            ret, frame = self.loader.cap.read()
            if not ret:
                break
                
            # Procesar frame actual
            small_frame = cv2.resize(frame, self.resize_dims)
            
            # Calcular diferencia
            diff = cv2.absdiff(small_frame, small_prev)
            diff_score = np.mean(diff)
            change_scores.append(diff_score)
            
            # Detectar cambio de escena
            if diff_score > self.scene_threshold:
                scene_changes.append(current_frame)
            
            # Actualizar frame previo
            if example_frame is None:
                small_prev = small_frame.copy()
                
            # Reportar progreso
            if (current_frame - start_frame) % UPDATE_INTERVAL == 0:
                if progress_callback:
                    progress_callback(current_frame - start_frame, total_frames)
                else:
                    print(f"Procesando frame {current_frame}/{end_frame}")
            
            current_frame += 1
                
        # Asegurar que reportamos 100% al final
        if progress_callback:
            progress_callback(total_frames, total_frames)
                
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