"""
Módulo para cargar y manejar videos.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class VideoLoader:
    """Clase para cargar y manejar videos."""
    
    def __init__(self, video_path: str):
        """
        Inicializa el loader de video.
        
        Args:
            video_path: Ruta al archivo de video
        
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el video no se puede abrir
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {self.video_path}")
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {self.video_path}")
            
        # Propiedades del video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def read_frame(self, frame_num: int) -> Optional[np.ndarray]:
        """
        Lee un frame específico del video.
        
        Args:
            frame_num: Número de frame a leer
            
        Returns:
            np.ndarray o None si el frame no se pudo leer
        """
        if not 0 <= frame_num < self.frame_count:
            return None
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_frames(self, start_frame: int, end_frame: int, step: int = 1) -> np.ndarray:
        """
        Obtiene un rango de frames del video.
        
        Args:
            start_frame: Frame inicial
            end_frame: Frame final
            step: Intervalo entre frames
            
        Returns:
            np.ndarray con los frames
        """
        frames = []
        for frame_num in range(start_frame, end_frame, step):
            frame = self.read_frame(frame_num)
            if frame is not None:
                frames.append(frame)
        return np.array(frames)
    
    def release(self):
        """Libera los recursos del video."""
        if self.cap is not None:
            self.cap.release()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        
    @property
    def video_info(self) -> dict:
        """Retorna información básica del video."""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.frame_count / self.fps
        }