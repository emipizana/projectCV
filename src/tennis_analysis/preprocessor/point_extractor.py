"""
Módulo para extraer segmentos de video de puntos individuales.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from .data_structures import TennisPoint
from .video_loader import VideoLoader

class PointExtractor:
    """Extrae segmentos de video para puntos individuales."""
    
    def __init__(self, video_loader: VideoLoader):
        """
        Args:
            video_loader: Instancia de VideoLoader
        """
        self.loader = video_loader
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    def extract_point(
        self,
        point: TennisPoint,
        output_path: str,
        add_overlay: bool = True,
        point_number: Optional[int] = None
    ) -> bool:
        """
        Extrae un punto individual a un nuevo archivo de video.
        
        Args:
            point: Punto de tenis a extraer
            output_path: Ruta donde guardar el video
            add_overlay: Si se debe añadir información superpuesta
            point_number: Número del punto para el overlay
            
        Returns:
            bool indicando si la extracción fue exitosa
        """
        try:
            # Preparar writer
            frame = self.loader.read_frame(point.start_frame)
            if frame is None:
                return False
                
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                output_path,
                self.fourcc,
                self.loader.fps,
                (width, height)
            )
            
            # Escribir frames
            total_frames = point.end_frame - point.start_frame
            for rel_frame_num in range(total_frames):
                abs_frame_num = point.start_frame + rel_frame_num
                frame = self.loader.read_frame(abs_frame_num)
                
                if frame is None:
                    break
                    
                if add_overlay and point_number is not None:
                    frame = self._add_overlay(
                        frame,
                        point_number,
                        rel_frame_num,
                        total_frames,
                        point.confidence
                    )
                
                writer.write(frame)
            
            writer.release()
            return True
            
        except Exception as e:
            print(f"Error al extraer punto: {str(e)}")
            return False
    
    def _add_overlay(
        self,
        frame: np.ndarray,
        point_num: int,
        frame_num: int,
        total_frames: int,
        confidence: float
    ) -> np.ndarray:
        """
        Añade información superpuesta al frame.
        """
        overlay = frame.copy()
        
        # Configurar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        # Información a mostrar
        texts = [
            (f"Point: {point_num}", (10, 30)),
            (f"Conf: {confidence:.2f}", (10, 60)),
            (f"Frame: {frame_num}/{total_frames}", (10, 90))
        ]
        
        # Añadir fondo semi-transparente
        cv2.rectangle(overlay, (5, 5), (200, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Añadir textos
        for text, pos in texts:
            cv2.putText(frame, text, pos, font, font_scale,
                       (0, 0, 0), thickness + 1)
            cv2.putText(frame, text, pos, font, font_scale,
                       (255, 255, 255), thickness)
        
        return frame
    
    def preview_point(self, point: TennisPoint, num_frames: int = 4):
        """
        Obtiene frames representativos del punto para previsualización.
        
        Args:
            point: Punto a previsualizar
            num_frames: Número de frames a extraer
            
        Returns:
            Lista de frames representativos
        """
        frames = []
        frame_indices = np.linspace(
            point.start_frame,
            point.end_frame,
            num_frames,
            dtype=int
        )
        
        for idx in frame_indices:
            frame = self.loader.read_frame(idx)
            if frame is not None:
                frames.append(frame)
        
        return frames