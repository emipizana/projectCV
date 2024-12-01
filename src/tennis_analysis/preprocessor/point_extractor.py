"""
Módulo optimizado para extraer segmentos de video de puntos individuales.
"""

import cv2
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List
from .data_structures import TennisPoint
from .video_loader import VideoLoader

class PointExtractor:
    """Extrae segmentos de video para puntos individuales usando métodos optimizados."""
    
    def __init__(self, video_loader: VideoLoader):
        """
        Args:
            video_loader: Instancia de VideoLoader
        """
        self.loader = video_loader
        self.video_path = video_loader.video_path
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verifica si ffmpeg está disponible en el sistema."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True)
            self.use_ffmpeg = True
        except FileNotFoundError:
            print("FFmpeg no encontrado. Usando método alternativo más lento.")
            self.use_ffmpeg = False
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def extract_point(
        self,
        point: TennisPoint,
        output_path: str,
        add_overlay: bool = True,
        point_number: Optional[int] = None
    ) -> bool:
        """
        Extrae un punto individual a un nuevo archivo de video usando FFmpeg.
        
        Args:
            point: Punto de tenis a extraer
            output_path: Ruta donde guardar el video
            add_overlay: Si se debe añadir información superpuesta
            point_number: Número del punto para el overlay
            
        Returns:
            bool indicando si la extracción fue exitosa
        """
        try:
            if self.use_ffmpeg:
                return self._extract_with_ffmpeg(point, output_path, add_overlay, point_number)
            else:
                return self._extract_with_opencv(point, output_path, add_overlay, point_number)
                
        except Exception as e:
            print(f"Error al extraer punto: {str(e)}")
            return False

    def _extract_with_ffmpeg(
        self,
        point: TennisPoint,
        output_path: str,
        add_overlay: bool = True,
        point_number: Optional[int] = None
    ) -> bool:
        """Extrae el punto usando FFmpeg para máxima velocidad."""
        start_time = round(point.start_time,4)
        duration = round(point.duration,4)
        cmd = [
            'ffmpeg',
            '-y',  # Sobrescribir archivo si existe
            '-ss', str(start_time),  # Tiempo de inicio
            '-i', str(self.video_path),  # Archivo de entrada
            '-t', str(duration),  # Duración
            '-c', 'copy',  # Copiar sin recodificar
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except subprocess.SubprocessError:
            return False

    def _extract_with_opencv(
        self,
        point: TennisPoint,
        output_path: str,
        add_overlay: bool = True,
        point_number: Optional[int] = None
    ) -> bool:
        """Método de respaldo usando OpenCV."""
        try:
            # Configurar el writer
            self.loader.cap.set(cv2.CAP_PROP_POS_FRAMES, point.start_frame)
            ret, frame = self.loader.cap.read()
            if not ret:
                return False
                
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                output_path,
                self.fourcc,
                self.loader.fps,
                (width, height)
            )
            
            # Usar chunks más grandes para lectura/escritura
            CHUNK_SIZE = 30  # Procesar 30 frames a la vez
            total_frames = point.end_frame - point.start_frame
            
            for chunk_start in range(point.start_frame, point.end_frame, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, point.end_frame)
                frames = []
                
                # Leer chunk de frames
                for _ in range(chunk_end - chunk_start):
                    ret, frame = self.loader.cap.read()
                    if not ret:
                        break
                    if add_overlay and point_number is not None:
                        frame = self._add_overlay(
                            frame,
                            point_number,
                            len(frames),
                            total_frames,
                            point.confidence
                        )
                    frames.append(frame)
                
                # Escribir chunk de frames
                for frame in frames:
                    writer.write(frame)
            
            writer.release()
            return True
            
        except Exception as e:
            print(f"Error en extracción OpenCV: {str(e)}")
            return False

    def _add_overlay(
        self,
        frame: np.ndarray,
        point_num: int,
        frame_num: int,
        total_frames: int,
        confidence: float
    ) -> np.ndarray:
        """Añade información superpuesta al frame."""
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

    def preview_point(self, point: TennisPoint, num_frames: int = 4) -> List[np.ndarray]:
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
            self.loader.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.loader.cap.read()
            if ret:
                frames.append(frame.copy())
            
        return frames