"""
Clase principal para la descarga de videos.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import yt_dlp
from .utils import time_to_seconds, get_available_formats
from .exceptions import DownloadError

class VideoDownloader:
    """Clase para descargar segmentos de video de alta calidad."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Inicializa el downloader.
        
        Args:
            temp_dir: Directorio temporal para descargas. Si es None, usa './temp_download'
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path("temp_download")
        
    def download_segment(
        self,
        url: str,
        start_time: str,
        duration: str,
        output_path: str,
        format_id: Optional[str] = None,
        interactive: bool = True
    ) -> str:
        """
        Descarga un segmento de video.
        
        Args:
            url: URL del video
            start_time: Tiempo de inicio (HH:MM:SS o MM:SS)
            duration: Duración (HH:MM:SS o MM:SS)
            output_path: Ruta de salida para el video
            format_id: ID del formato a descargar. Si es None, usa el mejor formato o permite selección
            interactive: Si es True, permite seleccionar el formato interactivamente
        
        Returns:
            str: Ruta del archivo descargado
        
        Raises:
            DownloadError: Si hay un error en la descarga
        """
        try:
            # Crear directorio temporal
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = self.temp_dir / "full_video.mp4"
            
            # Convertir tiempos a segundos
            start_seconds = time_to_seconds(start_time)
            duration_seconds = time_to_seconds(duration)
            
            # Obtener formato
            if interactive and not format_id:
                format_id = self._select_format(url)
            elif not format_id:
                format_id = 'bestvideo+bestaudio'
            
            try:
                # Descargar video completo
                self._download_full_video(url, temp_path, format_id)
                
                # Extraer segmento
                self._extract_segment(
                    temp_path,
                    output_path,
                    start_seconds,
                    duration_seconds
                )
                
                return output_path
                
            finally:
                # Limpiar archivos temporales
                self._cleanup()
                
        except Exception as e:
            raise DownloadError(f"Error en la descarga: {str(e)}")
    
    def _select_format(self, url: str) -> str:
        """Permite al usuario seleccionar el formato de video."""
        formats = get_available_formats(url)
        
        print("\nFormatos disponibles:")
        for i, f in enumerate(formats):
            print(f"{i+1}. Resolución: {f['resolution']}")
            print(f"   FPS: {f['fps']}")
            print(f"   Codec: {f['vcodec']}")
            print(f"   ID: {f['format_id']}\n")
        
        selection = input("Selecciona número de formato (Enter para mejor calidad): ")
        return formats[int(selection)-1]['format_id'] if selection.strip() else 'bestvideo+bestaudio'
    
    def _download_full_video(self, url: str, output_path: str, format_id: str):
        """Descarga el video completo."""
        ydl_opts = {
            'format': format_id,
            'outtmpl': str(output_path),
            'quiet': False,
            'no_warnings': True,
            'merge_output_format': 'mp4',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    
    def _extract_segment(
        self,
        input_path: str,
        output_path: str,
        start_seconds: int,
        duration_seconds: int
    ):
        """Extrae un segmento del video usando ffmpeg."""
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ss', str(start_seconds),
            '-t', str(duration_seconds),
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-y',
            str(output_path)
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise DownloadError(f"Error en ffmpeg: {stderr.decode()}")
    
    def _cleanup(self):
        """Limpia los archivos temporales."""
        temp_file = self.temp_dir / "full_video.mp4"
        if temp_file.exists():
            temp_file.unlink()
        if self.temp_dir.exists():
            self.temp_dir.rmdir()