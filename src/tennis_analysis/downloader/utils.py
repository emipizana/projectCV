"""
Utilidades para el módulo downloader.
"""

from datetime import datetime
from typing import List, Dict, Any
import yt_dlp
from .exceptions import TimeFormatError, VideoFormatError

def time_to_seconds(time_str: str) -> int:
    """
    Convierte una cadena de tiempo a segundos.
    
    Args:
        time_str: Tiempo en formato 'HH:MM:SS' o 'MM:SS'
    
    Returns:
        int: Tiempo en segundos
    
    Raises:
        TimeFormatError: Si el formato de tiempo es inválido
    """
    try:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    except ValueError:
        try:
            time_obj = datetime.strptime(time_str, '%M:%S')
            return time_obj.minute * 60 + time_obj.second
        except ValueError:
            raise TimeFormatError("El tiempo debe estar en formato HH:MM:SS o MM:SS")

def get_available_formats(url: str) -> List[Dict[str, Any]]:
    """
    Obtiene los formatos disponibles para un video.
    
    Args:
        url: URL del video
    
    Returns:
        List[Dict]: Lista de formatos disponibles
    
    Raises:
        VideoFormatError: Si no se pueden obtener los formatos
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            formats = []
            
            for f in info['formats']:
                if f.get('vcodec', 'none') != 'none':
                    formats.append({
                        'format_id': f['format_id'],
                        'ext': f['ext'],
                        'resolution': f.get('resolution', 'unknown'),
                        'filesize': f.get('filesize', 0),
                        'fps': f.get('fps', 0),
                        'vcodec': f['vcodec'],
                        'acodec': f.get('acodec', 'none')
                    })
            
            # Ordenar por resolución y fps
            formats.sort(key=lambda x: (
                int(x['resolution'].split('x')[0]) if 'x' in x['resolution'] else 0,
                x['fps'] or 0
            ), reverse=True)
            
            return formats
            
        except Exception as e:
            raise VideoFormatError(f"No se pudieron obtener los formatos: {str(e)}")