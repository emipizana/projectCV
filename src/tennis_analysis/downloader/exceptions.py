"""
Excepciones personalizadas para el módulo downloader.
"""

class DownloaderError(Exception):
    """Excepción base para errores en el downloader."""
    pass

class VideoFormatError(DownloaderError):
    """Error al obtener formatos de video."""
    pass

class DownloadError(DownloaderError):
    """Error durante la descarga del video."""
    pass

class TimeFormatError(DownloaderError):
    """Error en el formato de tiempo especificado."""
    pass