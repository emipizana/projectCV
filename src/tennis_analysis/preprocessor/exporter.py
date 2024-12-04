"""
Módulo para exportar resultados del preprocesamiento.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm

from .data_structures import TennisPoint
from .video_loader import VideoLoader
from .point_extractor import PointExtractor

class TennisExporter:
    """Maneja la exportación de puntos y metadatos."""
    
    def __init__(self, video_loader: VideoLoader, output_dir: str):
        """
        Args:
            video_loader: Instancia de VideoLoader
            output_dir: Directorio base para la exportación
        """
        self.loader = video_loader
        self.output_dir = Path(output_dir)
        self.extractor = PointExtractor(video_loader)
        
    def export_points(
        self,
        points: List[TennisPoint],
        start_index: int = 1,
        add_overlay: bool = True,
        progress_callback: Optional[Callable[[], None]] = None,
        show_tqdm: bool = False
    ) -> Dict[str, Any]:
        """
        Exporta los puntos detectados y genera metadatos.
        
        Args:
            points: Lista de puntos a exportar
            start_index: Índice inicial para la numeración
            add_overlay: Si se debe añadir información superpuesta
            progress_callback: Función callback para reportar progreso
            show_tqdm: Si se debe mostrar la barra de progreso de tqdm
            
        Returns:
            Diccionario con resumen de la exportación
        """
        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        videos_dir = self.output_dir
        videos_dir.mkdir(exist_ok=True)
        
        # Preparar metadatos
        metadata = self._prepare_metadata(points)
        successful_exports = 0
        failed_exports = []
        
        # Exportar cada punto
        if show_tqdm:
            point_iterator = tqdm(enumerate(points, start=start_index), total=len(points), desc="Exportando puntos")
        else:
            point_iterator = enumerate(points, start=start_index)
        
            
        for i, point in point_iterator:
            point_data = self._create_point_metadata(point, i)
            output_path = videos_dir / f"point_{i:03d}.mp4"
            
            if self.extractor.extract_point(
                point,
                str(output_path),
                add_overlay,
                i
            ):
                successful_exports += 1
                point_data["export_status"] = "success"
                point_data["video_path"] = str(output_path.relative_to(self.output_dir))
            else:
                failed_exports.append(i)
                point_data["export_status"] = "failed"
            
            metadata["points"].append(point_data)
            
            if progress_callback:
                progress_callback()

        if len(points) == 0:
            point = TennisPoint(
                start_frame = 0,
                end_frame = self.loader.frame_count,
                start_time = 0,
                end_time = self.loader.frame_count / self.loader.fps,
                score_shown = False,
                confidence = 0,
                scene_change_score = 0
            )
            
            point_data = self._create_point_metadata(point, 0)
            output_path = videos_dir / f"point_001.mp4"
            
            if self.extractor.extract_point(
                point,
                str(output_path),
                add_overlay,
                0
            ):
                successful_exports += 1
                point_data["export_status"] = "success"
                point_data["video_path"] = str(output_path.relative_to(self.output_dir))
            else:
                failed_exports.append(0)
                point_data["export_status"] = "failed"
            
            metadata["points"].append(point_data)
            
            if progress_callback:
                progress_callback()
        
        # Guardar metadatos
        self._save_metadata(metadata)
        
        # Generar y retornar resumen
        return self._create_summary(
            total_points=len(points),
            successful=successful_exports,
            failed=failed_exports
        )
    
    def _prepare_metadata(self, points: List[TennisPoint]) -> Dict[str, Any]:
        """Prepara la estructura base de metadatos."""
        return {
            "export_info": {
                "date": datetime.now().isoformat(),
                "total_frames": self.loader.frame_count,
                "video_fps": self.loader.fps,
                "total_points": len(points),
                "video_resolution": f"{self.loader.width}x{self.loader.height}"
            },
            "points": []
        }
    
    def _create_point_metadata(self, point: TennisPoint, index: int) -> Dict[str, Any]:
        """Crea metadatos para un punto individual."""
        return {
            "point_id": index,
            **point.to_dict(),
            "video_file": f"point_{index:03d}.mp4"
        }
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Guarda los metadatos en un archivo JSON."""
        metadata_path = self.output_dir / "points_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
    
    def _create_summary(
        self,
        total_points: int,
        successful: int,
        failed: List[int]
    ) -> Dict[str, Any]:
        """Crea un resumen de la exportación."""
        return {
            "total_points": total_points,
            "successful_exports": successful,
            "failed_exports": failed,
            "output_directory": str(self.output_dir),
            "metadata_path": str(self.output_dir / "points_metadata.json")
        }