"""
Pipeline completo de análisis de tenis.
"""

import argparse
from pathlib import Path
from typing import Dict, Any
import os
import sys
from pathlib import Path

# Obtener el path raíz del proyecto
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from tennis_analysis.downloader import VideoDownloader
from tennis_analysis.preprocessor import (
    VideoLoader,
    SceneDetector,
    PointIdentifier,
    TennisExporter
)
from tennis_analysis.postprocessing import VideoProcessor

def process_complete(
    video_url: str,
    output_dir: str,
    model_players_path: str,
    model_ball_path: str,
    device: str = 'cuda',
    start_time: str = None,
    duration: str = None
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo: descarga → preprocesamiento → tracking.
    
    Args:
        video_url: URL del video
        output_dir: Directorio base para resultados
        model_players_path: Ruta al modelo de jugadores
        model_ball_path: Ruta al modelo de pelota
        device: Dispositivo para inferencia
        start_time: Tiempo de inicio para descarga (HH:MM:SS)
        duration: Duración a descargar (HH:MM:SS)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Descargar video
    print("\n1. Descargando video...")
    downloader = VideoDownloader()
    video_path = output_dir / "full_match.mp4"
    
    if start_time and duration:
        downloader.download_segment(
            url=video_url,
            start_time=start_time,
            duration=duration,
            output_path=str(video_path)
        )
    else:
        downloader.download_video(url=video_url, output_path=str(video_path))
    
    # 2. Preprocesar y segmentar en puntos
    print("\n2. Preprocesando video...")
    loader = VideoLoader(str(video_path))
    detector = SceneDetector(loader)
    identifier = PointIdentifier(loader)
    
    # Detectar escenas y puntos
    scene_changes, change_scores = detector.detect_scenes()
    points = identifier.identify_points(scene_changes, change_scores)
    
    # Exportar puntos individuales
    print(f"\nPuntos detectados: {len(points)}")
    points_dir = output_dir / "points"
    exporter = TennisExporter(loader, str(points_dir))
    export_results = exporter.export_points(points)
    
    # 3. Procesar cada punto con tracking
    print("\n3. Procesando tracking para cada punto...")
    processor = VideoProcessor(
        model_players_path=model_players_path,
        model_ball_path=model_ball_path,
        device=device
    )
    
    tracked_dir = output_dir / "tracked_points"
    tracked_dir.mkdir(exist_ok=True)
    
    tracking_stats = []
    for i, point in enumerate(points, 1):
        print(f"\nProcesando punto {i}/{len(points)}")
        input_path = points_dir / f"point_{i:03d}.mp4"
        output_path = tracked_dir / f"point_{i:03d}_tracked.mp4"
        
        if input_path.exists():
            stats = processor.process_video(
                str(input_path),
                str(output_path),
                show_progress=True
            )
            tracking_stats.append(stats)
    
    # 4. Generar resumen
    summary = {
        'total_points': len(points),
        'exported_points': export_results['successful_exports'],
        'tracking_results': tracking_stats,
        'output_directory': str(output_dir)
    }
    
    print("\nProcesamiento completo!")
    print(f"Videos procesados guardados en: {tracked_dir}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Pipeline completo de análisis de tenis')
    parser.add_argument('video_url', help='URL del video')
    parser.add_argument('output_dir', help='Directorio para resultados')
    parser.add_argument('--model-players', required=True,
                      help='Ruta al modelo YOLO de jugadores')
    parser.add_argument('--model-ball', required=True,
                      help='Ruta al modelo YOLO de pelota')
    parser.add_argument('--device', default='cuda',
                      choices=['cuda', 'cpu', 'mps'],
                      help='Dispositivo para inferencia')
    parser.add_argument('--start-time', help='Tiempo de inicio (HH:MM:SS)')
    parser.add_argument('--duration', help='Duración (HH:MM:SS)')
    
    args = parser.parse_args()
    
    summary = process_complete(
        video_url=args.video_url,
        output_dir=args.output_dir,
        model_players_path=args.model_players,
        model_ball_path=args.model_ball,
        device=args.device,
        start_time=args.start_time,
        duration=args.duration
    )
    
    # Mostrar resumen
    print("\nResumen del procesamiento:")
    print(f"Total de puntos detectados: {summary['total_points']}")
    print(f"Puntos exportados exitosamente: {summary['exported_points']}")
    print(f"\nResultados guardados en: {summary['output_directory']}")

if __name__ == '__main__':
    main()