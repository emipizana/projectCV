"""
Simplified Tennis Analysis Pipeline
Processes tennis videos to detect and analyze points.
"""

import logging
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from rich.console import Console
from rich.progress import Progress
import psutil

import pandas as pd

# Initialize Rich console
console = Console()

@dataclass
class PipelineStats:
    """Simple container for pipeline statistics"""
    total_points: int = 0
    successful_exports: int = 0
    failed_exports: int = 0
    tracking_successes: int = 0
    tracking_failures: int = 0
    processing_time: float = 0
    peak_memory_mb: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_points': self.total_points,
            'successful_exports': self.successful_exports,
            'failed_exports': self.failed_exports,
            'tracking_successes': self.tracking_successes,
            'tracking_failures': self.tracking_failures,
            'processing_time': self.processing_time,
            'peak_memory_mb': self.peak_memory_mb
        }

class TennisPipeline:
    """Main pipeline for tennis video analysis"""
    
    def __init__(
        self,
        output_dir: str,
        model_players_path: str,
        model_ball_path: str,
        example_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        self.output_dir = Path(output_dir)
        self.model_players_path = model_players_path
        self.model_ball_path = model_ball_path
        self.device = device
        self.stats = PipelineStats()
        self.example_path = example_path
        
        # Setup logging
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / "pipeline.log")
            ]
        )
        self.logger = logging.getLogger("tennis_pipeline")
        
    def download_video(self, video_url: str, start_time: Optional[str] = None, 
                      duration: Optional[str] = None) -> Path:
        """Download video from URL with optional time segments"""
        from tennis_analysis.downloader import VideoDownloader
        
        video_path = self.output_dir / "match.mp4"
        downloader = VideoDownloader()
        
        if start_time and duration:
            downloader.download_segment(
                url=video_url,
                start_time=start_time,
                duration=duration,
                output_path=str(video_path)
            )
        else:
            downloader.download_video(url=video_url, output_path=str(video_path))
            
        return video_path

    def detect_points(self, video_path: Path) -> list:
        """Detect tennis points in the video"""
        from tennis_analysis.preprocessor import VideoLoader, SceneDetector, PointIdentifier
        
        loader = VideoLoader(str(video_path))
        detector = SceneDetector(loader)
        identifier = PointIdentifier(loader)
        
        with Progress() as progress:
            task = progress.add_task("Detecting scenes...", total=100)
            
            def update_progress(current: int, total: int):
                progress.update(task, completed=(current / total) * 100)
            
            if self.example_path is not None:
                scene_changes, scores = detector.detect_scenes(
                    progress_callback=update_progress,
                    example_frame=loader.read_frame(self.example_path)
                )
            else:
                scene_changes, scores = detector.detect_scenes(
                    progress_callback=update_progress
                )
        
        points = identifier.identify_points(scene_changes, scores)
        self.stats.total_points = len(points)
        return points, loader

    def export_points(self, points: list, video_loader) -> None:
        """Export individual point videos"""
        from tennis_analysis.preprocessor import TennisExporter
        
        points_dir = self.output_dir / "points"
        
        if points_dir.exists():
            import shutil
            shutil.rmtree(points_dir)
        exporter = TennisExporter(video_loader, str(points_dir))
        
        with Progress() as progress:
            task = progress.add_task("Exporting points...", total=len(points))
            results = exporter.export_points(
                points,
                progress_callback=lambda: progress.advance(task)
            )
        
        self.stats.successful_exports = results['successful_exports']
        self.stats.failed_exports = len(results['failed_exports'])

    def track_points(self, points: list) -> list:
        """Process points with player and ball tracking"""
        from tennis_analysis.tracking_display import VideoProcessor
        
        points_dir = self.output_dir / "points"
        tracked_dir = self.output_dir / "tracked_points"
        if tracked_dir.exists():
            import shutil
            shutil.rmtree(tracked_dir)
        tracked_dir.mkdir(exist_ok=True)
        
        processor = VideoProcessor(
            model_players_path=self.model_players_path,
            model_ball_path=self.model_ball_path,
            device=self.device
        )
        
        tracking_stats = []
        with Progress() as progress:
            total_points = max(len(points), 1)
            task = progress.add_task("Analyzing points...", total=total_points)
            
            for i in range(1, total_points+1):
                input_path = points_dir / f"point_{i:03d}.mp4"
                output_path = tracked_dir / f"point_{i:03d}_tracked.mp4"
                
                if input_path.exists():
                    try:
                        stats = processor.process_video(str(input_path), str(output_path))
                        tracking_stats.append(stats)
                        self.stats.tracking_successes += 1
                    except Exception as e:
                        self.stats.tracking_failures += 1
                        self.logger.error(f"Failed to track point {i}: {str(e)}")
                
                progress.advance(task)
                
        return tracking_stats
    
    
    def process_minimaps(self, points: list, player_boxes: Any) -> list:
        """Add minimap overlays to tracked point videos."""
        from tennis_analysis.projector.minimap import add_minimaps
        
        tracked_dir = self.output_dir / "tracked_points"
        minimap_dir = self.output_dir / "minimap_points"
        
        if minimap_dir.exists():
            import shutil
            shutil.rmtree(minimap_dir)
        minimap_dir.mkdir(exist_ok=True)
        
        minimap_stats = add_minimaps(
            tracked_dir=str(tracked_dir),
            output_dir=str(minimap_dir),
            player_boxes=player_boxes
        )
        
        return minimap_stats

    def assemble_tracked_videos(self) -> None:
        """Assemble all tracked point videos into a single video"""
        from tennis_analysis.postprocessor.assembler import VideoAssembler
        
        tracked_dir = self.output_dir / "tracked_points"
        assembled_path = self.output_dir / "assembled_match.mp4"
        
        with console.status("[bold green]Assembling final video..."):
            assembler = VideoAssembler(
                input_folder=str(tracked_dir),
                output_path=str(assembled_path),
                transition_frames=30,  # Default transitions
                output_fps=30         # Default fps
            )
            assembler.assemble_videos()
            self.logger.info(f"Video assembly completed. Output saved to: {assembled_path}")

def process_complete(
    video_url: Optional[str],
    output_dir: str,
    model_players_path: str,
    model_ball_path: str,
    device: str = 'cuda',
    example_path: Optional[str] = None,
    start_time: Optional[str] = None,
    duration: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute complete tennis analysis pipeline with monitoring.
    This function maintains compatibility with the original pipeline.
    """
    start = time.time()
    
    # Initialize pipeline
    pipeline = TennisPipeline(
        output_dir=output_dir,
        model_players_path=model_players_path,
        model_ball_path=model_ball_path,
        device=device
    )
    
    try:
        # Get video
        if video_url:
            video_path = Path(pipeline.download_video(video_url, start_time, duration))
        else:
            video_path = Path('./examples/output/match.mp4')
            
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Process video
        points, video_loader = pipeline.detect_points(video_path)
        pipeline.export_points(points, video_loader)
        tracking_stats = pipeline.track_points(points)
        
        # Minimap
        player_detections = [s['player_detections'] for s in tracking_stats]
        minimap_stats = pipeline.process_minimaps(points, player_detections)
        
        # Añadir el ensamblado final
        pipeline.assemble_tracked_videos()
        
        # Finalize statistics
        pipeline.stats.processing_time = time.time() - start
        pipeline.stats.peak_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Prepare summary
        summary = {
            'total_points': pipeline.stats.total_points,
            'exported_points': pipeline.stats.successful_exports,
            #'tracking_results': tracking_stats,
            #'minimap_results': minimap_stats,
            'output_directory': str(pipeline.output_dir),
            'assembled_video': str(pipeline.output_dir / "assembled_match.mp4")# ,
            #'metrics': pipeline.stats.to_dict()
        }
        
        # Save report
        report_path = pipeline.output_dir / "pipeline_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        return summary
        
    except Exception as e:
        pipeline.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tennis Analysis Pipeline')
    parser.add_argument('--video-url', help='URL of the tennis video')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--model-players', required=True, help='YOLO players model path')
    parser.add_argument('--model-ball', required=True, help='YOLO ball model path')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--example_path', help='Example target path')
    parser.add_argument('--start-time', help='Start time (HH:MM:SS)')
    parser.add_argument('--duration', help='Duration (HH:MM:SS)')
    
    args = parser.parse_args()
    
    try:
        with console.status("[bold green]Running tennis analysis pipeline..."):
            summary = process_complete(
                video_url=args.video_url,
                output_dir=args.output_dir,
                model_players_path=args.model_players,
                model_ball_path=args.model_ball,
                device=args.device,
                example_path=args.example_path,
                start_time=args.start_time,
                duration=args.duration
            )
        
        console.print("[green]Pipeline completed successfully!")
        console.print(summary)
        
    except Exception as e:
        console.print(f"[red]Pipeline failed: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()