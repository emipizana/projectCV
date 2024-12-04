"""
Module for adding court minimap to tracked tennis videos.
"""

import cv2
import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from rich.progress import Progress
import pandas as pd

from .court_projector import CourtProjector

class MinimapPostProcessor:
    """Adds court minimap to tracked tennis videos."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        player_boxes: Any,
        minimap_size: tuple = (160, 80),
        position: tuple = (20, 20)
    ):
        """
        Initialize the MinimapPostProcessor.
        
        Args:
            input_dir: Directory containing tracked videos
            output_dir: Directory for videos with minimaps
            minimap_size: Size of the minimap overlay (width, height)
            position: Position of minimap in frame (x, y)
            player_boxes: List of player bounding boxes for each frame so it is like [(x, y, w, h, ...), ...]
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.player_boxes = player_boxes
        
        # Initialize court projector
        self.projector = CourtProjector(
            minimap_size=minimap_size,
            position=position
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def process_videos(self, progress_callback=None) -> Dict[str, Any]:
        """
        Process all tracked videos and add minimap.
        
        Returns:
            Dictionary with processing statistics
        """
        video_files = list(self.input_dir.glob('*_tracked.mp4'))
        stats = {
            'total_videos': len(video_files),
            'processed_videos': 0,
            'failed_videos': 0
        }
        
        for video_file in video_files:
            try:
                output_path = self.output_dir / f"{video_file.stem}_with_minimap.mp4"
                self.process_single_video(video_file, output_path)
                stats['processed_videos'] += 1
                
                if progress_callback:
                    progress_callback()
                    
            except Exception as e:
                self.logger.error(f"Failed to process {video_file}: {str(e)}")
                stats['failed_videos'] += 1
                
        return stats
    
    def process_single_video(self, input_path: Path, output_path: Path) -> None:
        """
        Process a single video file, adding the minimap overlay.
        """
        cap = cv2.VideoCapture(str(input_path))
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add minimap overlay
            if i < len(self.player_boxes):
                player_boxes_r = self.player_boxes[i]
            else:
                player_boxes_r = []
            frame_with_minimap = self.projector.add_minimap(
                frame.copy(),
                player_boxes_r
            )
            
            out.write(frame_with_minimap)
            i += 1
        
        cap.release()
        out.release()

def add_minimaps(tracked_dir: str, output_dir: str, player_boxes: Any) -> Dict[str, Any]:
    """
    Main function to add minimaps to all tracked videos.
    
    Args:
        tracked_dir: Directory containing tracked videos
        output_dir: Directory for output videos with minimaps
        
    Returns:
        Dictionary with processing statistics
    """
    processor = MinimapPostProcessor(tracked_dir, output_dir, player_boxes)
    
    with Progress() as progress:
        task = progress.add_task("Adding minimaps...", total=len(list(Path(tracked_dir).glob('*_tracked.mp4'))))
        
        stats = processor.process_videos(
            progress_callback=lambda: progress.advance(task)
        )
    
    return stats