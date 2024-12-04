"""
Module for assembling multiple tennis point videos into a single video.
"""

import cv2
import os
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import logging

class VideoAssembler:
    """Class to assemble multiple tennis point videos into one."""
    
    def __init__(
        self,
        input_folder: str,
        output_path: str,
        transition_frames: int = 30,
        output_fps: int = 30,
        resize_dims: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the VideoAssembler.
        
        Args:
            input_folder: Path to folder containing point videos
            output_path: Path where the assembled video will be saved
            transition_frames: Number of black frames between points
            output_fps: FPS of the output video
            resize_dims: Optional (width, height) to resize all videos
        """
        self.input_folder = Path(input_folder)
        self.output_path = Path(output_path)
        self.transition_frames = transition_frames
        self.output_fps = output_fps
        self.resize_dims = resize_dims
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_video_files(self) -> List[Path]:
        """Get all video files from input folder sorted by name."""
        video_extensions = {'.mp4', '.avi', '.mov'}
        video_files = []
        
        for file in self.input_folder.iterdir():
            if file.suffix.lower() in video_extensions:
                video_files.append(file)
                
        # Sort videos by name
        return sorted(video_files)
        
    def get_video_properties(self, video_path: Path) -> Tuple[int, int, int]:
        """
        Get video properties.
        
        Returns:
            Tuple of (width, height, total_frames)
        """
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return width, height, total_frames
        
    def create_transition_frame(self, width: int, height: int) -> np.ndarray:
        """Create a black transition frame with given dimensions."""
        return np.zeros((height, width, 3), dtype=np.uint8)
        
    def assemble_videos(self) -> None:
        """
        Main method to assemble all videos into one.
        """
        video_files = self.get_video_files()
        if not video_files:
            self.logger.error("No video files found in input folder")
            return
            
        self.logger.info(f"Found {len(video_files)} video files")
        
        # Get dimensions from first video if resize not specified
        first_width, first_height, _ = self.get_video_properties(video_files[0])
        width = self.resize_dims[0] if self.resize_dims else first_width
        height = self.resize_dims[1] if self.resize_dims else first_height
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.output_fps,
            (width, height)
        )
        
        # Create transition frame
        transition = self.create_transition_frame(width, height)
        
        # Process each video
        for idx, video_file in enumerate(video_files):
            self.logger.info(f"Processing video {idx + 1}/{len(video_files)}: {video_file.name}")
            
            cap = cv2.VideoCapture(str(video_file))
            
            # Add transition before each point (except first)
            if idx > 0:
                for _ in range(self.transition_frames):
                    out.write(transition)
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize if needed
                if self.resize_dims:
                    frame = cv2.resize(frame, self.resize_dims)
                    
                out.write(frame)
                
            cap.release()
            
        # Cleanup
        out.release()
        self.logger.info(f"Video assembly completed. Output saved to: {self.output_path}")
        
    def get_total_duration(self) -> float:
        """
        Calculate total duration of assembled video in seconds.
        
        Returns:
            Float representing total seconds
        """
        total_frames = 0
        video_files = self.get_video_files()
        
        # Sum frames from all videos
        for video_file in video_files:
            _, _, frames = self.get_video_properties(video_file)
            total_frames += frames
            
        # Add transition frames
        if len(video_files) > 1:
            total_frames += self.transition_frames * (len(video_files) - 1)
            
        return total_frames / self.output_fps