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
        minimap_size: tuple = (320, 280),
        position: tuple = (1500, 20)
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
        
        if video_files:  # If there are any videos
                #Here we are just taking the first point detected
                try:
                    video_file = video_files[0]  # Take just the first one
                    output_path = self.output_dir / f"{video_file.stem}_with_minimap.mp4"
                    self.process_single_video(video_file, output_path)
                    stats['processed_videos'] += 1
                    
                    if progress_callback:
                        progress_callback()
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {video_file}: {str(e)}")
                    stats['failed_videos'] += 1
            
        return stats
        ##Code to process all the videos

        # for video_file in video_files:
        #     try:
        #         output_path = self.output_dir / f"{video_file.stem}_with_minimap.mp4"
        #         self.process_single_video(video_file, output_path)
        #         stats['processed_videos'] += 1
                
        #         if progress_callback:
        #             progress_callback()
                    
        #     except Exception as e:
        #         self.logger.error(f"Failed to process {video_file}: {str(e)}")
        #         stats['failed_videos'] += 1
                
        # return stats    
    
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
            ##
            for box in player_boxes_r:
                x = box[0]  # x coordinate
                y = box[1]  # y coordinate
                w = box[2]  # width
                h = box[3]  # height
                player_class = box[5]  # player class (0 or 1)

                # Calculate and draw bottom center point
                center_x = int(x + w//2)
                center_y = int(y+h)  # Using top y
                cv2.circle(frame, (center_x-50, center_y-100), 5, (255, 0, 0), -1)
        
            #Hard code cordinates frame 0
            # 74, 76, 173, 175
            # Point 74: (282, 840)
            # Point 76: (1333, 828)
            # Point 173: (513, 288)
            # Point 175: (1098, 286)

            Hard_code_cords = [
                (513, 288),    # Top left to match (215, 52)
                (1098, 286),   # Top right to match (385, 52)
                (282, 840),    # Bottom left to match (215, 548)
                (1333, 828)    # Bottom right to match (385, 548)
            ]
            if i == 0:
                orig_cords = Hard_code_cords
            else:
                _, white_edges = self.projector.detect_court_edges(frame)
                _, corner_coords = self.projector.process_court_corners(white_edges, frame)
                if corner_coords is not None:
                    next_cords = []
                    for cord in orig_cords: 
                        k = self.projector.nearest_point(cord, corner_coords)
                        next_cords.append(corner_coords[k])
                    orig_cords = next_cords

            #Draw corners in frame
            for point in orig_cords:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0,0,255), -1)

            ####################################
            #Classification of points from first frame

            #     # Draw all coordinates to get points of interest
            #     if corner_coords is not None:
            #         for idx, coord in enumerate(corner_coords):
            #             x, y = coord
            #             # Draw circle at coordinate
            #             cv2.circle(debug_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            #             # Add coordinate number
            #             cv2.putText(debug_frame, str(idx), (int(x)+10, int(y)+10), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #     cv2.imwrite('frame_0_coords.jpg', debug_frame)


            #     # Check points of interest
            #     cv2.imwrite('frame_1_coords.jpg', debug_frame)
            #     if i == 0:
            #         print("Frame 0 corners:", corner_coords)
            #         points_of_interest = [74, 76, 173, 175]
            #         print("\nCoordinates for points of interest:")
            #         for idx in points_of_interest:
            #             if idx < len(corner_coords):
            #                 print(f"Point {idx}: {corner_coords[idx]}")
            #         debug_frame = frame.copy()

        ##############################################################################

            homography = self.projector.calculate_homography(orig_cords, None)

            ##When homography is None
            if homography is None:
                frame_with_minimap = cv2.resize(frame, self.projector.minimap_size)
            #Add_minimap does the projection
            frame_with_minimap = self.projector.add_minimap(
                frame.copy(),
                player_boxes_r,
                homography
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
