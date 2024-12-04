"""
Module for projecting tennis court positions onto a minimap overlay.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Any
import pandas as pd
from pathlib import Path
import logging

class CourtProjector:
    """Projects player positions onto a miniature court visualization."""
    
    def __init__(
        self,
        court_dims: Tuple[float, float] = (23.77, 10.97),  # Standard tennis court dimensions in meters
        minimap_size: Tuple[int, int] = (160, 80),  # Size of the minimap in pixels
        position: Tuple[int, int] = (420, 420),  # Top-right corner position
        background_color: Tuple[int, int, int] = (255, 255, 255),  # White background
        court_color: Tuple[int, int, int] = (0, 100, 0),  # Dark green court
        player_colors: Tuple[Tuple[int, int, int], ...] = ((255, 0, 0), (0, 0, 255))  # Red and blue for players
    ):
        """
        Initialize the CourtProjector.
        
        Args:
            court_dims: Real court dimensions (length, width) in meters
            minimap_size: Size of the minimap overlay (width, height) in pixels
            position: Position of the minimap in the video frame (x, y)
            background_color: RGB color for minimap background
            court_color: RGB color for court lines
            player_colors: RGB colors for player markers
        """
        self.court_dims = court_dims
        self.minimap_size = minimap_size
        self.position = position
        self.background_color = background_color
        self.court_color = court_color
        self.player_colors = player_colors
        
        # Calculate scaling factors
        self.scale_x = minimap_size[0] / court_dims[0]
        self.scale_y = minimap_size[1] / court_dims[1]
        
        # Court landmarks in meters (from center)
        self.court_landmarks = {
            'singles_line': (court_dims[1]/2, court_dims[0]/2),
            'service_line': (6.40, court_dims[1]/2),
            'net': (0, court_dims[1]/2)
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def create_court_base(self) -> np.ndarray:
        """Create the base court minimap with lines."""
        # Create base image
        court = np.full((self.minimap_size[1], self.minimap_size[0], 3), 
                       self.background_color, 
                       dtype=np.uint8)
        
        # Calculate court corners and lines in pixels
        half_width = int(self.minimap_size[0] / 2)
        half_height = int(self.minimap_size[1] / 2)
        
        # Draw outer court lines
        cv2.rectangle(court, 
                     (0, 0),
                     (self.minimap_size[0]-1, self.minimap_size[1]-1),
                     self.court_color,
                     1)
        
        # Draw net line
        net_x = half_width
        cv2.line(court,
                 (net_x, 0),
                 (net_x, self.minimap_size[1]),
                 self.court_color,
                 1)
        
        # Draw service lines
        service_line_dist = int(6.40 * self.scale_x)
        cv2.line(court,
                 (half_width - service_line_dist, 0),
                 (half_width - service_line_dist, self.minimap_size[1]),
                 self.court_color,
                 1)
        cv2.line(court,
                 (half_width + service_line_dist, 0),
                 (half_width + service_line_dist, self.minimap_size[1]),
                 self.court_color,
                 1)
        
        return court
        
    def project_positions(
        self,
        player_boxes: Any,
        homography_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Project player positions onto court minimap.
        
        Args:
            player_boxes: DataFrame with player detections (x, y coordinates)
            homography_matrix: Optional transformation matrix for perspective correction
            
        Returns:
            Minimap with player positions marked
        """
        court = self.create_court_base()
        
        if len(player_boxes) == 2:
            for box in player_boxes:
                # Get player position (assuming bottom center of bounding box is feet position)
                x = 0# this should be box.x
                y = 0# this should be box.y
                
                # if homography_matrix is not None:
                #     # Apply perspective transformation if available
                #     pos = np.array([[x, y, 1]], dtype=np.float32)
                #     transformed = cv2.perspectiveTransform(pos[None, :, :], homography_matrix)
                #     court_x, court_y = transformed[0][0][:2]
                # else:
                #     # Simple scaling if no homography available
                #     court_x = x * self.scale_x
                #     court_y = y * self.scale_y
                
                # Draw player marker
                color = (255, 0, 0)
                cv2.circle(court, 
                          (int(0), int(0)),
                          3,  # radius
                          color,
                          -1)  # filled circle
                
        return court
    
    def add_minimap(
        self,
        frame: np.ndarray,
        player_boxes: Any,
        homography_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Add court minimap to video frame.
        
        Args:
            frame: Video frame to modify
            player_boxes: DataFrame with player detections
            homography_matrix: Optional perspective transformation matrix
            
        Returns:
            Frame with minimap overlay
        """
        # Create minimap with player positions
        minimap = self.project_positions(player_boxes, homography_matrix)
        
        # Define minimap region in frame
        x, y = self.position
        h, w = minimap.shape[:2]
        
        # Create slightly larger background for border
        border = 2
        bg = np.full((h + 2*border, w + 2*border, 3), (0, 0, 0), dtype=np.uint8)
        bg[border:-border, border:-border] = minimap
        
        # Add minimap to frame
        frame_region = frame[y:y+h+2*border, x:x+w+2*border]
        alpha = 0.8
        frame[y:y+h+2*border, x:x+w+2*border] = cv2.addWeighted(
            frame_region, 1-alpha,
            bg, alpha,
            0
        )
        
        return frame
