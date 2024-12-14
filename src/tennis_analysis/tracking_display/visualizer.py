"""
Module for visualization of tracking in video with interpolation support.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List, Any

class TrackingVisualizer:
    """Visualizer for tennis tracking with interpolation support."""
    
    def __init__(
        self,
        width: int,
        height: int,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ):
        self.width = width
        self.height = height
        self.colors = colors or {
            'ball': (0, 0, 255),      # Red
            'trajectory': (0, 165, 255),      # Orange
            'player1': (0, 255, 0),    # Green
            'player2': (255, 0, 0),    # Blue
            'overlay_bg': (0, 0, 0),   # Black
            'overlay_text': (255, 255, 255)  # White
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_scale_overlay = 0.4
        self.thickness = 2
        self.thickness_overlay = 1
        
        
    def draw_tracking(
        self,
        frame: np.ndarray,
        current_ball_pos: Optional[np.ndarray],
        player_boxes: pd.DataFrame,
        trajectory_segments: List[Tuple[np.ndarray, np.ndarray]],
        frame_number: int,
        stats: Dict[str, int],
        show_stats: bool = True
    ) -> np.ndarray:
        """
        Draws tracking visualization on frame.
        
        Args:
            frame: Frame to draw on
            current_ball_pos: Current ball position (if available)
            player_boxes: DataFrame with player detections
            trajectory_segments: List of point pairs for trajectory
            frame_number: Current frame number
            stats: Dictionary with tracking statistics
            show_stats: Whether to show statistics overlay
        """
        if len(player_boxes) == 2:
            # Draw players
            frame = self._draw_players(frame, player_boxes)
        
            # Draw ball trajectory
            frame = self._draw_ball_trajectory(frame, trajectory_segments)
        
        # Draw current ball position
        if current_ball_pos is not None:
            frame = self._draw_ball(frame, current_ball_pos)
        
        # Add statistics overlay
        if show_stats:
            frame = self.add_overlay(frame, {
                'frame': frame_number,
                'detections': stats['total_detections'],
                'outliers_filtered': stats['outliers_filtered'],
                'players': len(player_boxes)
            })
        
        return frame
    
    def _draw_players(
        self,
        frame: np.ndarray,
        player_boxes: pd.DataFrame
    ) -> np.ndarray:
        """Draws player detections."""
        for _, box in player_boxes.iterrows():
            # Get color based on player class
            color = self.colors[f'player{int(box["class"]) + 1}']
            
            # Calculate box coordinates
            x, y = int(box['x']), int(box['y'])
            w, h = int(box['w']), int(box['h'])
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f'Player {int(box["class"]) + 1}: {box["conf"]:.2f}'
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                self.font,
                self.font_scale,
                color,
                self.thickness
            )
        
        return frame
    
    def _draw_ball_trajectory(
        self,
        frame: np.ndarray,
        trajectory_segments: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Draws ball trajectory."""
        # Draw trajectory segments with gradually increasing thickness and alpha
        n_segments = len(trajectory_segments)
        if n_segments > 0:
            for i, (start, end) in enumerate(trajectory_segments):
                # Make more recent segments more visible
                alpha = 0.3 + 0.7 * (i / n_segments)
                thickness = max(1, int(1 + 2 * (i / n_segments)))
                
                overlay = frame.copy()
                cv2.line(
                    overlay,
                    tuple(map(int, start)),
                    tuple(map(int, end)),
                    self.colors['trajectory'],
                    thickness
                )
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def _draw_ball(
        self,
        frame: np.ndarray,
        ball_position: np.ndarray,
    ) -> np.ndarray:
        """Draws ball position."""
        x, y = map(int, ball_position)
        cv2.circle(frame, (x, y), 5, self.colors['ball'], -1)
        return frame
        
    def add_overlay(
        self,
        frame: np.ndarray,
        stats: Dict[str, Any],
        position: str = 'top-left'
    ) -> np.ndarray:
        """Adds statistics overlay to frame."""
        texts = [
            f"Frame: {stats['frame']}",
            f"Ball detections: {stats['detections']}",
            f"Players: {stats['players']}",
            f"Outliers filtered: {stats['outliers_filtered']}"
        ]
        
        # Calculate dimensions
        padding = 10
        line_height = 20
        overlay_width = 200  # Increased for longer text
        overlay_height = (len(texts) + 1) * line_height
        
        # Set position
        if position == 'top-left':
            x, y = padding, padding
        elif position == 'top-right':
            x, y = self.width - overlay_width - padding, padding
        else:
            x, y = padding, padding
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + overlay_width, y + overlay_height),
            self.colors['overlay_bg'],
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add text
        for i, text in enumerate(texts):
            cv2.putText(
                frame,
                text,
                (x + 5, y + (i + 1) * line_height),
                self.font,
                self.font_scale_overlay,
                self.colors['overlay_text'],
                self.thickness_overlay
            )
        
        return frame