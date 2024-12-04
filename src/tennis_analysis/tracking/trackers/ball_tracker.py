"""
Module for ball tracking with trajectory smoothing and outlier detection.
"""

from collections import deque
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple, List

class BallTracker:
    """
    Tennis ball tracker with position prediction, outlier filtering,
    and smoothed trajectory.
    """
    
    def __init__(
        self,
        buffer_size: int = 30,
        max_frames_to_predict: int = 5,
        min_points_for_prediction: int = 3,
        max_speed: float = 1000.0,  # Maximum allowed speed for outlier filtering
        trajectory_points: int = 50
    ):
        """
        Args:
            buffer_size: Size of buffer to store previous positions
            max_frames_to_predict: Maximum number of frames to use prediction
            min_points_for_prediction: Minimum number of points needed for prediction
            max_speed: Maximum allowed speed (pixels/second) for outlier filtering
            trajectory_points: Number of points to use in interpolation
        """
        self.positions = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.confidences = deque(maxlen=buffer_size)
        self.last_predicted = None
        self.frames_without_detection = 0
        self.max_frames_to_predict = max_frames_to_predict
        self.min_points_for_prediction = min_points_for_prediction
        self.max_speed = max_speed
        self.trajectory_points = trajectory_points
        
        # For storing filtered positions
        self.filtered_positions = []
        self.filtered_timestamps = []
        self.outlier_positions = []
        self.outlier_timestamps = []
        
        # Interpolation functions
        self.fx = None
        self.fy = None

    def add_detection(self, position: np.ndarray, timestamp: float, confidence: float = 1.0):
        """
        Adds a new ball detection.
        """
        position = np.asarray(position).flatten()
        timestamp = float(timestamp)
        
        # Filter outliers based on speed
        if len(self.filtered_positions) > 0:
            dt = timestamp - self.filtered_timestamps[-1]
            if dt > 0:  # Avoid division by zero
                dx = position[0] - self.filtered_positions[-1][0]
                dy = position[1] - self.filtered_positions[-1][1]
                speed = np.sqrt((dx/dt)**2 + (dy/dt)**2)
                
                if speed <= self.max_speed:
                    self.filtered_positions.append(position)
                    self.filtered_timestamps.append(timestamp)
                    # Update interpolation functions
                    self._update_interpolation()
                else:
                    self.outlier_positions.append(position)
                    self.outlier_timestamps.append(timestamp)
                    # Don't add to main buffer if outlier
                    return
        else:
            # Always accept first point
            self.filtered_positions.append(position)
            self.filtered_timestamps.append(timestamp)
            # Initialize interpolation functions
            self._update_interpolation()
        
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.confidences.append(confidence)
        self.frames_without_detection = 0
        self.last_predicted = None

    def _update_interpolation(self):
        """Updates interpolation functions with filtered positions."""
        if len(self.filtered_positions) >= 3:  # Need at least 3 points for quadratic
            positions = np.array(self.filtered_positions)
            timestamps = np.array(self.filtered_timestamps)
            
            try:
                self.fx = interp1d(timestamps, positions[:, 0], 
                                 kind='quadratic', fill_value='extrapolate')
                self.fy = interp1d(timestamps, positions[:, 1], 
                                 kind='quadratic', fill_value='extrapolate')
            except ValueError:
                self.fx = None
                self.fy = None

    def predict_position(self, current_timestamp: float) -> Optional[np.ndarray]:
        """
        Predicts ball position using current interpolation functions.
        """
        if self.fx is None or self.fy is None:
            return None

        try:
            predicted_x = float(self.fx(current_timestamp))
            predicted_y = float(self.fy(current_timestamp))
            self.last_predicted = np.array([predicted_x, predicted_y])
            return self.last_predicted
        except ValueError:
            return None

    def get_trajectory_segments(self, trajectory_length: int = 7) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Gets line segments for the smoothed trajectory visualization.
        Similar to the notebook implementation.
        """
        if len(self.filtered_positions) < trajectory_length:
            return []
        
        current_timestamp = self.filtered_timestamps[-1]
        from_timestamp = max(current_timestamp - 1.0,0)
            
        # Get the last N positions and timestamps
        recent_positions = np.array(self.filtered_positions[-trajectory_length:])
        recent_timestamps = np.array(self.filtered_timestamps[-trajectory_length:])
        
        try:
            # Create interpolation functions for recent trajectory
            fx_recent = interp1d(self.filtered_timestamps, self.filtered_positions[:, 0], 
                               kind='quadratic', fill_value='extrapolate')
            fy_recent = interp1d(self.filtered_timestamps, self.filtered_positions[:, 1], 
                               kind='quadratic', fill_value='extrapolate')
            
            # Create timestamps for smooth trajectory
            t_smooth = np.linspace(from_timestamp, current_timestamp, 20)
            
            # Generate smooth points
            points = []
            for t in t_smooth:
                try:
                    x = fx_recent(t)
                    y = fy_recent(t)
                    points.append(np.array([x, y]))
                except ValueError:
                    continue
            
            # Create segments from points
            segments = []
            for i in range(len(points) - 1):
                segments.append((points[i], points[i + 1]))
                
            return segments
            
        except (ValueError, IndexError):
            # If interpolation fails, return segments from filtered positions
            segments = []
            for i in range(len(recent_positions) - 1):
                segments.append((recent_positions[i], recent_positions[i + 1]))
            return segments

    def reset(self):
        """Resets tracker state."""
        self.positions.clear()
        self.timestamps.clear()
        self.confidences.clear()
        self.filtered_positions = []
        self.filtered_timestamps = []
        self.outlier_positions = []
        self.outlier_timestamps = []
        self.last_predicted = None
        self.frames_without_detection = 0
        self.fx = None
        self.fy = None