"""
Module for video processing with tracking and interpolation.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from ultralytics import YOLO
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

@dataclass
class TrackingData:
    """Stores tracking data for a complete video."""
    ball_detections: List[Dict[str, Any]]  # List of dicts with pos, timestamp, confidence
    filtered_ball_positions: List[np.ndarray]  # After outlier removal
    filtered_timestamps: List[float]
    interpolated_positions: Dict[int, np.ndarray]  # frame_number -> position
    player_detections: List[pd.DataFrame]
    frame_timestamps: List[float]
    outlier_positions: List[np.ndarray]
    outlier_timestamps: List[float]
    total_frames: int
    fps: float
    width: int
    height: int
    
class VideoProcessor:
    """
    Processes tennis videos with tracking and interpolation.
    """
    
    def __init__(
        self,
        model_players_path: str,
        model_ball_path: str,
        device: str = 'cuda',
        conf_threshold: Dict[str, float] = None,
        trajectory_length: int = 7,
        max_speed: float = 1000.0  # Maximum allowed speed for outlier filtering
    ):
        self.conf_threshold = conf_threshold or {'ball': 0.3, 'players': 0.5}
        self.trajectory_length = trajectory_length
        self.max_speed = max_speed
        
        # Load models
        self.model_players = YOLO(model_players_path)
        self.model_ball = YOLO(model_ball_path)
        
        # Move models to specified device
        self.model_players.to(device)
        self.model_ball.to(device)
        self.device = device
    
    def collect_tracking_data(
        self,
        video_path: str,
        show_progress: bool = True
    ) -> TrackingData:
        """
        First pass: collect all tracking data from the video.
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize collections
        ball_detections = []
        player_detections = []
        frame_timestamps = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_count / fps
            frame_timestamps.append(timestamp)
            
            # Process players
            player_boxes = self._process_players(frame)
            player_detections.append(player_boxes)
            
            # Process ball
            ball_pos, ball_conf = self._process_ball(frame)
            if ball_pos is not None:
                ball_detections.append({
                    'position': ball_pos,
                    'timestamp': timestamp,
                    'confidence': ball_conf,
                    'frame': frame_count
                })
            
            frame_count += 1
            if show_progress and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f'Collecting data: {progress:.1f}% ({frame_count}/{total_frames} frames)', 
                      end='\r')
        
        cap.release()
        
        # Post-process the collected data
        filtered_data = self._filter_outliers(ball_detections, total_frames, fps)
        
        return TrackingData(
            ball_detections=ball_detections,
            filtered_ball_positions=filtered_data['filtered_positions'],
            filtered_timestamps=filtered_data['filtered_timestamps'],
            interpolated_positions=filtered_data['interpolated_positions'],
            player_detections=player_detections,
            frame_timestamps=frame_timestamps,
            outlier_positions=filtered_data['outlier_positions'],
            outlier_timestamps=filtered_data['outlier_timestamps'],
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height
        )
    
    def _filter_outliers(
        self, 
        ball_detections: List[Dict[str, Any]], 
        total_frames: int,
        fps: float
    ) -> Dict[str, Any]:
        """
        Filter outliers using a simple speed-based approach and interpolate ball positions.
        """
        if not ball_detections:
            return {
                'filtered_positions': [],
                'filtered_timestamps': [],
                'interpolated_positions': {},
                'outlier_positions': [],
                'outlier_timestamps': []
            }
        
        # Sort detections by timestamp
        ball_detections.sort(key=lambda x: x['timestamp'])
        
        # Initialize lists for filtering
        filtered_positions = []
        filtered_timestamps = []
        outlier_positions = []
        outlier_timestamps = []
        
        # Always keep first detection
        filtered_positions.append(ball_detections[0]['position'])
        filtered_timestamps.append(ball_detections[0]['timestamp'])
        last_valid_pos = ball_detections[0]['position']
        last_valid_ts = ball_detections[0]['timestamp']
        
        # Simple speed-based filtering
        for detection in ball_detections[1:]:
            curr_pos = detection['position']
            curr_ts = detection['timestamp']
            
            # Calculate speed
            dt = curr_ts - last_valid_ts
            if dt > 0:  # Avoid division by zero
                dx = curr_pos[0] - last_valid_pos[0]
                dy = curr_pos[1] - last_valid_pos[1]
                speed = np.sqrt((dx/dt)**2 + (dy/dt)**2)
                
                if speed <= self.max_speed:
                    filtered_positions.append(curr_pos)
                    filtered_timestamps.append(curr_ts)
                    last_valid_pos = curr_pos
                    last_valid_ts = curr_ts
                else:
                    outlier_positions.append(curr_pos)
                    outlier_timestamps.append(curr_ts)
        
        # Convert to numpy arrays
        filtered_positions = np.array(filtered_positions)
        filtered_timestamps = np.array(filtered_timestamps)
        
        # Create interpolation for all frames
        interpolated_positions = {}
        
        if len(filtered_positions) >= 3:  # Need at least 3 points for quadratic interpolation
            # Create interpolation functions
            fx = interp1d(filtered_timestamps, filtered_positions[:, 0], 
                        kind='quadratic', fill_value='extrapolate')
            fy = interp1d(filtered_timestamps, filtered_positions[:, 1], 
                        kind='quadratic', fill_value='extrapolate')
            
            # Interpolate for every frame
            for frame in range(total_frames):
                timestamp = frame / fps
                try:
                    x = fx(timestamp)
                    y = fy(timestamp)
                    interpolated_positions[frame] = np.array([x, y])
                except ValueError:
                    continue
        
        return {
            'filtered_positions': filtered_positions,
            'filtered_timestamps': filtered_timestamps,
            'interpolated_positions': interpolated_positions,
            'outlier_positions': np.array(outlier_positions) if outlier_positions else [],
            'outlier_timestamps': np.array(outlier_timestamps) if outlier_timestamps else []
        }

    def get_trajectory_segments(
        self,
        tracking_data: TrackingData,
        current_frame: int,
        window_size: int = 30
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get line segments for trajectory visualization around current frame.
        """
        segments = []
        positions = []
        
        # Get positions for window around current frame
        start_frame = max(0, current_frame - window_size)
        end_frame = min(tracking_data.total_frames, current_frame + 1)
        
        for frame in range(start_frame, end_frame):
            if frame in tracking_data.interpolated_positions:
                positions.append(tracking_data.interpolated_positions[frame])
        
        # Create segments from consecutive positions
        for i in range(len(positions) - 1):
            segments.append((positions[i], positions[i + 1]))
            
        return segments

    def process_video(
        self,
        video_path: str,
        output_path: str,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process complete video with tracking and interpolation.
        """
        # First pass: collect and process all tracking data
        tracking_data = self.collect_tracking_data(video_path, show_progress)
        
        # Configure video writer
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, 
            fourcc, 
            tracking_data.fps,
            (tracking_data.width, tracking_data.height)
        )
        
        # Initialize visualizer
        from .visualizer import TrackingVisualizer
        visualizer = TrackingVisualizer(tracking_data.width, tracking_data.height)
        
        # Second pass: create output video with complete trajectory data
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get trajectory segments for current frame
            trajectory_segments = self.get_trajectory_segments(
                tracking_data,
                frame_count,
                self.trajectory_length
            )
            
            # Get players positions
            player_boxes = tracking_data.player_detections[frame_count]
            if len(player_boxes) > 2: 
                player_boxes = player_boxes.sort_values('conf', ascending=False).groupby('class').head(1)
            
            # Get current ball position
            current_ball_pos = None
            if len(player_boxes) == 2:
                current_ball_pos = tracking_data.interpolated_positions.get(frame_count)
            
            # Generate visualization
            frame_vis = visualizer.draw_tracking(
                frame.copy(),
                current_ball_pos,
                player_boxes,
                trajectory_segments,
                frame_count,
                {
                    'total_detections': len(tracking_data.ball_detections),
                    'outliers_filtered': len(tracking_data.outlier_positions)
                }
            )
            
            out.write(frame_vis)
            
            frame_count += 1
            if show_progress and frame_count % 30 == 0:
                progress = (frame_count / tracking_data.total_frames) * 100
                print(f'Creating video: {progress:.1f}% ({frame_count}/{tracking_data.total_frames} frames)', 
                      end='\r')
        
        cap.release()
        out.release()
        
        # Compute statistics
        stats = {
            'total_frames': tracking_data.total_frames,
            'ball_detections': len(tracking_data.ball_detections),
            'interpolated_positions': len(tracking_data.interpolated_positions),
            'outliers_filtered': len(tracking_data.outlier_positions),
            'player_detections': sum(len(boxes) for boxes in tracking_data.player_detections)
        }
        
        return stats

    def _process_players(self, frame: np.ndarray) -> pd.DataFrame:
        """Process players in frame."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_3ch = np.stack((frame_gray,) * 3, axis=-1)
        
        results = self.model_players(
            frame_gray_3ch,
            conf=self.conf_threshold['players'],
            device=self.device,
            verbose=False
        )
        
        boxes_df = pd.DataFrame()
        if len(results) > 0:
            r = results[0]
            if len(r.boxes) > 0:
                boxes = r.boxes
                boxes_data = boxes.xywh.cpu().numpy()
                boxes_df = pd.DataFrame(boxes_data, columns=['x', 'y', 'w', 'h'])
                boxes_df['conf'] = boxes.conf.cpu().numpy()
                boxes_df['class'] = boxes.cls.cpu().numpy()
                boxes_df = boxes_df[boxes_df['class'] < 2]
                
        return boxes_df
        
    def _process_ball(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Process ball in frame.
        Returns (position, confidence) tuple or (None, None) if no detection.
        """
        results = self.model_ball(
            frame,
            conf=self.conf_threshold['ball'],
            device=self.device,
            verbose=False
        )
        
        if len(results) > 0:
            r = results[0]
            boxes = r.boxes
            
            if len(boxes) > 0:
                best_box = boxes[boxes.conf.argmax()]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                conf = float(best_box.conf[0])
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                return np.array([center_x, center_y]), conf
        
        return None, None