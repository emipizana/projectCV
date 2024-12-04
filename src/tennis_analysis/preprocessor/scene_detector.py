"""
Módulo para detectar cambios de escena en videos de tenis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Callable
from .video_loader import VideoLoader
from typing import Dict

class SceneDetector:
    """Detector de cambios de escena en videos."""
    
    def __init__(
        self,
        video_loader: VideoLoader,
        scene_threshold: float = 45.0,
        resize_width: int = 320,
        resize_height: int = 180,
        static_window: int = 30,  # Number of frames to confirm static scene
        min_static_duration: int = 45  # Minimum frames for a tennis point
    ):
        self.loader = video_loader
        self.scene_threshold = scene_threshold
        self.resize_dims = (resize_width, resize_height)
        self.static_window = static_window
        self.min_static_duration = min_static_duration
        self.static_threshold = None
        self.dynamic_threshold = None
        self.detected_points = []  # Store points here
        
    def detect_scenes(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        example_frame: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Detecta cambios de escena en el video usando lectura secuencial.
        También detecta puntos de tenis pero los almacena internamente.
        
        Returns:
            Tuple containing:
            - List of scene change frames
            - List of change scores
        """
        scene_changes = []
        change_scores = []
        self.detected_points = []  # Reset points
        
        # Determinar frame final
        if end_frame is None:
            end_frame = self.loader.frame_count
            
        total_frames = end_frame - start_frame
        UPDATE_INTERVAL = max(1, total_frames // 1000)
        
        # Si no tenemos umbrales optimizados, calcularlos
        if self.static_threshold is None:
            self._optimize_thresholds(min(1000, total_frames), start_frame)
        
        # Posicionar el video en el frame inicial
        self.loader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Leer el primer frame
        ret, prev_frame = self.loader.cap.read()
        if not ret:
            return [], []
            
        # Redimensionar frame previo
        if example_frame is not None:
            small_prev = cv2.resize(example_frame, self.resize_dims)
        else:
            small_prev = cv2.resize(prev_frame, self.resize_dims)
        
        current_frame = start_frame + 1
        
        # Variables para detección de puntos
        static_frame_count = 0
        potential_point_start = None
        recent_diffs = []
        
        # Procesar frames secuencialmente
        while current_frame < end_frame:
            # Leer siguiente frame
            ret, frame = self.loader.cap.read()
            if not ret:
                break
                
            # Procesar frame actual
            small_frame = cv2.resize(frame, self.resize_dims)
            
            # Calcular diferencia
            diff = cv2.absdiff(small_frame, small_prev)
            diff_score = np.mean(diff)
            change_scores.append(diff_score)
            recent_diffs.append(diff_score)
            
            # Mantener ventana de diferencias recientes
            if len(recent_diffs) > self.static_window:
                recent_diffs.pop(0)
            
            # Detectar cambio de escena
            if diff_score > self.scene_threshold:
                scene_changes.append(current_frame)
                # Reset point detection si hay cambio de escena
                static_frame_count = 0
                potential_point_start = None
            
            # Detectar secuencias estáticas (posibles puntos)
            mean_recent_diff = np.mean(recent_diffs) if recent_diffs else diff_score
            
            if mean_recent_diff < self.static_threshold:
                if potential_point_start is None:
                    potential_point_start = current_frame
                static_frame_count += 1
            else:
                # Si teníamos una secuencia estática suficientemente larga, guardarla como punto
                if (potential_point_start is not None and 
                    static_frame_count >= self.min_static_duration):
                    self.detected_points.append({
                        'start_frame': potential_point_start,
                        'end_frame': current_frame,
                        'duration': current_frame - potential_point_start
                    })
                static_frame_count = 0
                potential_point_start = None
            
            # Actualizar frame previo
            if example_frame is None:
                small_prev = small_frame.copy()
                
            # Reportar progreso
            if (current_frame - start_frame) % UPDATE_INTERVAL == 0:
                if progress_callback:
                    progress_callback(current_frame - start_frame, total_frames)
                else:
                    print(f"Procesando frame {current_frame}/{end_frame}")
            
            current_frame += 1
        
        # Verificar último punto potencial
        if (potential_point_start is not None and 
            static_frame_count >= self.min_static_duration):
            self.detected_points.append({
                'start_frame': potential_point_start,
                'end_frame': current_frame,
                'duration': current_frame - potential_point_start
            })
                
        # Asegurar que reportamos 100% al final
        if progress_callback:
            progress_callback(total_frames, total_frames)
            
        plot_scene_detection(change_scores, scene_changes, self.static_threshold, self.dynamic_threshold, self.scene_threshold)
                
        return scene_changes, change_scores
    
    def get_detected_points(self) -> List[Dict[str, int]]:
        """
        Returns the list of detected tennis points.
        
        Returns:
            List of dictionaries containing:
            - start_frame: Frame where point starts
            - end_frame: Frame where point ends
            - duration: Duration in frames
        """
        return self.detected_points
    
    def _optimize_thresholds(
        self,
        sample_size: int = 1000,
        start_frame: int = 0
    ) -> None:
        """
        Optimiza los umbrales de detección usando una muestra de frames.
        """
        # Position video at start frame
        self.loader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize collections
        frame_diffs = []
        
        # Process sample frames
        for i in range(sample_size):
            ret, frame = self.loader.cap.read()
            if not ret:
                break
                
            # Resize frame
            small_frame = cv2.resize(frame, self.resize_dims)
            
            # For first frame, just store and continue
            if i == 0:
                prev_frame = small_frame
                continue
            
            # Calculate frame difference
            diff = cv2.absdiff(small_frame, prev_frame)
            diff_score = np.mean(diff)
            frame_diffs.append(diff_score)
            
            # Update previous frame
            prev_frame = small_frame
        
        if not frame_diffs:
            self.static_threshold = self.scene_threshold * 0.2
            self.dynamic_threshold = self.scene_threshold
            return
            
        # Analyze frame differences distribution
        diffs_array = np.array(frame_diffs)
        
        # Calculate statistics
        mean_diff = np.mean(diffs_array)
        std_diff = np.std(diffs_array)
        
        # Set thresholds
        self.static_threshold = mean_diff - std_diff/2
        self.dynamic_threshold = 0.8*mean_diff
        
        # Adjust if needed based on percentiles
        static_frames = np.sum(diffs_array < self.static_threshold)
        if static_frames < 0.2 * len(diffs_array):
            self.static_threshold = np.percentile(diffs_array, 20)
            
        dynamic_frames = np.sum(diffs_array > self.dynamic_threshold)
        if dynamic_frames < 0.1 * len(diffs_array):
            self.dynamic_threshold = np.percentile(diffs_array, 90)
        
        # Update scene threshold if not manually set
        if self.scene_threshold == 45.0:  # Default value
            self.scene_threshold = self.dynamic_threshold
            
import matplotlib.pyplot as plt
import numpy as np

def plot_scene_detection(change_scores, scene_changes, static_threshold, dynamic_threshold, scene_threshold):
    """
    Plot scene detection analysis including:
    - Frame differences over time
    - Various thresholds
    - Detected scene changes
    """
    plt.figure(figsize=(15, 8))
    
    # Plot frame differences
    frames = np.arange(len(change_scores))
    plt.plot(frames, change_scores, 'b-', alpha=0.5, label='Frame Differences')
    
    # Plot thresholds as horizontal lines
    plt.axhline(y=static_threshold, color='g', linestyle='--', label='Static Threshold')
    plt.axhline(y=dynamic_threshold, color='r', linestyle='--', label='Dynamic Threshold')
    plt.axhline(y=scene_threshold, color='purple', linestyle='--', label='Scene Threshold')
    
    # Plot scene changes as vertical lines
    for change in scene_changes:
        plt.axvline(x=change, color='red', alpha=0.3)
    
    # Annotate key points
    for change in scene_changes:
        plt.plot(change, change_scores[change] if change < len(change_scores) else 0, 
                'ro', markersize=8)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Difference Score')
    plt.title('Scene Detection Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits with some padding
    max_score = max(max(change_scores), scene_threshold, dynamic_threshold) * 1.1
    plt.ylim(0, max_score)
    
    plt.tight_layout()
    
    # To use in a Jupyter notebook:
    plt.show()
    
    # Or to save to file:
    # plt.savefig('scene_detection_analysis.png', dpi=300, bbox_inches='tight')