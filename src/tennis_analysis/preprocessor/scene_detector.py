"""
Módulo para detectar cambios de escena en videos de tenis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Callable
from .video_loader import VideoLoader
from typing import Dict
import matplotlib.pyplot as plt

class SceneDetector:
    """Detector de cambios de escena en videos."""
    
    def __init__(
        self,
        video_loader: VideoLoader,
        scene_threshold: float = 10.0,
        resize_width: int = 320,
        resize_height: int = 180,
        fps: int = 30,  # Frames por segundo del video
        min_scene_duration: int = 4*30
    ):
        self.loader = video_loader
        self.scene_threshold = scene_threshold
        self.resize_dims = (resize_width, resize_height)
        self.fps = fps
        self.min_scene_duration = min_scene_duration
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
        Almacena tanto puntos como no-puntos que cumplan la duración mínima.
        
        Returns:
            Tuple containing:
            - List of scene change frames
            - List of change scores
        """
        scene_changes = []
        change_scores = []
        self.detected_points = []
        
        # Determinar frame final
        if end_frame is None:
            end_frame = self.loader.frame_count
            
        total_frames = end_frame - start_frame
        UPDATE_INTERVAL = max(1, total_frames // 1000)
        
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
        
        # Variables para detección de escenas
        current_scene_start = start_frame
        temp_scene_start = None  # Para almacenar temporalmente posibles cambios de escena
        current_scene_type = 'low'  # 'low' para movimiento bajo, 'high' para alto
        scene_diffs = []  # Lista de diferencias en la escena actual
        temp_scene_type = None  # Para almacenar temporalmente el tipo de la posible nueva escena
        frames_in_temp_scene = 0  # Contador de frames en la escena temporal
        
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
            scene_diffs.append(diff_score)
            
            # Determinar si hay un cambio significativo en el tipo de escena
            current_mean_diff = np.mean(scene_diffs[-min(15, len(scene_diffs)):])  # Media móvil de 15 frames
            new_scene_type = 'high' if current_mean_diff > self.scene_threshold else 'low'
            
            # Manejar cambios de escena
            if new_scene_type != current_scene_type:
                if temp_scene_start is None:
                    # Iniciar una posible nueva escena
                    temp_scene_start = current_frame
                    temp_scene_type = new_scene_type
                    frames_in_temp_scene = 1
                elif new_scene_type != temp_scene_type:
                    # Reset del contador temporal si el tipo cambia
                    temp_scene_start = current_frame
                    temp_scene_type = new_scene_type
                    frames_in_temp_scene = 1
                else:
                    # Incrementar contador de frames en la escena temporal
                    frames_in_temp_scene += 1
                    
                    # Si la escena temporal supera la duración mínima, confirmar el cambio
                    if frames_in_temp_scene >= self.min_scene_duration:
                        scene_duration = temp_scene_start - current_scene_start
                        
                        # Registrar la escena anterior
                        self.detected_points.append({
                            'start_frame': current_scene_start,
                            'end_frame': temp_scene_start,
                            'duration': scene_duration,
                            'is_point': current_scene_type == 'low'
                        })
                        scene_changes.append(current_scene_start)
                        
                        # Iniciar nueva escena
                        current_scene_start = temp_scene_start
                        current_scene_type = temp_scene_type
                        scene_diffs = scene_diffs[-frames_in_temp_scene:]
                        temp_scene_start = None
                        frames_in_temp_scene = 0
            else:
                # Resetear variables temporales si volvemos al tipo original
                temp_scene_start = None
                frames_in_temp_scene = 0
            
            # Actualizar frame previo
            if example_frame is None:
                small_prev = small_frame.copy()
            
            # Reportar progreso
            if (current_frame - start_frame) % UPDATE_INTERVAL == 0 and progress_callback:
                progress_callback(current_frame - start_frame, total_frames)
            
            current_frame += 1
        
        # Procesar última escena
        if (current_frame - current_scene_start) >= self.min_scene_duration:
            self.detected_points.append({
                'start_frame': current_scene_start,
                'end_frame': current_frame,
                'duration': current_frame - current_scene_start,
                'is_point': current_scene_type == 'low'
            })
            scene_changes.append(current_scene_start)
        
        # Asegurar que reportamos 100% al final
        if progress_callback:
            progress_callback(total_frames, total_frames)
            
        # plot_scene_detection(change_scores, scene_changes, self.scene_threshold)
                
        return scene_changes, change_scores
    
    def get_detected_points(self) -> List[Dict[str, int]]:
        """
        Returns the list of detected scenes (both points and non-points).
        
        Returns:
            List of dictionaries containing:
            - start_frame: Frame where scene starts
            - end_frame: Frame where scene ends
            - duration: Duration in frames
            - is_point: Boolean indicating if the scene is a tennis point
        """
        return self.detected_points

def plot_scene_detection(change_scores, scene_changes, scene_threshold):
    """
    Plot scene detection analysis including:
    - Frame differences over time
    - Scene threshold
    - Detected scene changes
    """
    plt.figure(figsize=(15, 8))
    
    # Plot frame differences
    frames = np.arange(len(change_scores))
    plt.plot(frames, change_scores, 'b-', alpha=0.5, label='Frame Differences')
    
    # Plot threshold as horizontal line
    plt.axhline(y=scene_threshold, color='r', linestyle='--', label='Scene Threshold')
    
    # Plot scene changes as vertical lines
    for change in scene_changes:
        plt.axvline(x=change, color='red', alpha=0.3)
        plt.plot(change, change_scores[change] if change < len(change_scores) else 0, 
                'ro', markersize=8)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Difference Score')
    plt.title('Scene Detection Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits with some padding
    max_score = max(max(change_scores), scene_threshold) * 1.1
    plt.ylim(0, max_score)
    
    plt.tight_layout()
    plt.show()