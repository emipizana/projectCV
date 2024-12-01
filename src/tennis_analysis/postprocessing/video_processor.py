"""
Módulo para el procesamiento de video con tracking.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from ultralytics import YOLO

from ..tracking.trackers import BallTracker, PlayerTracker
from .visualizer import TrackingVisualizer

class VideoProcessor:
    """
    Procesa videos de tenis aplicando tracking y visualización.
    """
    
    def __init__(
        self,
        model_players_path: str,
        model_ball_path: str,
        device: str = 'cuda',
        conf_threshold: Dict[str, float] = None
    ):
        """
        Args:
            model_players_path: Ruta al modelo YOLO de jugadores
            model_ball_path: Ruta al modelo YOLO de pelota
            device: Dispositivo para inferencia ('cuda', 'cpu', 'mps')
            conf_threshold: Umbrales de confianza {'ball': 0.3, 'players': 0.5}
        """
        self.conf_threshold = conf_threshold or {'ball': 0.3, 'players': 0.5}
        
        # Cargar modelos
        self.model_players = YOLO(model_players_path)
        self.model_ball = YOLO(model_ball_path)
        
        # Mover modelos al dispositivo especificado
        self.model_players.to(device)
        self.model_ball.to(device)
        
        self.device = device
        self.classes_players = self.model_players.names
    
    def process_video(
        self,
        video_path: str,
        output_path: str,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Procesa un video completo aplicando tracking y visualización.
        
        Args:
            video_path: Ruta al video de entrada
            output_path: Ruta para el video procesado
            show_progress: Si se debe mostrar la barra de progreso
            
        Returns:
            Diccionario con estadísticas del procesamiento
        """
        # Inicializar trackers
        ball_tracker = BallTracker()
        player_tracker = PlayerTracker()
        
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        
        # Obtener propiedades del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configurar writer
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Inicializar visualizador
        visualizer = TrackingVisualizer(width, height)
        
        # Procesar frames
        frame_count = 0
        stats = {
            'total_frames': total_frames,
            'ball_detections': 0,
            'ball_predictions': 0,
            'player_detections': 0
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Procesar jugadores
            results_players = self._process_players(frame, player_tracker, frame_count)
            stats['player_detections'] += len(results_players)
            
            # Procesar pelota
            ball_detected = self._process_ball(frame, ball_tracker, frame_count)
            if ball_detected:
                stats['ball_detections'] += 1
            elif ball_tracker.last_predicted is not None:
                stats['ball_predictions'] += 1
            
            # Generar visualización
            frame_vis = visualizer.draw_tracking(
                frame.copy(),
                ball_tracker,
                player_tracker,
                frame_count
            )
            
            # Escribir frame
            out.write(frame_vis)
            
            # Actualizar progreso
            frame_count += 1
            if show_progress and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f'Procesado: {progress:.1f}% ({frame_count}/{total_frames} frames)', 
                      end='\r')
        
        # Liberar recursos
        cap.release()
        out.release()
        
        # if show_progress:
        #     print('\nProcesamiento completado!')
            
        return stats
    
    def _process_players(
        self,
        frame: np.ndarray,
        player_tracker: PlayerTracker,
        frame_count: int
    ) -> pd.DataFrame:
        """
        Procesa un frame para detectar y trackear jugadores.
        """
        # Convertir a escala de grises para mejor detección
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_3ch = np.stack((frame_gray,) * 3, axis=-1)
        
        # Detectar jugadores
        results = self.model_players(
            frame_gray_3ch,
            conf=self.conf_threshold['players'],
            device=self.device,
            verbose=False
        )
        
        # Procesar resultados
        boxes_df = pd.DataFrame()
        if len(results) > 0:
            r = results[0]
            if len(r.boxes) > 0:
                boxes = r.boxes
                boxes_data = boxes.xywh.cpu().numpy()
                boxes_df = pd.DataFrame(boxes_data, columns=['x', 'y', 'w', 'h'])
                boxes_df['conf'] = boxes.conf.cpu().numpy()
                boxes_df['class'] = boxes.cls.cpu().numpy()
        
        # Actualizar tracker
        player_tracker.update(boxes_df, frame_count, frame_count/30.0)  # Assuming 30fps
        
        return boxes_df
    
    def _process_ball(
        self,
        frame: np.ndarray,
        ball_tracker: BallTracker,
        frame_count: int
    ) -> bool:
        """
        Procesa un frame para detectar y trackear la pelota.
        
        Returns:
            bool indicando si se detectó la pelota
        """
        results = self.model_ball(
            frame,
            conf=self.conf_threshold['ball'],
            device=self.device,
            verbose=False
        )
        
        ball_detected = False
        if len(results) > 0:
            r = results[0]
            boxes = r.boxes
            
            if len(boxes) > 0:
                # Tomar la detección con mayor confianza
                best_box = boxes[boxes.conf.argmax()]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                # Calcular centro
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Actualizar tracker
                ball_tracker.add_detection(np.array([center_x, center_y]), frame_count)
                ball_detected = True
        
        # Si no se detectó, intentar predecir
        if not ball_detected:
            if ball_tracker.frames_without_detection < ball_tracker.max_frames_to_predict:
                predicted_pos = ball_tracker.predict_position(frame_count)
                if predicted_pos is not None:
                    ball_tracker.frames_without_detection += 1
                    ball_tracker.last_predicted = predicted_pos
        
        return ball_detected

    def process_frame(
        self,
        frame: np.ndarray,
        ball_tracker: BallTracker,
        player_tracker: PlayerTracker,
        frame_count: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Procesa un único frame para tracking en tiempo real.
        
        Args:
            frame: Frame a procesar
            ball_tracker: Instancia de BallTracker
            player_tracker: Instancia de PlayerTracker
            frame_count: Número de frame actual
            
        Returns:
            Tupla (frame procesado, estadísticas)
        """
        # Procesar detecciones
        results_players = self._process_players(frame, player_tracker, frame_count)
        ball_detected = self._process_ball(frame, ball_tracker, frame_count)
        
        # Visualizar resultados
        visualizer = TrackingVisualizer(frame.shape[1], frame.shape[0])
        frame_vis = visualizer.draw_tracking(
            frame.copy(),
            ball_tracker,
            player_tracker,
            frame_count
        )
        
        # Recopilar estadísticas
        stats = {
            'ball_detected': ball_detected,
            'ball_predicted': ball_tracker.last_predicted is not None,
            'players_detected': len(results_players)
        }
        
        return frame_vis, stats