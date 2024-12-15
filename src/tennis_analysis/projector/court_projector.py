"""
Module for projecting tennis court positions onto a minimap overlay.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Any
import pandas as pd
from pathlib import Path
import logging
import math

class CourtProjector:
    """Projects player positions onto a miniature court visualization."""
    
    def __init__(
        self,
        minimap_size: Tuple[int, int] = (320, 280), #Half before(320, 160)
        position: Tuple[int, int] = (1500, 20), #(420, 420) before
        player_colors: Tuple[Tuple[int, int, int], ...] = ((255, 0, 0), (0, 0, 255))
    ):
        self.minimap_size = minimap_size
        self.position = position
        self.player_colors = player_colors

        ###
        
        
        # Load court image
        current_dir = Path(__file__).parent  # Gets projector directory
        court_path = current_dir / 'Static' / 'courts_tennis_court_1.png'

        #print(f"Looking for court image at: {court_path}")
        #print(f"Path exists: {court_path.exists()}")
        
        # Load and process the court image
        self.court_base = cv2.imread(str(court_path))
        if self.court_base is None:
            raise FileNotFoundError(f"Court minimap image not found at {court_path}")
        
        ###Rescale the diagram corners
        original_height, original_width = self.court_base.shape[:2]
        diagram_coords = [(215, 52), (385, 52), (215, 548), (385, 548)]

        # Calculate scaling factors
        scale_x = minimap_size[0] / original_width
        scale_y = minimap_size[1] / original_height

        # Scale coordinates
        self.scaled_coords = []
        for x, y in diagram_coords:
            new_x = int(x * scale_x)
            new_y = int(y * scale_y)
            self.scaled_coords.append((new_x, new_y))

        # Resize court image to desired minimap size
        self.court_base = cv2.resize(self.court_base, minimap_size)
        #########
        #Print
        #minimap_example = self.court_base.copy()
        #for point in self.scaled_coords:
        #    cv2.circle(minimap_example, point, 5, (0,0,255), -1)
        #cv2.imwrite("scaled_diagram_coords.png", minimap_example)

        #################

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def create_court_base(self) -> np.ndarray:
        """Return a copy of the base court image."""
        return self.court_base.copy()
    
    ########################################################################################
    # Detect points in court
    def detect_court_edges(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect white lines/edges in a tennis court frame
        
        Args:
            frame: Input video frame
        Returns:
            tuple: (original frame, masked output showing detected edges)
        """
        # Define the white color boundaries for tennis court lines
        lower_white = np.array([180, 180, 100], dtype="uint8")
        upper_white = np.array([255, 255, 255], dtype="uint8")

        # Create mask for white colors
        mask = cv2.inRange(frame, lower_white, upper_white)
        
        # Apply mask to get only the court lines
        white_edges = cv2.bitwise_and(frame, frame, mask=mask)
        
        return frame, white_edges
    
    def process_court_corners(self, white_edges: np.ndarray, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Process the detected edges to find and mark court corners
        
        Args:
            white_edges: Output from detect_court_edges showing white lines
            frame: Original frame to mark corners on
        Returns:
            tuple: (processed frame with marked corners, corner coordinates)
        """
        frame_copy = frame.copy()
        # Convert to grayscale
        gray = cv2.cvtColor(white_edges, cv2.COLOR_BGR2GRAY)
        
        # Detect corners using Harris
        corners = cv2.cornerHarris(gray, blockSize=9, ksize=3, k=0.01)
        
        # Normalize corners to 0-255 range
        corners_normalized = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(corners_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilate the thresholded image
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store corner coordinates
        self.corner_coords = []
        
        # Draw circles at corner points
        for contour in contours:
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw red circle at corner
                cv2.circle(frame_copy, (cx, cy), 3, (0, 0, 255), -1)
                self.corner_coords.append((cx, cy))
        
        return frame_copy, self.corner_coords
        

    #############################################################################################
        
    ##############################
    #Helpers

    def distance(self, x: Tuple[float, float], y: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two individual points.
        
        Args:
            x: First point coordinates (x, y)
            y: Second point coordinates (x, y)
        Returns:
            float: Euclidean distance between the points
        """
        x = np.array(x)  # Convert single point to numpy array
        y = np.array(y)  # Convert single point to numpy array
        d = x - y        # Calculate difference vector
        dist = np.linalg.norm(d)  # Calculate Euclidean norm (distance)
        return dist

    def nearest_point(self, point: Tuple[float, float], coordinates: List[Tuple[float, float]]) -> int:
        """
        Find index of the nearest point in next frame by comparing distances iteratively.
        
        Args:
            point: Current point coordinates (x, y)
            coordinates: List of all points in next frame
        Returns:
            int: Index of nearest point
        """
        min_dist = math.inf
        idx = -1
        
        # Iterate through each point in the next frame
        for i, cord in enumerate(coordinates):
            dist = self.distance(point, cord)  # Compare current point to each point in next frame
            if dist < min_dist:
                min_dist = dist
                idx = i
                
        return idx
    
    ###############################
    #Homography
    def calculate_homography(self, source_points, destination_points = None):
        """
        Calculate homography matrix between source and destination points.

        Args:
            source_points: List of 4 points in the source image.
            destination_points: List of 4 points in the destination image.

        Returns:
            numpy.ndarray: Homography matrix.
        """
        if destination_points is None:
            destination_points = self.scaled_coords

        if len(source_points) != 4 or len(destination_points) != 4:
            #return np.identity(3)
            return None

        # Convert points to numpy arrays
        source_points = np.float32(source_points)
        destination_points = np.float32(destination_points)

        # Calculate homography matrix using RANSAC for robustness
        H, status = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
    
        return H
    

    ###############################################################
    #Uses Homography and and projects the point
    
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

        def Projection_point(point, H):
            """Transform a 2D point using the homography matrix."""
            p = np.array([[point[0]], [point[1]], [1]])  # Convert point to homogeneous coordinates
            p_transformed = H @ p  # Apply the homography
            
            if abs(p_transformed[2][0]) < 1e-6: 
                return (0, 0) 
            
            x = int(p_transformed[0][0] / p_transformed[2][0])  # Normalize to get the 2D coordinates
            y = int(p_transformed[1][0] / p_transformed[2][0])
            return (x, y)
        
        if len(player_boxes) == 2:
            for box in player_boxes:
                # Box contains (x, y, w, h, prediction, player)
                x = box[0] + box[2] // 2  # Get the x-coordinate of the bottom center
                y = box[1] + box[3]  # Get the y-coordinate of the bottom center (y + height)

                player_class = box[5]
                if player_class == 0:
                    x_offset = -50
                    y_offset = -70  
                    color = (0, 0, 255)
                else:  # Bottom player (closer)
                    x_offset = -50
                    y_offset = -150
                    color = (0, 255, 0)

                # Apply homography transformation if available
                if homography_matrix is not None:
                    transformed_point = Projection_point((x + x_offset, y + y_offset), homography_matrix)
                    court_x, court_y = transformed_point
                else:
                    # Simple scaling if no homography matrix is provided
                    court_x = int(x * self.scale_x)
                    court_y = int(y * self.scale_y)

                # Draw player marker on the minimap
                cv2.circle(court, (court_x, court_y), 4, color, -1)  # Draw filled circle

        return court
    ########################################


    ###########################
    #Add mini map already checked
    def add_minimap(
        self,
        frame: np.ndarray,
        player_boxes: Any,
        homography_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
    Add court minimap to video frame with transparency effect and border.
    
    Args:
        frame: Video frame to modify
        player_boxes: DataFrame with player detections
        homography_matrix: Optional perspective transformation matrix
        
    Returns:
        Frame with minimap overlay
    """
        # Create minimap with player positions
        minimap = self.project_positions(player_boxes, homography_matrix)
        # Get position for overlay

        #self.scaled_coords = [(215, 52), (385, 52), (215, 548), (385, 548)]
        for point in self.scaled_coords:
            cv2.circle(minimap, point, 5,(0, 0, 255), -1)

        x, y = self.position
        h, w = minimap.shape[:2]

        try:
            # Create slightly larger background for border
            border = 2
            bg = np.full((h + 2*border, w + 2*border, 3), (0, 0, 0), dtype=np.uint8)
        
            # Add transparency to minimap
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
            # Place minimap in background with border
            bg[border:-border, border:-border] = minimap
        
            # Create ROI and apply transparency
            frame_region = frame[y:y+h+2*border, x:x+w+2*border]
            #print(f"Frame region shape: {frame_region.shape}")
            alpha = 0.98  # You can adjust this value for different transparency levels
        
            frame[y:y+h+2*border, x:x+w+2*border] = cv2.addWeighted(
                frame_region, 1-alpha,
                bg, alpha,
                0
            )
        
        except Exception as e:
            print(f"Detailed error in add_minimap: {str(e)}")
            self.logger.error(f"Error adding minimap overlay: {str(e)}")
            return frame  # Return original frame if overlay fails
    
        return frame