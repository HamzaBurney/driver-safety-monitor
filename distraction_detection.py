import cv2
import mediapipe as mp
import numpy as np
import time

class DistractionDetector:
    def __init__(self, 
                 time_limit=2.0,
                 yaw_left=-40, 
                 yaw_right=35,
                 pitch_up=25,
                 pitch_down=-20):
        # Mediapipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            refine_landmarks=True
        )
        
        # 3D model points and landmark indices
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])
        self.landmark_indices = [1, 152, 263, 33, 287, 57]
        
        # Threshold configuration
        self.yaw_left_threshold = yaw_left
        self.yaw_right_threshold = yaw_right
        self.pitch_up_threshold = pitch_up
        self.pitch_down_threshold = pitch_down
        self.distraction_time_limit = time_limit
        
        # State tracking
        self.distraction_start_time = None
        self.last_angles = (0, 0, 0)

    def process(self, frame):
        """
        Process a frame and return distraction status
        Returns: (is_distracted: bool, metadata: dict)
        """
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        is_distracted = False
        angles = {"pitch": 0, "yaw": 0, "roll": 0}
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # # Extract image points
            # image_points = np.array([(
            #     int(face_landmarks.landmark[idx].x * w),
            #     int(face_landmarks.landmark[idx].y * h)
            #     for idx in self.landmark_indices
            # ], dtype='double')
                                    
            # Get selected landmark points
            image_points = []
            # h, w, _ = img.shape
            for idx in self.landmark_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                image_points.append((x, y))
            image_points = np.array(image_points, dtype='double')

            # Calculate head pose
            camera_matrix = np.array([
                [w, 0, w/2],
                [0, w, h/2],
                [0, 0, 1]
            ], dtype='double')
            
            _, rotation_vector, _ = cv2.solvePnP(
                self.model_points, image_points, 
                camera_matrix, np.zeros((4,1)), 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # Convert rotation vector to angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, np.zeros((3,1))))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            pitch, yaw, roll = [a[0] for a in euler_angles]
            angles = {"pitch": pitch, "yaw": yaw, "roll": roll}
            self.last_angles = (pitch, yaw, roll)

            # Check distraction thresholds
            if (yaw < self.yaw_left_threshold or 
                yaw > self.yaw_right_threshold or 
                pitch < self.pitch_down_threshold or 
                pitch > self.pitch_up_threshold):
                
                if self.distraction_start_time is None:
                    self.distraction_start_time = time.time()
                else:
                    if time.time() - self.distraction_start_time > self.distraction_time_limit:
                        is_distracted = True
            else:
                self.distraction_start_time = None

        return is_distracted, angles

    def get_visualization_data(self):
        """Return angles for display purposes"""
        return {
            'pitch': self.last_angles[0],
            'yaw': self.last_angles[1],
            'roll': self.last_angles[2]
        }

    def release(self):
        """Cleanup resources"""
        self.face_mesh.close()