import cv2
import mediapipe as mp
import time
import numpy as np

class DrowsinessDetector:
    def __init__(self,
                 ear_threshold=0.25,
                 min_eye_frames=3,
                 yawn_threshold=25,
                 min_yawn_frames=15,
                 yawn_duration=3.5,
                 per_minute_blink_threshold=30):
        
        # Mediapipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Threshold configuration
        self.EAR_THRESHOLD = ear_threshold        # Proper EAR threshold (0.2-0.25)
        self.MIN_EYE_CLOSED_FRAMES = min_eye_frames
        self.YAWN_THRESHOLD = yawn_threshold      # Pixel distance threshold
        self.MIN_YAWN_FRAMES = min_yawn_frames
        self.YAWN_DURATION = yawn_duration
        self.PER_MINUTE_BLINK_THRESHOLD = per_minute_blink_threshold

        self.warmup_duration = 10   # Seconds to wait before calculating blink rate
        self.start_time = time.time()
        
        # State tracking
        self.eye_closed_frames = 0
        self.mouth_open_frames = 0
        self.blink_count = 0
        self.start_minute = time.time()
        self.last_yawn_time = 0
        self.last_blink_time = None
        self.is_drowsy = False
        self.eye_state = "open"
        self.mouth_state = "closed"
        self.blink_rate = 0

    def process(self, frame):
        """Process frame and return drowsiness status"""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.is_drowsy = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            # Analyze eyes using proper EAR calculation
            self._analyze_eyes(face_landmarks, h, w)
            
            # Analyze yawns
            self._analyze_yawn(face_landmarks, h)
            
            # Update blink rate metrics
            self._update_blink_rate()
            
            # Check for drowsiness conditions
            self._check_drowsiness()

        return self.is_drowsy, self.get_recent_metrics()

    def _analyze_eyes(self, landmarks, img_height, img_width):
        # Eye landmarks indices (Mediapipe format)
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        right_eye_indices = [33, 160, 158, 133, 153, 144]

        def calculate_ear(eye_indices):
            # Calculate proper Eye Aspect Ratio (EAR)
            p1 = landmarks.landmark[eye_indices[0]]  # Left corner
            p2 = landmarks.landmark[eye_indices[1]]  # Top
            p3 = landmarks.landmark[eye_indices[2]]  # Right corner
            p4 = landmarks.landmark[eye_indices[3]]  # Bottom
            p5 = landmarks.landmark[eye_indices[4]]  # Inner top
            p6 = landmarks.landmark[eye_indices[5]]  # Inner bottom

            # Vertical distances
            v1 = np.linalg.norm([p2.x - p6.x, p2.y - p6.y])
            v2 = np.linalg.norm([p3.x - p5.x, p3.y - p5.y])
            
            # Horizontal distance
            h_dist = np.linalg.norm([p1.x - p4.x, p1.y - p4.y])
            
            return (v1 + v2) / (2 * h_dist)

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye_indices)
        right_ear = calculate_ear(right_eye_indices)
        avg_ear = (left_ear + right_ear) / 2

        # Update eye state
        if avg_ear < self.EAR_THRESHOLD:
            self.eye_closed_frames += 1
            if self.eye_closed_frames > self.MIN_EYE_CLOSED_FRAMES:
                self.eye_state = "closed"
        else:
            if self.eye_closed_frames > self.MIN_EYE_CLOSED_FRAMES:
                self._register_blink()
            self.eye_closed_frames = 0
            self.eye_state = "open"

    def _analyze_yawn(self, landmarks, img_height):
        # Mouth landmarks indices
        top_lip = 13
        bottom_lip = 14
        
        # Calculate vertical mouth distance
        vertical_dist = abs(landmarks.landmark[top_lip].y - 
                       landmarks.landmark[bottom_lip].y) * img_height

        if vertical_dist > self.YAWN_THRESHOLD:
            self.mouth_open_frames += 1
            if self.mouth_open_frames > self.MIN_YAWN_FRAMES:
                self.mouth_state = "yawn"
                self.last_yawn_time = time.time()
        else:
            self.mouth_open_frames = 0
            self.mouth_state = "closed"

    def _register_blink(self):
        self.blink_count += 1
        self.last_blink_time = time.time()

    def _update_blink_rate(self):
        elapsed_time = time.time() - self.start_minute
        if elapsed_time > 0:
            self.blink_rate = (self.blink_count / elapsed_time) * 60  # Blinks per minute
        
        # Reset counter every minute
        if elapsed_time > 60:
            self.blink_count = 0
            self.start_minute = time.time()

    def _check_drowsiness(self):
        # Check multiple drowsiness indicators
        self.is_drowsy = False
        elapsed_since_start = time.time() - self.start_time
        
        # Skip checks during warmup period
        if elapsed_since_start < self.warmup_duration:
            return False
            
        # 1. Excessive blinking (now using direct 60-second window count)
        if self.blink_rate > self.PER_MINUTE_BLINK_THRESHOLD:
            self.is_drowsy = True
            
        # 2. Prolonged eye closure
        if self.last_blink_time is not None:
            if self.eye_state == "closed" and \
               (time.time() - self.last_blink_time) > 3.0:
                self.is_drowsy = True
                print("Prolonged eye closure detected.", time.time() - self.last_blink_time)
            
        # 3. Recent yawning
        if (time.time() - self.last_yawn_time) < self.YAWN_DURATION:
            self.is_drowsy = True

    def get_recent_metrics(self):
        
        elapsed_since_start = time.time() - self.start_time
        warmup_remaining = max(0, self.warmup_duration - elapsed_since_start)
        
        return {
            'eye_state': self.eye_state,
            'mouth_state': self.mouth_state,
            'blink_rate': round(self.blink_rate, 1), 
            'warmup_remaining': warmup_remaining,
            'is_drowsy': self.is_drowsy,
        }

    def release(self):
        self.face_mesh.close()