from ultralytics import YOLO
import time

class SeatbeltDetector:
    def __init__(self, 
                 model_path='best3.pt',
                 conf_threshold=0.3,
                 frame_skip=3):
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.frame_counter = 0
        self.last_bbox = None
        self.last_detection_time = 0
        
        self.recent_metrics = {
            'bbox': None,
            'confidence': 0.0,
            'last_detected': 0.0,
            'alert': True  # Default to alert (no seatbelt)
        }

    def process(self, frame):
        """Returns: (alert_status: bool, metadata: dict)"""
        self.frame_counter += 1
        alert = True  # Default to alert state
        current_time = time.time()
        metadata = {
            'bbox': None,
            'confidence': 0.0,
            'last_detected': self.recent_metrics['last_detected'],
            'alert': True
        }

        if self.frame_counter % self.frame_skip == 0:
            results = self.model.predict(
                source=frame,
                imgsz=640,
                conf=self.conf_threshold,
                verbose=False,
                stream=True
            )

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    alert = False  # Seatbelt detected
                    best_box = boxes[0].astype(int)
                    metadata.update({
                        'bbox': (best_box[0], best_box[1], best_box[2], best_box[3]),
                        'confidence': float(result.boxes.conf[0]),
                        'last_detected': time.time(),
                        'alert': False
                    })

        # Maintain detection for skipped frames
        elif not self.recent_metrics['alert'] and (current_time - self.recent_metrics['last_detected']) < 10:
            metadata = self.recent_metrics.copy()
            alert = False
            
        # Update alert based on 10-second threshold
        if (current_time - metadata['last_detected']) >= 10:
            alert = True

        self.recent_metrics = metadata.copy()
        self.recent_metrics['alert'] = alert
        
        return alert, self.recent_metrics
    
    def get_recent_metrics(self):
        """Returns the latest detection metrics"""
        return self.recent_metrics

    def release(self):
        pass  # No explicit cleanup needed