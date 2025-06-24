import cv2
import time
from ultralytics import YOLO

class MobileDetector:
    def __init__(self,
                 confidence_threshold=0.2,
                 alert_threshold=2.0,
                 detection_interval=10,
                 iou_threshold=0.6,
                 tracking_fails_threshold=8,
                 max_tracking_duration=1.5):
        
        # Model initialization
        self.model = YOLO('yolov8n.pt')
        self.PHONE_CLASS_ID = 67
        
        # Configuration parameters
        self.CONF_THRESH = confidence_threshold
        self.ALERT_THRESHOLD = alert_threshold
        self.DETECTION_INTERVAL = detection_interval
        self.IOU_THRESHOLD = iou_threshold
        self.TRACKING_FAILS_THRESHOLD = tracking_fails_threshold
        self.MAX_TRACKING_DURATION = max_tracking_duration

        # State variables
        self.tracker = None
        self.tracking_bbox = None
        self.last_detection_time = 0.0
        self.last_tracking_time = 0.0
        self.tracking_fails = 0
        self.phone_present_time = 0.0
        self.current_confidence = 0.0

    def process(self, frame):
        current_time = time.time()
        phone_detected = False
        new_detection = False
        
        # 1. Detection Phase
        if self._should_detect(current_time):
            phone_detected, confidence, bbox = self._detect_phone(frame)
            if phone_detected:
                new_detection = True
                self._update_tracker(frame, bbox)
                self.last_detection_time = current_time
                self.current_confidence = confidence

        # 2. Tracking Phase
        tracked = False
        if self.tracker is not None:
            tracked, bbox = self.tracker.update(frame)
            if tracked:
                self.tracking_bbox = bbox
                self.last_tracking_time = current_time
                self.tracking_fails = 0
            else:
                self.tracking_fails += 1

        # 3. State Validation
        valid_tracking = self._validate_tracking(current_time, tracked)
        time_elapsed = current_time - self.last_detection_time
        
        # 4. Time Accumulation
        if valid_tracking or new_detection:
            time_weight = 1.5 if valid_tracking else 1.0
            confidence_weight = min(1.0, self.current_confidence/self.CONF_THRESH)
            self.phone_present_time += time_weight * confidence_weight * (1/30)  # Assume 30 FPS
        else:
            # Quadratic decay when inactive
            self.phone_present_time = max(0, self.phone_present_time - (time_elapsed**1.5))
        
        self.phone_present_time = min(self.phone_present_time, self.ALERT_THRESHOLD * 1.2)

        # 5. Cleanup
        if not valid_tracking:
            self._reset_tracking()

        return self._get_state(current_time)

    def _should_detect(self, current_time):
        elapsed = current_time - self.last_detection_time
        return (elapsed >= 1/self.DETECTION_INTERVAL or 
                self.tracker is None or 
                not self._validate_tracking(current_time, True))

    def _detect_phone(self, frame):
        results = self.model.predict(
            frame, classes=[self.PHONE_CLASS_ID], 
            conf=self.CONF_THRESH, verbose=False
        )
        
        best_conf = 0.0
        best_box = None
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self.PHONE_CLASS_ID:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        best_box = (xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1])
                        best_conf = conf
        
        return best_box is not None, best_conf, best_box

    def _update_tracker(self, frame, bbox):
        if self._should_update_tracker(bbox):
            self.tracker = cv2.legacy.TrackerCSRT_create()
            self.tracker.init(frame, bbox)
            self.tracking_bbox = bbox
            self.tracking_fails = 0

    def _validate_tracking(self, current_time, tracked):
        if self.tracker is None:
            return False
        if not tracked:
            return False
        time_since_detection = current_time - self.last_detection_time
        return time_since_detection <= self.MAX_TRACKING_DURATION

    def _should_update_tracker(self, new_bbox):
        if self.tracking_bbox is None:
            return True
        return self._iou(new_bbox, self.tracking_bbox) < self.IOU_THRESHOLD

    def _iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi = max(x1, x2)
        yi = max(y1, y2)
        xu = min(x1+w1, x2+w2)
        yu = min(y1+h1, y2+h2)
        
        inter_area = max(0, xu - xi) * max(0, yu - yi)
        union_area = w1*h1 + w2*h2 - inter_area
        return inter_area / (union_area + 1e-6)

    def _reset_tracking(self):
        self.tracker = None
        self.tracking_bbox = None
        self.tracking_fails = 0
        self.current_confidence = 0.0

    def _get_state(self, current_time):
        alert = self.phone_present_time >= self.ALERT_THRESHOLD
        return alert, {
            'bbox': self.tracking_bbox,
            'confidence': self.current_confidence,
            'time_since_detection': self.phone_present_time,
            'alert_triggered': alert,
            'tracking_status': "TRACKING" if self.tracker else "SEARCHING",
            'last_detection_time': current_time - self.last_detection_time
        }

    def get_recent_metrics(self):
        return self._get_state(time.time())[1]