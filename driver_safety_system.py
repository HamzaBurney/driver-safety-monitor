import cv2
import time
from threading import Thread, Lock
from collections import deque

# Import detector modules
from distraction_detection import DistractionDetector
from drowsiness_detection import DrowsinessDetector
from mobile_detection import MobileDetector
from seatbelt_detection import SeatbeltDetector

class DriverSafetySystem:
    def __init__(self, camera_source=0, frame_width=640, frame_height=480, web_mode = False):
        self.camera = cv2.VideoCapture(camera_source)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.running = False
        self.frame_lock = Lock()
        self.latest_frame = None
        self.web_mode = web_mode
        self.web_frame = None

        # Initialize detection modules
        self.distraction_detector = DistractionDetector()
        self.drowsiness_detector = DrowsinessDetector(
            ear_threshold=0.25,  # Match your original pixel distance threshold
            yawn_threshold=25,
            per_minute_blink_threshold=30
        )
        self.mobile_detector = MobileDetector(
            confidence_threshold=0.2,
            alert_threshold=2.0,
            detection_interval=10
        )
        self.seatbelt_detector = SeatbeltDetector(
            model_path='best3.pt',
            conf_threshold=0.3,
            frame_skip=3
        )

        # Alert system configuration
        self.active_alerts = {}  # Format: {alert_type: (message, start_time)}
        self.alert_persistence = 1.5  # Seconds to show alert after resolution
        self.alert_colors = {
            'distraction': (0, 0, 255),
            'drowsiness': (255, 0, 0),
            'mobile': (0, 255, 255),
            'seatbelt': (255, 0, 255)
        }

        # Performance tracking
        self._frame_times = deque(maxlen=10)
        
        print('Driver Safety System initialized.')

    def start(self):
        self.running = True
        Thread(target=self._capture_frames, daemon=True).start()
        Thread(target=self._process_frames, daemon=True).start()
        # self._main_loop()

    # Inside DriverSafetySystem
    def get_latest_frame(self):
        
        if self.web_frame is not None:
            _, jpeg = cv2.imencode('.png', self.web_frame)
            return jpeg.tobytes()
        return None


    def _capture_frames(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            with self.frame_lock:
                self.latest_frame = frame.copy()

    def _process_frames(self):
        while self.running:
            if self.latest_frame is not None:
                with self.frame_lock:
                    process_frame = self.latest_frame.copy()

                # Process all detectors
                results = {
                    'distraction': self.distraction_detector.process(process_frame)[0],
                }
                
                drowsy, _ = self.drowsiness_detector.process(process_frame)
                results['drowsiness'] = drowsy

                mobile_alert, _ = self.mobile_detector.process(process_frame)
                results['mobile'] = mobile_alert
                
                seatbelt_alert, _ = self.seatbelt_detector.process(process_frame)
                results['seatbelt'] = seatbelt_alert

                # Update alert states and UI
                self._update_alert_states(results)
                self.web_frame = self._update_ui(process_frame)

                # Track FPS
                self._frame_times.append(time.time())

            time.sleep(0.05)

    def _get_alert_message(self, alert_type):
        messages = {
            'distraction': "DISTRACTION ALERT! Face away from road",
            'drowsiness': "DROWSINESS DETECTED!",
            'mobile': "PHONE USE DETECTED!",
            'seatbelt': "SEATBELT NOT FASTENED!"
        }
        return messages.get(alert_type, "Unknown alert")

    def _update_alert_states(self, results):
        current_time = time.time()

        # Update existing or add new alerts for all detection types
        for alert_type in results:
            if results[alert_type]:
                # Get human-readable message
                message = self._get_alert_message(alert_type)
                # Update alert with current timestamp
                self.active_alerts[alert_type] = (message, current_time)

        # Remove resolved alerts that have passed persistence time
        to_remove = []
        for alert_type, (_, start_time) in self.active_alerts.items():
            # Check if either:
            # 1. The alert condition is no longer active
            # 2. The alert has been showing past its persistence time
            if not results.get(alert_type, False) and \
               (current_time - start_time > self.alert_persistence):
                to_remove.append(alert_type)

        # Clean up expired alerts
        for alert_type in to_remove:
            del self.active_alerts[alert_type]

    def _update_ui(self, frame):
        # Display active alerts with fade effect
        alert_y = 50
        current_time = time.time()

        # Show priority alerts first
        priority_order = ['drowsiness', 'distraction', 'mobile', 'seatbelt']
        for alert_type in priority_order:
            if alert_type in self.active_alerts:
                message, start_time = self.active_alerts[alert_type]
                time_active = current_time - start_time

                # Skip if alert should be expired
                if time_active > self.alert_persistence:
                    continue
                
                if alert_type == 'drowsiness':
                    drowsy_meta = self.drowsiness_detector.get_recent_metrics()
                    if drowsy_meta['is_drowsy'] != True:
                        # Skip drawing if in warmup period
                        continue

                # Calculate fade effect
                fade_progress = min(time_active / self.alert_persistence, 1.0)
                base_color = self.alert_colors.get(alert_type, (0, 0, 255))
                fade_color = tuple(int(c * (1 - fade_progress*0.5)) for c in base_color)

                # Draw alert text
                cv2.putText(frame, message, (20, alert_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, fade_color, 2)
                alert_y += 40

        # Display drowsiness metrics (bottom-left)
        drowsy_meta = self.drowsiness_detector.get_recent_metrics()
        cv2.putText(frame, f"Blinks/min: {drowsy_meta['blink_rate']:.1f}", 
                    (20, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"Eyes: {drowsy_meta['eye_state'].upper()}", 
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if drowsy_meta['eye_state'] == 'open' else (0, 0, 255), 
                    1)
        
        mobile_meta = self.mobile_detector.get_recent_metrics()
        if mobile_meta['bbox'] is not None:
            # Convert all coordinates to integers
            x, y, w, h = mobile_meta['bbox']
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            
            # Now use these integer coordinates
            status_color = (0, 255, 0) if mobile_meta['tracking_status'] == "TRACKING" else (0, 165, 255)
            
            # Draw semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), status_color, -1)
            alpha = 0.2  # 20% opacity
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw solid border
            cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
            
            # Draw status text
            text = f"{mobile_meta['tracking_status']} ({mobile_meta['confidence']:.0%})"
            cv2.putText(frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Display mouth state (right of eye info)
        cv2.putText(frame, f"Mouth: {drowsy_meta['mouth_state'].upper()}", 
                    (250, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if drowsy_meta['mouth_state'] == 'closed' else (0, 0, 255),
                    1)
        
        seatbelt_meta = self.seatbelt_detector.get_recent_metrics()
        if seatbelt_meta['bbox'] is not None:
            x1, y1, x2, y2 = seatbelt_meta['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.alert_colors['seatbelt'], 2)
            cv2.putText(frame, f"Seatbelt: {seatbelt_meta['confidence']:.0%}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.alert_colors['seatbelt'], 2)

        # System info (top-right)
        cv2.putText(frame, f"FPS: {self._get_fps():.1f}", 
                    (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if not self.web_mode:
            cv2.imshow("Driver Safety Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
        
        return frame

    def _get_fps(self):
        if len(self._frame_times) < 2:
            return 0
        return (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])

    def _main_loop(self):
        while self.running:
            time.sleep(0.1)

    def stop(self):
        self.running = False
        self.camera.release()
        cv2.destroyAllWindows()
        self.distraction_detector.release()

# if __name__ == "__main__":
#     system = DriverSafetySystem()
#     try:
#         system.start()
#     except KeyboardInterrupt:
#         system.stop()