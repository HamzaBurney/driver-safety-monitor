# driver-safety-monitor
Real-time Driver Monitoring System using AI and Computer Vision. Detects driver drowsiness, distraction, mobile phone usage, and seatbelt compliance using YOLOv8, facial landmarks, and head pose estimation—all integrated into a Flask-based web interface.

## Problem Statement

Driver fatigue, distraction, mobile usage, and neglect of seat belt use contribute significantly to traffic accidents. This project proposes an automated AI-driven monitoring system using computer vision to improve road safety in real-time.

## Features

- **Drowsiness Detection:** Detects closed eyes, yawning, and low blink rates using Eye Aspect Ratio (EAR) and facial landmarks.
- **Distraction Detection:** Estimates head pose (pitch, yaw, roll) to identify when a driver looks away from the road.
- **Mobile Phone Detection:** YOLOv8-based detection of phone use while driving.
- **Seatbelt Detection:** Custom-trained YOLOv8 model to identify seatbelt compliance.
- **Real-Time Feedback Interface:** Web-based interface using Flask to monitor alerts and video stream.
- **System Integration:** Threaded architecture to manage all modules concurrently for optimal performance.

## Dataset

- Seatbelt detection dataset in YOLOv10 format.
- Training: 1117 images | Testing: 164 | Validation: 312
- Download: [Seatbelt Dataset on Roboflow](https://universe.roboflow.com/dti-fzj6g/seat-belt-detection-c3csr/dataset/3)

## Preprocessing Techniques

- YOLO-based automatic processing: resizing, normalization, and data augmentation.
- Custom enhancement: CLAHE (Contrast Limited Adaptive Histogram Equalization) for low-light images.

## How to Run the Project

1. Install Python 3.12.
2. Open the project directory in your terminal.
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
5. (Optional) If using NVIDIA GPU:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
6. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
7. Run the application:
   ```bash
   python app.py
   ```

## Requirements

```
opencv-contrib-python
mediapipe
numpy
ultralytics
flask
```

## Known Limitations

- Dataset is primarily focused on seatbelt detection and may lack diversity.
- Real-time performance may degrade on low-end hardware.
- Occlusions and poor lighting can reduce detection accuracy.

## Future Improvements

- Add audio alerts for detected violations.
- Expand datasets with real-world diversity.
- Optimize models for edge devices.
- Implement driver behavior prediction over time.
- Integrate cloud-based monitoring for fleet management.

---
