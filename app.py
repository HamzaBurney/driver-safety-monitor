from flask import Flask, render_template, Response, jsonify
from driver_safety_system import DriverSafetySystem  # import your class
import threading
import time

app = Flask(__name__)
safety_system = DriverSafetySystem(web_mode=True)
# system_thread = None
is_streaming = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_stream')
def toggle_stream():
    global is_streaming
    if is_streaming:
        is_streaming = False
        safety_system.stop()
    else:
        is_streaming = True
        system_thread = threading.Thread(target=safety_system.start)
        system_thread.start()
    return ('', 204)

@app.route('/video_feed')
def video_feed():
    def generate():
        while safety_system.running:
            frame = safety_system.get_latest_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
