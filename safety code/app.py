import cv2
import os
import threading
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, Response, request, redirect, url_for
import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# Global state
class AppState:
    paused = False
    source = 'camera'  # 'camera' or 'video'
    video_name = 'Camera 0'
    threshold = 7
    inner_count = 0
    outer_count = 0
    plates = [
        {"id": "P1", "inner_ok": False, "outer_ok": False},
        {"id": "P2", "inner_ok": False, "outer_ok": False},
        {"id": "P3", "inner_ok": False, "outer_ok": False},
    ]
    plate_frames = [{'inner': 0, 'outer': 0} for _ in range(3)]
    model_loaded = False
    model = None
    new_source_path = None
    trigger_restart = False
    empty_frames = 0
    # IP Camera credentials
    cam_ip = ''
    cam_port = '554'
    cam_user = ''
    cam_pass = ''
    cam_stream = '/cam/realmonitor?channel=1&subtype=0'
    cam_connected = False

state = AppState()

def load_model():
    model_path = os.path.join('model', 'best.pt')
    if os.path.exists(model_path) and YOLO is not None:
        try:
            state.model = YOLO(model_path)
            state.model_loaded = True
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model best.pt not found or Ultralytics not installed. Waiting for upload...")

load_model()
cap = None

import time

class CameraReader:
    """Dedicated background thread for reading RTSP streams instantly without buffering lag."""
    def __init__(self, source):
        if isinstance(source, str) and source.startswith('rtsp://'):
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None
            
    def release(self):
        self.running = False
        self.cap.release()
        
    def get(self, prop):
        return self.cap.get(prop)
        
    def set(self, prop, val):
        self.cap.set(prop, val)

class VideoProcessor:
    def __init__(self):
        self.jpeg_bytes = b''
        self.update_count = 0
        self.running = True
        
        self.latest_frame_for_yolo = None
        self._last_boxes = None
        self._last_classes = {}
        
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

    def inference_loop(self):
        global state
        while self.running:
            frame = self.latest_frame_for_yolo
            if frame is not None:
                self.latest_frame_for_yolo = None   # consume immediately
                if getattr(state, 'model_loaded', False) and state.model:
                    results = state.model(frame, conf=state.threshold / 10.0, verbose=False)
                    self._last_boxes  = results[0].boxes
                    self._last_classes = results[0].names
            else:
                time.sleep(0.003)  # only sleep when idle, not between frames

    @staticmethod
    def _open_cap(source):
        """Open a VideoCapture with RTSP-aware settings."""
        if isinstance(source, str) and source.startswith('rtsp://'):
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimal RTSP buffer
        else:
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def _set_placeholder(self, msg):
        """Encode and store a dark placeholder frame with a centred message."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (25, 30, 40)          # dark blue-grey background
        font = cv2.FONT_HERSHEY_SIMPLEX
        icon = "[ No Camera Connected ]"
        (iw, _), _ = cv2.getTextSize(icon, font, 0.7, 2)
        cv2.putText(frame, icon, ((640-iw)//2, 200), font, 0.7, (80, 120, 180), 2)
        (mw, _), _ = cv2.getTextSize(msg, font, 0.55, 1)
        cv2.putText(frame, msg, ((640-mw)//2, 260), font, 0.55, (160, 200, 220), 1)
        ret, buf = cv2.imencode('.jpg', frame)
        if ret:
            self.jpeg_bytes = buf.tobytes()
            self.update_count += 1

    def update(self):
        global cap, state
        # Do NOT open webcam(0) at startup — this machine uses an IP camera.
        # Start with no capture; show a placeholder until credentials are entered.
        cap = None
        fps = 30.0
        consecutive_failures = 0

        # Draw a static "waiting" frame so the browser shows something on load
        self._set_placeholder("Waiting for IP Camera — click Camera to connect")

        while self.running:
            loop_start = time.time()
            if getattr(state, 'new_source_path', None) is not None:
                if cap:
                    cap.release()
                if state.source == 'camera':
                    cap = CameraReader(state.new_source_path)
                else:
                    cap = VideoProcessor._open_cap(state.new_source_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                state.new_source_path = None
                consecutive_failures = 0
                state.inner_count = 0
                state.outer_count = 0
                for p in state.plates: 
                    p['inner_ok'] = False
                    p['outer_ok'] = False
                state.plate_frames = [{'inner': 0, 'outer': 0} for _ in range(3)]
                
            if getattr(state, 'trigger_restart', False):
                if state.source == 'video' and state.video_name != 'No video found' and cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state.trigger_restart = False
                state.inner_count = 0
                state.outer_count = 0
                for p in state.plates: 
                    p['inner_ok'] = False
                    p['outer_ok'] = False
                state.plate_frames = [{'inner': 0, 'outer': 0} for _ in range(3)]
                
            if state.paused:
                if getattr(self, 'last_frame', None) is not None:
                    # Keep streaming the last frame so the browser doesn't black screen on refresh
                    frame = self.last_frame.copy()
                    
                    # Draw a neat PAUSED badge in the corner
                    text = "PAUSED"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.0
                    thick = 2
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                    cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 0), -1)
                    cv2.putText(frame, text, (20, 10 + th + 10), font, scale, (0, 0, 255), thick)
                    
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        self.jpeg_bytes = buffer.tobytes()
                        self.update_count += 1
                time.sleep(0.1)
                continue
                
            if cap is None:
                self._set_placeholder("Waiting for IP Camera \u2014 click Camera to connect")
                time.sleep(0.5)
                continue


            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                if state.source == 'video' and state.video_name != 'No video found' and consecutive_failures < 5:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    msg = "Camera disconnected or video file invalid"
                    cv2.putText(frame, msg, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        self.jpeg_bytes = buffer.tobytes()
                        self.update_count += 1
                    time.sleep(0.1)
                    continue
            else:
                consecutive_failures = 0
            
            if not state.model_loaded:
                load_model()

            if state.model_loaded and state.model:
                # Provide the freshest frame to the inference thread
                self.latest_frame_for_yolo = frame.copy()

                boxes = getattr(self, '_last_boxes', None)
                classes = getattr(self, '_last_classes', {})

                # Always display latest annotated frame using the most recent boxes
                if boxes is not None and len(boxes) > 0:
                    for i, c in enumerate(boxes.cls):
                        cls_name = classes[int(c)].lower()
                        coords = boxes.xyxy[i].tolist()
                        x1, y1, x2, y2 = [int(v) for v in coords]
                        # Cyan for outer, Blue for inner
                        color = (255, 255, 0) if 'outer' in cls_name else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)  # Very thick box for visibility
                        
                        # Draw label background
                        label = f"{cls_name} {boxes.conf[i]:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.8
                        thickness = 4
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Ensure label doesn't go off the top of the screen
                        y_bg = max(y1, th + 20)
                        cv2.rectangle(frame, (x1, y_bg - th - 20), (x1 + tw + 15, y_bg), color, -1)
                        
                        # Draw text over background (black on cyan, white on blue)
                        text_color = (0, 0, 0) if 'outer' in cls_name else (255, 255, 255)
                        cv2.putText(frame, label, (x1 + 8, y_bg - 10), font, font_scale, text_color, thickness)

                # Count detections in this frame (for auto-reset logic)
                inner_c = 0
                outer_c = 0
                if boxes is not None and len(boxes) > 0:
                    for c in boxes.cls:
                        cls_name = classes[int(c)].lower()
                        if 'inner' in cls_name:
                            inner_c += 1
                        elif 'outer' in cls_name:
                            outer_c += 1

                # === BATCH COMPLETE OR ABORTED: Watch for auto-reset ===
                if inner_c == 0 and outer_c == 0:
                    state.empty_frames += 1
                else:
                    state.empty_frames = 0
                    
                # After 2 seconds of empty view (60 frames), reset for next batch
                if state.empty_frames > 60:
                    for p in state.plates:
                        p['inner_ok'] = False
                        p['outer_ok'] = False
                    for pf in state.plate_frames:
                        pf['inner'] = 0
                        pf['outer'] = 0
                    state.inner_count = 0
                    state.outer_count = 0
                    state.empty_frames = 0

                # === BATCH IN PROGRESS: Update per-plate state ===
                elif boxes is not None and len(boxes) > 0:
                    width = frame.shape[1]
                    outer_status = [False, False, False]
                    inner_status = [False, False, False]
                    
                    for i, c in enumerate(boxes.cls):
                        cls_name = classes[int(c)].lower()
                        coords = boxes.xyxy[i].tolist()
                        center_x = (coords[0] + coords[2]) / 2
                        rel_x = center_x / width
                        
                        if rel_x < 0.40:
                            idx = 0
                        elif rel_x < 0.60:
                            idx = 1
                        else:
                            idx = 2
                        
                        if 'inner' in cls_name:
                            inner_status[idx] = True
                        elif 'outer' in cls_name:
                            outer_status[idx] = True
                                
                    for idx in range(3):
                        if outer_status[idx]:
                            state.plate_frames[idx]['outer'] = min(state.plate_frames[idx]['outer'] + 1, 30)
                        else:
                            state.plate_frames[idx]['outer'] = max(0, state.plate_frames[idx]['outer'] - 1)
                            
                        if inner_status[idx]:
                            state.plate_frames[idx]['inner'] = min(state.plate_frames[idx]['inner'] + 1, 30)
                        else:
                            state.plate_frames[idx]['inner'] = max(0, state.plate_frames[idx]['inner'] - 1)
                            
                        # Lock only if detected consistently for 15 frames (~0.5s at 30fps)
                        if state.plate_frames[idx]['outer'] >= 15:
                            state.plates[idx]['outer_ok'] = True
                        if state.plate_frames[idx]['inner'] >= 15:
                            state.plates[idx]['inner_ok'] = True
                            
                    # Calculate global counts from locked per-plate status
                    state.inner_count = sum(1 for p in state.plates if p['inner_ok'])
                    state.outer_count = sum(1 for p in state.plates if p['outer_ok'])
            else:
                cv2.putText(frame, "Model 'model/best.pt' not found.", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Place your model in the model/ folder.", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.last_frame = frame.copy()
            # Quality 78 — visually identical to 95 but ~40% smaller → faster network
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ret:
                self.jpeg_bytes = buffer.tobytes()
                self.update_count += 1

            elapsed = time.time() - loop_start
            sleep_time = max(0.005, (1.0 / fps) - elapsed) if fps > 0 else 0.005

            if state.source == 'video':
                time.sleep(sleep_time)
            else:
                time.sleep(0.005)


processor = VideoProcessor()

def get_frame():
    """Always yield the latest JPEG — no stale-counter gating, minimal sleep."""
    last_bytes = None
    while True:
        current = processor.jpeg_bytes
        if current and current is not last_bytes:
            last_bytes = current
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current + b'\r\n')
        time.sleep(0.005)  # allow up to 200fps output limit to avoid browser lag

@app.route('/')
def index():
    # Step 1 → inner rings being placed
    # Step 2 → all inner done, outer rings being placed
    # Step 3 → BOTH inner AND outer all placed (true complete)
    current_step = 1
    if state.inner_count == 3:
        current_step = 2
    if state.inner_count == 3 and state.outer_count == 3:
        current_step = 3

    # Defect: outer rings detected but inner rings still missing
    defect = (state.outer_count > 0 and state.inner_count < 3)
    missing_inner = 3 - state.inner_count
    missing_outer = 3 - state.outer_count

    return render_template('index.html',
                           source=state.source,
                           video_name=state.video_name,
                           threshold=state.threshold,
                           paused=state.paused,
                           inner_count=state.inner_count,
                           outer_count=state.outer_count,
                           plates=state.plates,
                           current_step=current_step,
                           defect=defect,
                           missing_inner=missing_inner,
                           missing_outer=missing_outer,
                           cam_ip=state.cam_ip,
                           cam_port=state.cam_port,
                           cam_user=state.cam_user,
                           cam_stream=state.cam_stream,
                           cam_connected=state.cam_connected)

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

from flask import jsonify

@app.route('/status')
def status():
    """Lightweight JSON endpoint for sidebar live-update (avoids full page re-render)."""
    inner_count = state.inner_count
    outer_count = state.outer_count

    current_step = 1
    if inner_count == 3:
        current_step = 2
    if inner_count == 3 and outer_count == 3:
        current_step = 3

    defect = (outer_count > 0 and inner_count < 3)
    missing_inner = 3 - inner_count
    missing_outer = 3 - outer_count

    return jsonify(
        inner_count=inner_count,
        outer_count=outer_count,
        current_step=current_step,
        defect=defect,
        missing_inner=missing_inner,
        missing_outer=missing_outer,
        plates=state.plates,
        paused=state.paused,
    )

@app.route('/set_source/<src>')
def set_source(src):
    state.source = src
    if src == 'camera':
        # If IP camera credentials exist, use RTSP; otherwise fall back to webcam
        if state.cam_ip:
            sep = '' if state.cam_stream.startswith('/') else '/'
            rtsp_url = f"rtsp://{state.cam_user}:{state.cam_pass}@{state.cam_ip}:{state.cam_port}{sep}{state.cam_stream}"
            state.video_name = f'IP Cam — {state.cam_ip}'
            state.new_source_path = rtsp_url
            state.cam_connected = True
        else:
            state.video_name = 'Camera 0 (Local Webcam)'
            state.new_source_path = 0
            state.cam_connected = False
    elif src == 'video':
        video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
        if video_files:
            state.video_name = video_files[0]
            state.new_source_path = state.video_name
        else:
            state.video_name = 'No video found'
            state.new_source_path = 'nonexistent.mp4'
    return redirect(url_for('index'))

@app.route('/set_camera_creds', methods=['POST'])
def set_camera_creds():
    """Accept CP Plus / RTSP camera credentials from the UI form."""
    state.cam_ip     = request.form.get('cam_ip', '').strip()
    state.cam_port   = request.form.get('cam_port', '554').strip()
    state.cam_user   = request.form.get('cam_user', 'admin').strip()
    state.cam_pass   = request.form.get('cam_pass', '').strip()
    state.cam_stream = request.form.get('cam_stream', '/cam/realmonitor?channel=1&subtype=0').strip()

    if state.cam_ip:
        sep = '' if state.cam_stream.startswith('/') else '/'
        rtsp_url = f"rtsp://{state.cam_user}:{state.cam_pass}@{state.cam_ip}:{state.cam_port}{sep}{state.cam_stream}"
        state.source     = 'camera'
        state.video_name = f'IP Cam — {state.cam_ip}'
        state.new_source_path  = rtsp_url
        state.cam_connected    = True
        state.paused           = False
        # Reset detection state for the new stream
        state.inner_count = 0
        state.outer_count = 0
        for p in state.plates:
            p['inner_ok'] = False
            p['outer_ok'] = False
        state.plate_frames = [{'inner': 0, 'outer': 0} for _ in range(3)]
    return redirect(url_for('index'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['video_file']
    if file.filename == '':
        return redirect(url_for('index'))
        
    filename = file.filename
    file.save(filename)
    
    state.source = 'video'
    state.video_name = filename
    state.new_source_path = filename
    state.paused = False
    
    return redirect(url_for('index'))

@app.route('/toggle_resume')
def toggle_resume():
    state.paused = not state.paused
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    state.paused = True
    return redirect(url_for('index'))

@app.route('/restart')
def restart():
    state.trigger_restart = True
    state.paused = False
    return redirect(url_for('index'))

@app.route('/set_threshold')
def set_threshold():
    val = request.args.get('val', default=3, type=int)
    state.threshold = val
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
