import cv2
import os
import threading
import datetime
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, Response, request, redirect, url_for
import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

def get_current_date_str():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def get_log_dir():
    dir_path = os.path.join(os.path.dirname(__file__), 'assembly_logs')
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_log_file_path(date_str=None):
    if date_str is None:
        date_str = get_current_date_str()
    return os.path.join(get_log_dir(), f"{date_str}.txt")

def load_cycle_count():
    current_date = get_current_date_str()
    file_path = get_log_file_path(current_date)
    count = 0
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(':')
                if len(parts) == 2:
                    date_str = parts[0].strip()
                    val_str = parts[1].strip()
                    if date_str == current_date:
                        count = int(val_str)
                        break
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return count

def save_cycle_count(count):
    """Write today's count. When count==0, remove today's entire file."""
    current_date = get_current_date_str()
    file_path = get_log_file_path(current_date)

    if count == 0:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        return

    raw = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw = [l.rstrip('\n') for l in f.readlines()]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Update or add today's date header count (keep existing detail lines)
    output = [l for l in raw if l.strip()]
    found = False
    for i, line in enumerate(output):
        if line.strip().startswith(f"{current_date}:"):
            output[i] = f"{current_date}: {count}"
            found = True
            break
    if not found:
        output.insert(0, f"{current_date}: {count}")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in output:
                f.write(line + '\n')
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def _is_date_header(line):
    """Return True if line looks like a date header: YYYY-MM-DD: N"""
    parts = line.split(':')
    return len(parts) >= 2 and len(parts[0]) == 10 and parts[0][4] == '-' and parts[0][7] == '-'

def save_cycle_count_with_detail(count, start_ts, end_ts, duration, is_ok=True, details=None):
    """Append a cycle record grouped under OK / Not OK categories per date."""
    current_date = get_current_date_str()
    file_path = get_log_file_path(current_date)
    details_suffix = f" | {details}" if details else ""
    new_detail = f"- Cycle {count}: Start {start_ts}, End {end_ts} (Duration: {duration:.1f}s){details_suffix}"

    # --- Parse existing file into: {'ok': [...], 'notok': [...]} ---
    sections = {'ok': [], 'notok': []}

    raw = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw = [l.rstrip('\n') for l in f.readlines()]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    cur_cat  = None   # 'ok' | 'notok' | None
    for line in raw:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_date_header(stripped):
            cur_cat  = None
        elif stripped.startswith('OK:'):
            cur_cat = 'ok'
        elif stripped.startswith('Not OK:'):
            cur_cat = 'notok'
        elif stripped.startswith('- Cycle'):
            cat = cur_cat if cur_cat else 'ok'
            sections[cat].append(stripped)

    # --- Add new cycle to today's section ---
    cat_key = 'ok' if is_ok else 'notok'
    sections[cat_key].append(new_detail)

    # --- Write back in grouped format ---
    ok_list    = sections['ok']
    notok_list = sections['notok']
    total = len(ok_list) + len(notok_list)
    output = []
    output.append(f"{current_date}: {total}")
    if ok_list:
        output.append(f"  OK: {len(ok_list)}")
        for d in ok_list:
            output.append(f"    {d}")
    if notok_list:
        output.append(f"  Not OK: {len(notok_list)}")
        for d in notok_list:
            output.append(f"    {d}")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in output:
                f.write(line + '\n')
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def check_date_transition():
    current_date = get_current_date_str()
    if getattr(state, 'current_cycle_date', '') != current_date:
        state.current_cycle_date = current_date
        state.cycle_count = load_cycle_count()

# Global state
class AppState:
    paused = False
    source = 'camera'  # 'camera' or 'video'
    video_name = 'Camera 0'
    threshold = 6.0
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
    sealing_model = None
    person_model = None
    sealing_model_loaded = False
    person_model_loaded = False
    person_detected = False
    person_count = 0
    show_annotations = True
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
    
    # Cycle counting
    cycle_count = 0
    current_cycle_date = ''
    cycle_completed_pending = False
    
    # Timer variables
    cycle_active = False
    cycle_start_time = None
    cycle_end_time = None
    cycle_elapsed = 0.0
    cycle_start_timestamp = ''
    cycle_end_timestamp = ''
    cycle_is_completed = False
    cycle_logged = False

    # Phase gate: False = inner phase (monitor inner live)
    #             True  = outer phase (inner locked, monitor outer live)
    outer_phase_started = False
    # One-way latch: becomes True the first time inner_ok turns True for each cylinder.
    # Never cleared during Phase 1 — used as the snapshot when outer phase begins.
    inner_ever_confirmed = [False, False, False]
    # Snapshot locked at phase transition — hard-locks inner_ok for rest of cycle.
    inner_snapshot = [False, False, False]

state = AppState()
state.current_cycle_date = get_current_date_str()
state.cycle_count = load_cycle_count()
save_cycle_count(state.cycle_count)

def load_model():
    sealing_path = os.path.join('model', 'best.pt')
    person_path = os.path.join('model', 'person.pt')
    
    if YOLO is not None:
        # Load sealing model
        if os.path.exists(sealing_path):
            try:
                state.sealing_model = YOLO(sealing_path)
                state.sealing_model_loaded = True
                state.model = state.sealing_model
                state.model_loaded = True
                print("Sealing model loaded successfully")
            except Exception as e:
                print(f"Error loading sealing model: {e}")
        else:
            print("Sealing model best.pt not found. Waiting for upload...")
            
        # Load person model (temporarily disabled)
        # if os.path.exists(person_path):
        #     try:
        #         state.person_model = YOLO(person_path)
        #         state.person_model_loaded = True
        #         print("Person model loaded successfully")
        #     except Exception as e:
        #         print(f"Error loading person model: {e}")
        # else:
        #     print("Person model person.pt not found.")
        pass
    else:
        print("Ultralytics not installed. Waiting...")

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
        self._last_boxes_person = None
        self._last_classes_person = {}
        
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
                
                # Run sealing model
                if getattr(state, 'sealing_model_loaded', False) and state.sealing_model:
                    try:
                        results = state.sealing_model(frame, conf=state.threshold / 10.0, verbose=False)
                        self._last_boxes = results[0].boxes
                        self._last_classes = results[0].names
                    except Exception as e:
                        print(f"Error running sealing model inference: {e}")
                
                # Run person model (temporarily disabled)
                # if getattr(state, 'person_model_loaded', False) and state.person_model:
                #     try:
                #         results_p = state.person_model(frame, conf=0.4, verbose=False)
                #         self._last_boxes_person = results_p[0].boxes
                #         self._last_classes_person = results_p[0].names
                #     except Exception as e:
                #         print(f"Error running person model inference: {e}")
                pass
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
                state.outer_phase_started = False
                state.inner_ever_confirmed = [False, False, False]
                state.inner_snapshot = [False, False, False]
                for p in state.plates: 
                    p['inner_ok'] = False
                    p['outer_ok'] = False
                state.plate_frames = [{'inner': 0, 'outer': 0} for _ in range(3)]
                
            if getattr(state, 'trigger_restart', False):
                if state.source == 'video' and state.video_name != 'No video found':
                    if cap:
                        try:
                            cap.release()
                        except:
                            pass
                    cap = VideoProcessor._open_cap(state.video_name)
                elif state.source == 'camera':
                    if cap:
                        try:
                            cap.release()
                        except:
                            pass
                    if state.cam_connected and state.cam_ip:
                        sep = '' if state.cam_stream.startswith('/') else '/'
                        rtsp_url = f"rtsp://{state.cam_user}:{state.cam_pass}@{state.cam_ip}:{state.cam_port}{sep}{state.cam_stream}"
                        cap = CameraReader(rtsp_url)
                    else:
                        cap = CameraReader(0)
                        
                state.trigger_restart = False
                state.inner_count = 0
                state.outer_count = 0
                state.outer_phase_started = False
                state.inner_ever_confirmed = [False, False, False]
                state.inner_snapshot = [False, False, False]
                state.cycle_completed_pending = False
                state.cycle_active = False
                state.cycle_elapsed = 0.0
                state.cycle_is_completed = False
                for p in state.plates: 
                    p['inner_ok'] = False
                    p['outer_ok'] = False
                state.plate_frames = [{'inner': 0, 'outer': 0} for _ in range(3)]
                consecutive_failures = 0
                
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
                if state.source == 'video' and state.video_name != 'No video found':
                    if cap:
                        try:
                            cap.release()
                        except:
                            pass
                    cap = VideoProcessor._open_cap(state.video_name)
                    consecutive_failures = 0
                    time.sleep(0.05)
                    continue
                else:
                    consecutive_failures += 1
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
            
            if not getattr(state, 'sealing_model_loaded', False) and not getattr(state, 'person_model_loaded', False):
                load_model()

            if getattr(state, 'sealing_model_loaded', False) or getattr(state, 'person_model_loaded', False):
                # Provide the freshest frame to the inference thread
                self.latest_frame_for_yolo = frame.copy()

                boxes = getattr(self, '_last_boxes', None)
                classes = getattr(self, '_last_classes', {})
                boxes_person = getattr(self, '_last_boxes_person', None)
                classes_person = getattr(self, '_last_classes_person', {})

                # Proportional font scaling
                h, w = frame.shape[:2]
                base_w = 1280.0
                scale_factor = w / base_w
                font_scale = max(0.5, 1.2 * scale_factor)
                thickness = max(1, int(3 * scale_factor))

                # Always display latest annotated frame using the most recent boxes (sealing)
                if state.show_annotations and boxes is not None and len(boxes) > 0:
                    for i, c in enumerate(boxes.cls):
                        cls_name = classes[int(c)].lower()
                        coords = boxes.xyxy[i].tolist()
                        x1, y1, x2, y2 = [int(v) for v in coords]
                        # Cyan for outer, Blue for inner
                        color = (255, 255, 0) if 'outer' in cls_name else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, int(5 * scale_factor)))
                        
                        # Draw label background
                        label = f"{cls_name} {boxes.conf[i]:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Ensure label doesn't go off the top of the screen
                        y_bg = max(y1, th + int(15 * scale_factor))
                        cv2.rectangle(frame, (x1, y_bg - th - int(15 * scale_factor)), (x1 + tw + int(10 * scale_factor), y_bg), color, -1)
                        
                        # Draw text over background (black on cyan, white on blue)
                        text_color = (0, 0, 0) if 'outer' in cls_name else (255, 255, 255)
                        cv2.putText(frame, label, (x1 + int(5 * scale_factor), y_bg - int(5 * scale_factor)), font, font_scale, text_color, thickness)

                # Draw person detections (temporarily disabled)
                state.person_count = 0
                state.person_detected = False

                # ── Draw P1 / P2 / P3 individual boxes around the cylinders ────────
                # Defined using the physical center points and dimensions of the cylinders:
                # P1 center ~26%, P2 center ~52%, P3 center ~78% of frame width.
                box_y1 = int(h * 0.63)
                box_y2 = int(h * 0.80)
                
                # Boxes are positioned to align perfectly with the physical cylinders and gaps
                plates_regions = [
                    (int(w * 0.25), int(w * 0.47)),  # P1
                    (int(w * 0.48), int(w * 0.685)), # P2
                    (int(w * 0.695), int(w * 0.87))  # P3
                ]
                
                font_z = cv2.FONT_HERSHEY_SIMPLEX
                fscale_z = max(0.4, 0.7 * scale_factor)
                fthick_z = max(1, int(2 * scale_factor))

                for zi, (zx1, zx2) in enumerate(plates_regions):
                    plate = state.plates[zi]
                    label_z = f"P{zi + 1}"

                    # Choose colour based on current ring status
                    if not state.cycle_active:
                        col = (160, 160, 160)       # grey — idle
                    elif plate['inner_ok'] and plate['outer_ok']:
                        col = (0, 200, 60)           # green — both rings confirmed
                    elif plate['inner_ok']:
                        col = (0, 165, 255)          # orange — inner only
                    else:
                        col = (0, 0, 220)            # red — missing

                    # Semi-transparent fill inside the individual cylinder box
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (zx1, box_y1), (zx2, box_y2), col, -1)
                    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

                    # Draw the individual border
                    cv2.rectangle(frame, (zx1, box_y1), (zx2, box_y2), col,
                                  max(2, int(3 * scale_factor)))

                    # Label background + text centered on top of each box
                    (tw_z, th_z), _ = cv2.getTextSize(label_z, font_z, fscale_z, fthick_z)
                    lx = zx1 + (zx2 - zx1 - tw_z) // 2      # horizontally centred
                    ly = box_y1 + th_z + int(6 * scale_factor)
                    cv2.rectangle(frame, (lx - 4, ly - th_z - 4), (lx + tw_z + 4, ly + 4), col, -1)
                    cv2.putText(frame, label_z, (lx, ly), font_z, fscale_z, (255, 255, 255), fthick_z)

                    # Ring status sub-label (tiny)
                    inner_sym = "I:OK" if plate['inner_ok'] else "I:--"
                    outer_sym = "O:OK" if plate['outer_ok'] else "O:--"
                    sub_label = f"{inner_sym}  {outer_sym}"
                    fs_sub = max(0.28, 0.45 * scale_factor)
                    (sw, sh), _ = cv2.getTextSize(sub_label, font_z, fs_sub, 1)
                    sx = zx1 + (zx2 - zx1 - sw) // 2
                    sy = ly + sh + int(10 * scale_factor)
                    cv2.putText(frame, sub_label, (sx, sy), font_z, fs_sub, col, 1)

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
                    if state.cycle_active and not state.cycle_is_completed:
                        state.cycle_elapsed = time.time() - state.cycle_start_time
                else:
                    state.empty_frames = 0
                    if not state.cycle_active:
                        # Only start a new cycle if it's not a finished plate (e.g. outer rings < 2)
                        if outer_c < 2:
                            state.cycle_active = True
                            state.cycle_start_time = time.time()
                            state.cycle_start_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            state.cycle_is_completed = False
                            state.cycle_logged = False
                            state.cycle_elapsed = 0.0
                    elif not state.cycle_is_completed:
                        state.cycle_elapsed = time.time() - state.cycle_start_time
                        
                        # Immediate completion logging:
                        if state.inner_count == 3 and state.outer_count == 3:
                            if not state.cycle_logged:
                                state.cycle_is_completed = True
                                state.cycle_end_time = time.time()
                                state.cycle_end_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                                state.cycle_elapsed = state.cycle_end_time - state.cycle_start_time
                                check_date_transition()
                                state.cycle_count += 1
                                
                                # Gather plate counts
                                counts_list = []
                                total_rings = 0
                                for p in state.plates:
                                    inner_val = 1 if p['inner_ok'] else 0
                                    outer_val = 1 if p['outer_ok'] else 0
                                    total_rings += (inner_val + outer_val)
                                    counts_list.append(f"{p['id']}: Inner {inner_val}, Outer {outer_val}")
                                total_missing = 6 - total_rings
                                plate_counts_str = f"Total Rings: {total_rings} | Total Missing: {total_missing} | " + " | ".join(counts_list)

                                save_cycle_count_with_detail(
                                    state.cycle_count,
                                    state.cycle_start_timestamp,
                                    state.cycle_end_timestamp,
                                    state.cycle_elapsed,
                                    is_ok=True,
                                    details=plate_counts_str
                                )
                                state.cycle_logged = True
                                print(f"Cycle completed successfully! Total today: {state.cycle_count}")
                    
                # Reset cycle when the template is empty.
                # We use 75 frames (2.5 seconds) of empty view to reset.
                if state.empty_frames > 75:
                    # If a cycle was started but never completed and not logged, log it as Not OK:
                    if state.cycle_active and not state.cycle_logged:
                        if state.inner_count > 0 or state.outer_count > 0:
                            check_date_transition()
                            state.cycle_count += 1
                            end_time = time.time() - 2.5
                            duration = max(0.1, end_time - state.cycle_start_time)
                            end_ts = datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
                            
                            # Gather plate counts
                            counts_list = []
                            total_rings = 0
                            for p in state.plates:
                                inner_val = 1 if p['inner_ok'] else 0
                                outer_val = 1 if p['outer_ok'] else 0
                                total_rings += (inner_val + outer_val)
                                counts_list.append(f"{p['id']}: Inner {inner_val}, Outer {outer_val}")
                            total_missing = 6 - total_rings
                            plate_counts_str = f"Total Rings: {total_rings} | Total Missing: {total_missing} | " + " | ".join(counts_list)

                            # Gather missing info separately
                            missing_list = []
                            for p in state.plates:
                                if not p['inner_ok']:
                                    missing_list.append(f"{p['id']} Inner missing")
                                if not p['outer_ok']:
                                    missing_list.append(f"{p['id']} Outer missing")
                            
                            missing_info = None
                            if missing_list:
                                missing_info = f"Missing: {', '.join(missing_list)}"

                            details_str = plate_counts_str
                            if missing_info:
                                details_str += f" | {missing_info}"

                            save_cycle_count_with_detail(
                                state.cycle_count,
                                state.cycle_start_timestamp,
                                end_ts,
                                duration,
                                is_ok=False,
                                details=details_str
                            )
                            print(f"Cycle logged as Not OK! Total today: {state.cycle_count}")
                    
                    # Reset all variables for next cycle
                    state.cycle_active = False
                    state.cycle_elapsed = 0.0
                    state.cycle_is_completed = False
                    state.cycle_logged = False
                    
                    for p in state.plates:
                        p['inner_ok'] = False
                        p['outer_ok'] = False
                    for pf in state.plate_frames:
                        pf['inner'] = 0
                        pf['outer'] = 0
                    state.inner_count = 0
                    state.outer_count = 0
                    state.outer_phase_started = False
                    state.inner_ever_confirmed = [False, False, False]
                    state.inner_snapshot = [False, False, False]
                    state.empty_frames = 0

                # === ALWAYS: Update per-plate state — simple accumulate-only latches ===
                # Inner and outer are INDEPENDENT. Once confirmed, each stays confirmed
                # until the cycle resets. No phase gating needed.
                else:
                    width = frame.shape[1]
                    outer_status = [False, False, False]
                    inner_status = [False, False, False]

                    if boxes is not None and len(boxes) > 0:
                        for i, c in enumerate(boxes.cls):
                            cls_name = classes[int(c)].lower()
                            coords = boxes.xyxy[i].tolist()
                            center_x = (coords[0] + coords[2]) / 2
                            rel_x = center_x / width

                            if rel_x < 0.475:
                                idx = 0
                            elif rel_x < 0.69:
                                idx = 1
                            else:
                                idx = 2

                            if 'inner' in cls_name:
                                inner_status[idx] = True
                            elif 'outer' in cls_name:
                                outer_status[idx] = True

                    for idx in range(3):
                        # ── INNER ring: accumulate-only, one-way latch ──
                        # Counter only goes up. Once inner_ok = True, it never clears
                        # during this cycle — outer rings appearing cannot affect it.
                        if inner_status[idx]:
                            state.plate_frames[idx]['inner'] = min(state.plate_frames[idx]['inner'] + 1, 30)
                            if state.plate_frames[idx]['inner'] >= 8:
                                state.plates[idx]['inner_ok'] = True

                        # ── OUTER ring: accumulate-only, one-way latch ──
                        # Counter only goes up. Once outer_ok = True, it never clears
                        # during this cycle — press covering rings cannot affect it.
                        if outer_status[idx]:
                            state.plate_frames[idx]['outer'] = min(state.plate_frames[idx]['outer'] + 1, 30)
                            if state.plate_frames[idx]['outer'] >= 15:
                                state.plates[idx]['outer_ok'] = True

                    # Live counts always reflect current state
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
    check_date_transition()
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

    ok_count, notok_count = get_today_stats()
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
                           cam_connected=state.cam_connected,
                           person_detected=state.person_detected,
                           person_count=state.person_count,
                           show_annotations=state.show_annotations,
                           cycle_count=state.cycle_count,
                           current_cycle_date=state.current_cycle_date,
                           cycle_active=state.cycle_active,
                           cycle_elapsed=state.cycle_elapsed,
                           cycle_is_completed=state.cycle_is_completed,
                           ok_count=ok_count,
                           notok_count=notok_count)

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

from flask import jsonify

@app.route('/status')
def status():
    """Lightweight JSON endpoint for sidebar live-update (avoids full page re-render)."""
    check_date_transition()
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

    ok_count, notok_count = get_today_stats()
    return jsonify(
        inner_count=inner_count,
        outer_count=outer_count,
        current_step=current_step,
        defect=defect,
        missing_inner=missing_inner,
        missing_outer=missing_outer,
        plates=state.plates,
        paused=state.paused,
        person_detected=state.person_detected,
        person_count=state.person_count,
        show_annotations=state.show_annotations,
        cycle_count=state.cycle_count,
        current_cycle_date=state.current_cycle_date,
        cycle_active=state.cycle_active,
        cycle_elapsed=state.cycle_elapsed,
        cycle_is_completed=state.cycle_is_completed,
        ok_count=ok_count,
        notok_count=notok_count,
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
    state.cycle_completed_pending = False
    return redirect(url_for('index'))

@app.route('/reset_cycle')
def reset_cycle():
    state.cycle_count = 0
    state.cycle_active = False
    state.cycle_elapsed = 0.0
    state.cycle_is_completed = False
    state.cycle_completed_pending = False
    save_cycle_count(0)   # removes today's block from the file
    return redirect(url_for('index'))

@app.route('/set_threshold')
def set_threshold():
    val = request.args.get('val', default=6.0, type=float)
    state.threshold = val
    return redirect(url_for('index'))

@app.route('/toggle_annotations')
def toggle_annotations():
    state.show_annotations = not state.show_annotations
    return jsonify(show_annotations=state.show_annotations)

def get_today_stats():
    current_date = get_current_date_str()
    file_path = get_log_file_path(current_date)
    ok_count = 0
    notok_count = 0
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw = [l.rstrip('\n') for l in f.readlines()]
            cur_cat = None
            for line in raw:
                stripped = line.strip()
                if not stripped:
                    continue
                if _is_date_header(stripped):
                    cur_cat = None
                elif stripped.startswith('OK:'):
                    cur_cat = 'ok'
                elif stripped.startswith('Not OK:'):
                    cur_cat = 'notok'
                elif stripped.startswith('- Cycle'):
                    cat = cur_cat if cur_cat else 'ok'
                    if cat == 'ok':
                        ok_count += 1
                    else:
                        notok_count += 1
        except Exception as e:
            print(f"Error reading stats: {e}")
    return ok_count, notok_count

@app.route('/today_log')
def today_log():
    current_date = get_current_date_str()
    file_path = get_log_file_path(current_date)
    
    ok_list = []
    notok_list = []
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw = [l.rstrip('\n') for l in f.readlines()]
            cur_cat = None
            for line in raw:
                stripped = line.strip()
                if not stripped:
                    continue
                if _is_date_header(stripped):
                    cur_cat = None
                elif stripped.startswith('OK:'):
                    cur_cat = 'ok'
                elif stripped.startswith('Not OK:'):
                    cur_cat = 'notok'
                elif stripped.startswith('- Cycle'):
                    cat = cur_cat if cur_cat else 'ok'
                    if cat == 'ok':
                        ok_list.append(stripped)
                    else:
                        notok_list.append(stripped)
        except Exception as e:
            print(f"Error reading today's log: {e}")
            
    return jsonify(
        date=current_date,
        ok=ok_list,
        notok=notok_list,
        total=len(ok_list) + len(notok_list)
    )

from flask import send_file

@app.route('/download_log')
def download_log():
    current_date = get_current_date_str()
    file_path = get_log_file_path(current_date)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=f"assembly_log_{current_date}.txt")
    else:
        # Return empty text file if doesn't exist
        from io import BytesIO
        mem = BytesIO(b"No logs recorded for today yet.")
        return send_file(mem, as_attachment=True, download_name=f"assembly_log_{current_date}.txt", mimetype='text/plain')

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
