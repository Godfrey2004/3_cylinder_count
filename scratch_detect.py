import cv2
import os
import time
from ultralytics import YOLO

# Load model
model = YOLO('model/best.pt')

# Camera URL
rtsp_url = "rtsp://192.168.1.250:554/video/live?channel=1&subtype=0"
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Failed to open RTSP camera stream")
    # Try local webcam or first video file as fallback to see if script runs
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if video_files:
        print(f"Opening fallback video: {video_files[0]}")
        cap = cv2.VideoCapture(video_files[0])
    else:
        print("No fallback video found")
        exit(1)

print("Reading 10 frames and printing detections:")
frame_count = 0
while frame_count < 10:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        time.sleep(0.1)
        continue
    
    h, w = frame.shape[:2]
    results = model(frame, conf=0.1, verbose=False)
    boxes = results[0].boxes
    names = results[0].names
    
    print(f"\n--- Frame {frame_count} (Resolution: {w}x{h}) ---")
    if len(boxes) == 0:
        print("No detections")
    for i, c in enumerate(boxes.cls):
        cls_name = names[int(c)]
        conf_val = float(boxes.conf[i])
        coords = boxes.xyxy[i].tolist()
        x1, y1, x2, y2 = coords
        bbox_w_frac = (x2 - x1) / w
        center_x = (x1 + x2) / 2
        rel_x = center_x / w
        print(f"Class: {cls_name} | Conf: {conf_val:.4f} | Center X: {rel_x:.3f} | Width Frac: {bbox_w_frac:.4f}")
        
    frame_count += 1
    time.sleep(0.1)

cap.release()
