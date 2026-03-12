import base64
import json
import os
from collections import deque
from typing import Dict
import urllib.request
import requests
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory

MODEL_PATH = "model.tflite"
SCALER_JSON = "scaler.json"
LABELS_JSON = "labels.json"
SEQUENCE_LEN = 30
# Accuracy-first default: wait for full trained sequence length unless overridden.
MIN_PREDICT_FRAMES = int(os.environ.get("MIN_PREDICT_FRAMES", str(SEQUENCE_LEN)))
N_FEATURES = 225
THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.85"))
PROCESS_WIDTH = int(os.environ.get("PROCESS_WIDTH", "320"))
PROCESS_HEIGHT = int(os.environ.get("PROCESS_HEIGHT", "240"))
UPPER_BODY_IDX = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
]

MIN_PREDICT_FRAMES = max(1, min(MIN_PREDICT_FRAMES, SEQUENCE_LEN))

app = Flask(__name__, static_folder='static', template_folder='templates')

# Client buffers
client_buffers: Dict[str, deque] = {}

# Load scaler
with open(SCALER_JSON, "r", encoding="utf-8") as f:
    scaler_data = json.load(f)
mean = np.array(scaler_data["mean"], dtype=np.float32)
scale = np.array(scaler_data["scale"], dtype=np.float32)
scale = np.where(scale == 0.0, 1.0, scale)

# Load labels
with open(LABELS_JSON, "r", encoding="utf-8") as f:
    labels = np.array(json.load(f))

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2,
)
pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def normalize_hand(hand_flat: np.ndarray) -> np.ndarray:
    pts = hand_flat.reshape(21, 3).copy()
    wrist = pts[0].copy()
    pts -= wrist
    scale_norm = np.linalg.norm(pts[9])
    if scale_norm > 1e-6:
        pts /= scale_norm
    return pts.flatten().astype(np.float32)

def normalize_pose(pose_flat: np.ndarray) -> np.ndarray:
    pts = pose_flat.reshape(33, 3).copy()
    l_sh = pts[11].copy()
    r_sh = pts[12].copy()
    midpoint = (l_sh + r_sh) / 2.0
    pts -= midpoint
    width = np.linalg.norm(l_sh - r_sh)
    if width > 1e-6:
        pts /= width
    return pts.flatten().astype(np.float32)

def extract_keypoints(hand_results, pose_results) -> np.ndarray:
    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for landmarks, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness,
        ):
            raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            label = handedness.classification[0].label
            norm = normalize_hand(raw)
            if label == "Left":
                lh = norm
            else:
                rh = norm

    pose_vec = np.zeros(99, dtype=np.float32)
    if pose_results.pose_landmarks:
        raw_pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        pose_vec = normalize_pose(raw_pose)

    return np.concatenate([lh, rh, pose_vec])


def serialize_landmarks(hand_results, pose_results) -> dict:
    left_hand = []
    right_hand = []

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for landmarks, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness,
        ):
            pts = [
                {"x": float(lm.x), "y": float(lm.y)}
                for lm in landmarks.landmark
            ]
            label = handedness.classification[0].label
            if label == "Left":
                left_hand = pts
            else:
                right_hand = pts

    pose_upper = []
    if pose_results.pose_landmarks:
        pose_upper = [
            {
                "idx": idx,
                "x": float(pose_results.pose_landmarks.landmark[idx].x),
                "y": float(pose_results.pose_landmarks.landmark[idx].y),
                "visibility": float(pose_results.pose_landmarks.landmark[idx].visibility),
            }
            for idx in UPPER_BODY_IDX
        ]

    return {
        "left_hand": left_hand,
        "right_hand": right_hand,
        "pose_upper": pose_upper,
    }

def decode_image(data_url: str) -> np.ndarray:
    """Decode base64 image from webcam"""
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Invalid image data")
    frame_bgr = cv2.resize(frame_bgr, (PROCESS_WIDTH, PROCESS_HEIGHT))
    return frame_bgr

def fetch_ip_camera_frame(camera_url: str) -> np.ndarray:
    """Fetch frame from IP camera"""
    try:
        # Handle different URL types
        parsed = urlparse(camera_url)
        
        # For HTTP/HTTPS streams
        if parsed.scheme in ['http', 'https']:
            # Try MJPEG stream first
            if any(x in camera_url.lower() for x in ['mjpg', 'mjpeg', 'stream', 'video']):
                # For MJPEG streams, we need to keep connection open
                # This is simplified - in production you'd want persistent connection
                resp = requests.get(camera_url, stream=True, timeout=5)
                bytes_data = bytes()
                for chunk in resp.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            return cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            else:
                # Single image fetch
                resp = urllib.request.urlopen(camera_url, timeout=5)
                img_array = np.array(bytearray(resp.read()), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    return cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        # For RTSP streams
        elif parsed.scheme == 'rtsp':
            # Use OpenCV for RTSP (simplified - consider using GStreamer in production)
            cap = cv2.VideoCapture(camera_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        raise ValueError(f"Could not fetch frame from {camera_url}")
        
    except Exception as e:
        print(f"Error fetching IP camera frame: {e}")
        return None

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "healthy", "buffers": len(client_buffers)})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    client_id = str(payload.get("client_id", "default"))
    
    # Check if it's webcam or IP camera
    if "image" in payload:
        # Webcam mode
        image_data = payload.get("image")
        if not image_data:
            return jsonify({"error": "Missing image"}), 400
        
        try:
            frame_bgr = decode_image(image_data)
        except Exception as e:
            return jsonify({"error": f"Invalid image: {str(e)}"}), 400
            
    elif "camera_url" in payload:
        # IP camera mode
        camera_url = payload.get("camera_url")
        if not camera_url:
            return jsonify({"error": "Missing camera_url"}), 400
        
        frame_bgr = fetch_ip_camera_frame(camera_url)
        if frame_bgr is None:
            return jsonify({"error": "Could not fetch camera frame"}), 500
    else:
        return jsonify({"error": "Missing image or camera_url"}), 400

    try:
        # Process frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hr = hands.process(frame_rgb)
        pr = pose.process(frame_rgb)
        kp = extract_keypoints(hr, pr)
        landmarks = serialize_landmarks(hr, pr)

        # Only collect/predict when at least one hand is visible.
        hand_detected = bool(landmarks["left_hand"] or landmarks["right_hand"])

        if not hand_detected:
            if client_id in client_buffers:
                client_buffers[client_id].clear()
            return jsonify({
                "status": "idle",
                "frames": 0,
                "needed": SEQUENCE_LEN,
                "landmarks": landmarks,
            })

        # Update buffer
        if client_id not in client_buffers:
            client_buffers[client_id] = deque(maxlen=SEQUENCE_LEN)
        client_buffers[client_id].append(kp)

        buffer = client_buffers[client_id]
        if len(buffer) < MIN_PREDICT_FRAMES:
            return jsonify({
                "status": "collecting",
                "frames": len(buffer),
                "needed": MIN_PREDICT_FRAMES,
                "landmarks": landmarks,
            })

        # Predict
        # Keep model input shape fixed at SEQUENCE_LEN by padding early buffers.
        if len(buffer) < SEQUENCE_LEN:
            pad_count = SEQUENCE_LEN - len(buffer)
            last_kp = buffer[-1]
            padded = list(buffer) + [last_kp.copy() for _ in range(pad_count)]
            seq = np.stack(padded, axis=0).astype(np.float32)
        else:
            seq = np.stack(buffer, axis=0).astype(np.float32)
        flat = seq.reshape(-1, N_FEATURES)
        scaled = (flat - mean) / scale
        input_seq = scaled.reshape(1, SEQUENCE_LEN, N_FEATURES).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], input_seq)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        pred_idx = int(np.argmax(output))
        confidence = float(output[pred_idx])

        top_idx = np.argsort(output)[::-1][:3]
        top_predictions = [
            {"label": str(labels[i]), "confidence": float(output[i])}
            for i in top_idx
        ]

        return jsonify({
            "status": "ok",
            "prediction": str(labels[pred_idx]),
            "confidence": confidence,
            "accepted": confidence >= THRESHOLD,
            "top_predictions": top_predictions,
            "frames": len(buffer),
            "needed": MIN_PREDICT_FRAMES,
            "landmarks": landmarks,
        })
        
    except Exception as exc:
        print(f"Prediction error: {str(exc)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

@app.post("/reset")
def reset():
    payload = request.get_json(silent=True) or {}
    client_id = str(payload.get("client_id", "default"))
    if client_id in client_buffers:
        client_buffers[client_id].clear()
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)