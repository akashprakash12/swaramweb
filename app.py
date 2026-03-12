import base64
import json
import os
from collections import deque
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

MODEL_PATH = "model.tflite"
SCALER_JSON = "scaler.json"
LABELS_JSON = "labels.json"
SEQUENCE_LEN = 30
N_FEATURES = 225
THRESHOLD = 0.85
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

app = Flask(__name__)

# Keep a rolling frame buffer per browser client.
client_buffers: Dict[str, deque] = {}

with open(SCALER_JSON, "r", encoding="utf-8") as f:
    scaler_data = json.load(f)
mean = np.array(scaler_data["mean"], dtype=np.float32)
scale = np.array(scaler_data["scale"], dtype=np.float32)
scale = np.where(scale == 0.0, 1.0, scale)

with open(LABELS_JSON, "r", encoding="utf-8") as f:
    labels = np.array(json.load(f))

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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


def decode_image(data_url: str) -> np.ndarray:
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Invalid image data")
    # Reduce CPU load on hosted instances while preserving landmarks quality.
    frame_bgr = cv2.resize(frame_bgr, (PROCESS_WIDTH, PROCESS_HEIGHT))
    return frame_bgr


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    print(f"DEBUG: Received payload keys: {payload.keys()}")
    print(f"DEBUG: Image data length: {len(payload.get('image', ''))}")
    client_id = str(payload.get("client_id", "default"))
    image_data = payload.get("image")

    if not image_data:
        print(f"DEBUG: Missing image. Client: {client_id}")
        return jsonify({"error": "Missing image"}), 400
    
    if not isinstance(image_data, str) or len(image_data) < 100:
        print(f"DEBUG: Invalid image data format or too small")
        return jsonify({"error": "Invalid image data"}), 400

    try:
        frame_bgr = decode_image(image_data)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        hr = hands.process(frame_rgb)
        pr = pose.process(frame_rgb)
        kp = extract_keypoints(hr, pr)

        if client_id not in client_buffers:
            client_buffers[client_id] = deque(maxlen=SEQUENCE_LEN)
        client_buffers[client_id].append(kp)

        buffer = client_buffers[client_id]
        if len(buffer) < SEQUENCE_LEN:
            print(f"DEBUG: Buffer status - {len(buffer)}/{SEQUENCE_LEN} frames")
            return jsonify(
                {
                    "status": "collecting",
                    "frames": len(buffer),
                    "needed": SEQUENCE_LEN,
                }
            )

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

        response = {
            "status": "ok",
            "prediction": str(labels[pred_idx]),
            "confidence": confidence,
            "accepted": confidence >= THRESHOLD,
            "top_predictions": top_predictions,
        }
        print(f"DEBUG: Prediction - {response['prediction']} ({response['confidence']:.2f})")
        return jsonify(response)
    except Exception as exc:
        print(f"DEBUG: Exception in predict - {str(exc)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/reset")
def reset():
    payload = request.get_json(silent=True) or {}
    client_id = str(payload.get("client_id", "default"))
    client_buffers[client_id] = deque(maxlen=SEQUENCE_LEN)
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    # Allow overriding the port to avoid conflicts when 5000 is already in use.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
