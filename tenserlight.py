"""
Live test of TFLite model with OpenCV + MediaPipe.
Predicts sign language from webcam in real time.
"""

import json
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ------------------------------
# CONFIGURATION
# ------------------------------
MODEL_PATH    = "model.tflite"
SCALER_JSON   = "scaler.json"
LABELS_JSON   = "labels.json"
SEQUENCE_LEN  = 30
N_FEATURES    = 225
THRESHOLD     = 0.85          # minimum confidence to show result

# MediaPipe settings
MP_COMPLEXITY = 0
MIN_CONF      = 0.5

# ------------------------------
# Load TFLite model
# ------------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# Load scaler parameters
# ------------------------------
with open(SCALER_JSON, "r") as f:
    scaler_data = json.load(f)
mean  = np.array(scaler_data["mean"], dtype=np.float32)
scale = np.array(scaler_data["scale"], dtype=np.float32)

# ------------------------------
# Load labels
# ------------------------------
with open(LABELS_JSON, "r") as f:
    labels = json.load(f)

# ------------------------------
# Normalisation functions (identical to training)
# ------------------------------
def normalize_hand(hand_flat):
    pts = hand_flat.reshape(21, 3).copy()
    wrist = pts[0].copy()
    pts -= wrist
    scale_norm = np.linalg.norm(pts[9])
    if scale_norm > 1e-6:
        pts /= scale_norm
    return pts.flatten().astype(np.float32)

def normalize_pose(pose_flat):
    pts = pose_flat.reshape(33, 3).copy()
    l_sh = pts[11].copy()
    r_sh = pts[12].copy()
    midpoint = (l_sh + r_sh) / 2.0
    pts -= midpoint
    width = np.linalg.norm(l_sh - r_sh)
    if width > 1e-6:
        pts /= width
    return pts.flatten().astype(np.float32)

def extract_keypoints(hand_results, pose_results):
    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for landmarks, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness,
        ):
            raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
            label = handedness.classification[0].label
            norm = normalize_hand(raw)
            if label == "Left":
                lh = norm
            else:
                rh = norm

    pose = np.zeros(99, dtype=np.float32)
    if pose_results.pose_landmarks:
        raw_pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]).flatten()
        pose = normalize_pose(raw_pose)

    return np.concatenate([lh, rh, pose])   # (225,)

# ------------------------------
# Initialise MediaPipe
# ------------------------------
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity=MP_COMPLEXITY,
    min_detection_confidence=MIN_CONF,
    min_tracking_confidence=MIN_CONF,
    max_num_hands=2,
)
pose = mp_pose.Pose(
    model_complexity=MP_COMPLEXITY,
    min_detection_confidence=MIN_CONF,
    min_tracking_confidence=MIN_CONF,
)

# ------------------------------
# Open webcam
# ------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Buffer to hold a sliding window of frames (for sequence)
buffer = []

print("Starting live TFLite test. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    hr = hands.process(rgb)
    pr = pose.process(rgb)

    # Draw landmarks
    if hr.multi_hand_landmarks:
        for hl in hr.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style()
            )
    if pr.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw_styles.get_default_pose_landmarks_style()
        )

    # Extract keypoints for this frame
    kp = extract_keypoints(hr, pr)

    # Maintain a buffer of the last SEQUENCE_LEN frames
    buffer.append(kp)
    if len(buffer) > SEQUENCE_LEN:
        buffer.pop(0)

    # If we have enough frames, run inference
    if len(buffer) == SEQUENCE_LEN:
        # Stack into (30, 225)
        seq = np.stack(buffer, axis=0).astype(np.float32)

        # Scale using saved mean/scale
        flat = seq.reshape(-1, N_FEATURES)
        scaled = (flat - mean) / scale
        scaled_seq = scaled.reshape(1, SEQUENCE_LEN, N_FEATURES).astype(np.float32)

        # Run TFLite inference
        interpreter.set_tensor(input_details[0]['index'], scaled_seq)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get top prediction
        pred_class = int(np.argmax(output))
        confidence = float(output[pred_class])

        if confidence >= THRESHOLD:
            label = labels[pred_class]
            # Display on frame
            cv2.putText(frame, f"{label} ({confidence:.2f})", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Show frame
    cv2.imshow("TFLite Live Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()