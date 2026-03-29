"""
STEP 3 — LIVE LIP READING TEST
================================
Real‑time inference using webcam and the trained lip model.
Displays predicted word with confidence.

USAGE
  python lip_test.py
    python lip_test.py --arch bilstm
    python lip_test.py --arch 3dcnn

CONTROLS
  Q — quit
  R — reset state
  S — toggle landmark overlay
  F — fullscreen
"""

import argparse
import json
import os
import time
from collections import deque

import cv2
import joblib
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# MediaPipe FaceMesh indices for lips (same as collection)
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312
]

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH  = "lip_model.h5"
SCALER_PATH = "lip_scaler.pkl"
LABELS_PATH = "lip_labels.json"

CROP_H = 64
CROP_W = 96
CROP_PAD = 22
DISPLAY_LIP_ONLY = True

ARCH_ALIASES = {
    "lstm": "lstm",
    "bilstm": "bilstm",
    "bilstem": "bilstm",
    "3dcnn": "3dcnn",
    "3d-cnn": "3dcnn",
    "cnn3d": "3dcnn",
}

SEQUENCE_LENGTH = 30
THRESHOLD       = 0.85

# Motion gating (optional – you can adapt from sign version)
MOTION_THRESHOLD   = 0.02
STILLNESS_FRAMES   = 8
MIN_CAPTURE_FRAMES = 15
COOLDOWN_SECONDS   = 1.2

MP_COMPLEXITY = 0
MIN_CONF      = 0.5
WINDOW_NAME   = "Step 3 — Live Lip Reading"
START_FULLSCREEN = True
# ─────────────────────────────────────────────

# States (same as sign version)
IDLE, CAPTURING, PREDICTING, COOLDOWN = "IDLE", "CAPTURING", "PREDICTING", "COOLDOWN"
STATE_COLORS = {
    IDLE:       (100, 100, 100),
    CAPTURING:  (0, 165, 255),
    PREDICTING: (0, 220, 220),
    COOLDOWN:   (180, 80, 240),
}

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ═══════════════════════════════════════════
# NORMALISATION (identical to training)
# ═══════════════════════════════════════════
def normalise_lip_frame(frame: np.ndarray) -> np.ndarray:
    pts = frame.reshape(30, 3).copy()
    centre = pts.mean(axis=0)
    pts -= centre
    width = pts[:, 0].max() - pts[:, 0].min()
    if width > 1e-6:
        pts /= width
    return pts.flatten().astype(np.float32)


def extract_lip_landmarks(face_landmarks) -> np.ndarray:
    if face_landmarks is None:
        return np.zeros(90, dtype=np.float32)
    pts = []
    for idx in LIP_INDICES:
        lm = face_landmarks.landmark[idx]
        pts.extend([lm.x, lm.y, lm.z])
    raw = np.array(pts, dtype=np.float32)
    # Apply normalisation immediately
    return normalise_lip_frame(raw)


def extract_lip_crop(frame: np.ndarray, face_landmarks) -> np.ndarray:
    empty = np.zeros((CROP_H, CROP_W, 1), dtype=np.float32)
    if face_landmarks is None:
        return empty

    img_h, img_w = frame.shape[:2]
    xs = [face_landmarks.landmark[i].x * img_w for i in LIP_INDICES]
    ys = [face_landmarks.landmark[i].y * img_h for i in LIP_INDICES]

    x1 = max(int(min(xs)) - CROP_PAD, 0)
    y1 = max(int(min(ys)) - CROP_PAD, 0)
    x2 = min(int(max(xs)) + CROP_PAD, img_w)
    y2 = min(int(max(ys)) + CROP_PAD, img_h)

    if x2 <= x1 or y2 <= y1:
        return empty

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return empty

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (CROP_W, CROP_H)).astype(np.float32)
    return (resized / 255.0)[:, :, np.newaxis]


def _lip_bbox(face_landmarks, img_w: int, img_h: int, pad: int = CROP_PAD):
    if face_landmarks is None:
        return None

    xs = [face_landmarks.landmark[i].x * img_w for i in LIP_INDICES]
    ys = [face_landmarks.landmark[i].y * img_h for i in LIP_INDICES]

    x1 = max(int(min(xs)) - pad, 0)
    y1 = max(int(min(ys)) - pad, 0)
    x2 = min(int(max(xs)) + pad, img_w)
    y2 = min(int(max(ys)) + pad, img_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def extract_lip_view(frame: np.ndarray, face_landmarks) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    img_h, img_w = frame.shape[:2]
    bbox = _lip_bbox(face_landmarks, img_w, img_h, pad=int(CROP_PAD * 2))
    if bbox is None:
        return None, None

    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    view = cv2.resize(crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    return view, bbox


def draw_lip_overlay_on_view(view: np.ndarray, face_landmarks, bbox):
    if face_landmarks is None or bbox is None:
        return

    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    h, w = view.shape[:2]

    for idx in LIP_INDICES:
        lm = face_landmarks.landmark[idx]
        px = int(((lm.x * w - x1) / bw) * w)
        py = int(((lm.y * h - y1) / bh) * h)
        cv2.circle(view, (px, py), 2, (0, 220, 220), -1)


def scale_sequence(seq: np.ndarray, scaler) -> np.ndarray:
    flat = seq.reshape(-1, 90)
    scaled = scaler.transform(flat)
    return scaled.reshape(SEQUENCE_LENGTH, 90)


def normalise_arch_name(arch: str) -> str:
    key = arch.strip().lower()
    if key not in ARCH_ALIASES:
        valid = ", ".join(sorted(ARCH_ALIASES))
        raise ValueError(f"Unknown architecture '{arch}'. Use one of: {valid}")
    return ARCH_ALIASES[key]


def get_artifact_paths(arch: str) -> tuple[str, str, str]:
    if arch == "lstm":
        return MODEL_PATH, SCALER_PATH, LABELS_PATH
    prefix = f"lip_{arch}"
    return f"{prefix}_model.h5", f"{prefix}_scaler.pkl", f"{prefix}_labels.json"


def required_artifact_paths(arch: str) -> list[str]:
    model_path, scaler_path, labels_path = get_artifact_paths(arch)
    required = [model_path, labels_path]
    if arch in {"lstm", "bilstm"}:
        required.append(scaler_path)
    return required


def resolve_available_arch(requested_arch: str) -> tuple[str, str, str, str]:
    candidates = [requested_arch, "bilstm", "lstm", "3dcnn"]
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for arch in ordered:
        model_path, scaler_path, labels_path = get_artifact_paths(arch)
        missing = [p for p in required_artifact_paths(arch) if not os.path.exists(p)]
        if not missing:
            return arch, model_path, scaler_path, labels_path

    # Return requested paths so caller can show exact missing files.
    model_path, scaler_path, labels_path = get_artifact_paths(requested_arch)
    return requested_arch, model_path, scaler_path, labels_path


def prepare_model_input(seq: np.ndarray, model) -> np.ndarray:
    model_shape = tuple(model.input_shape)
    if model_shape == (None, SEQUENCE_LENGTH, 90):
        return seq.reshape(1, SEQUENCE_LENGTH, 90)
    if model_shape == (None, SEQUENCE_LENGTH, 30, 3, 1):
        return seq.reshape(1, SEQUENCE_LENGTH, 30, 3, 1)
    if len(model_shape) == 5 and model_shape[1] == SEQUENCE_LENGTH and model_shape[-1] == 1:
        return seq.reshape((1,) + tuple(seq.shape))
    raise ValueError(f"Unsupported model input shape: {model_shape}")


def compute_motion(prev: np.ndarray | None, curr: np.ndarray) -> float:
    if prev is None:
        return 0.0
    return float(np.linalg.norm(curr - prev))


# ═══════════════════════════════════════════
# UI
# ═══════════════════════════════════════════
def draw_state_bar(frame, state, capture_len):
    h, w = frame.shape[:2]
    color = STATE_COLORS[state]
    label = {
        IDLE:       "IDLE — move your lips to begin",
        CAPTURING:  f"CAPTURING  {capture_len}/{SEQUENCE_LENGTH}",
        PREDICTING: "PREDICTING...",
        COOLDOWN:   "COOLDOWN",
    }[state]

    cv2.rectangle(frame, (0, h-100), (w, h-80), (0,0,0), -1)
    cv2.putText(frame, label, (12, h-85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    if state == CAPTURING and capture_len > 0:
        prog = capture_len / SEQUENCE_LENGTH
        cv2.rectangle(frame, (0, h-80), (w, h-72), (40,40,40), -1)
        cv2.rectangle(frame, (0, h-80), (int(w*prog), h-72), color, -1)


def draw_prediction_box(frame, prediction, confidence):
    h, w = frame.shape[:2]
    color = (0, 220, 80) if confidence > 0.90 else (0, 165, 255)
    cv2.rectangle(frame, (0, h-70), (w, h), (0,0,0), -1)
    cv2.rectangle(frame, (0, h-8), (int(w*confidence), h), color, -1)
    cv2.putText(frame, prediction.upper(),
                (12, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (w-130, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def draw_top_probs(frame, probs, actions, top_n=3):
    top_idx = np.argsort(probs)[::-1][:top_n]
    cv2.rectangle(frame, (0, 0), (260, 30+top_n*35), (0,0,0), -1)
    cv2.putText(frame, "Top Predictions:", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
    for i, idx in enumerate(top_idx):
        y = 50 + i*35
        prob = probs[idx]
        label = actions[idx]
        cv2.rectangle(frame, (5, y-15), (255, y+8), (50,50,50), -1)
        bar = int(250*prob)
        c = (0,220,80) if i==0 else (0,180,200) if i==1 else (100,100,255)
        cv2.rectangle(frame, (5, y-15), (5+bar, y+8), c, -1)
        cv2.putText(frame, f"{label}: {prob*100:.1f}%",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)


def init_preview_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    fullscreen = False
    if START_FULLSCREEN:
        try:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            fullscreen = True
        except cv2.error:
            pass
    return fullscreen


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main(arch: str):
    requested_arch = normalise_arch_name(arch)
    arch, model_path, scaler_path, labels_path = resolve_available_arch(requested_arch)
    if arch != requested_arch:
        print(f"⚠️  Requested arch '{requested_arch}' artifacts are incomplete.")
        print(f"   Auto-selected available arch: '{arch}'.")

    # Check artifacts
    required_paths = required_artifact_paths(arch)

    for path in required_paths:
        if not os.path.exists(path):
            print(f"❌  Missing: {path}\n   Run lip_train.py first.")
            return

    model = load_model(model_path)
    scaler = joblib.load(scaler_path) if arch in {"lstm", "bilstm"} else None
    with open(labels_path, encoding="utf-8") as f:
        actions = json.load(f)
    actions = np.array(actions)

    print("=" * 60)
    print("👄  LIVE LIP READING — Motion‑Gated")
    print("=" * 60)
    print(f"   Model   : {model_path}")
    print(f"   Arch    : {arch}")
    print(f"   Classes : {list(actions)}")
    print(f"   Threshold: {THRESHOLD}")
    print("\n   Q: quit  |  R: reset  |  S: toggle landmarks  |  F: fullscreen")
    print("=" * 60 + "\n")

    state = IDLE
    sequence = []
    prev_kp = None
    stillness_counter = 0
    last_pred_time = 0.0
    current_prediction = ""
    current_confidence = 0.0
    last_probs = None
    motion_val = 0.0
    show_landmarks = True
    fps_times = deque(maxlen=30)
    is_fullscreen = init_preview_window()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=MIN_CONF,
        min_tracking_confidence=MIN_CONF,
    ) as face_mesh:

        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            face_detected = results.multi_face_landmarks is not None

            # Extract keypoints (normalised already inside extract_lip_landmarks)
            face_lm = results.multi_face_landmarks[0] if face_detected else None
            kp = extract_lip_landmarks(face_lm)
            motion_val = compute_motion(prev_kp, kp)
            prev_kp = kp.copy()
            now = time.time()

            if DISPLAY_LIP_ONLY:
                lip_view, lip_bbox = extract_lip_view(frame, face_lm)
                if lip_view is not None:
                    frame = lip_view
                    if show_landmarks:
                        draw_lip_overlay_on_view(frame, face_lm, lip_bbox)
                else:
                    frame = np.zeros_like(frame)
            elif show_landmarks and face_detected:
                # In full-frame mode, draw only lip contours (not full face mesh).
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lm,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

            sample = extract_lip_crop(frame, face_lm) if arch == "3dcnn" else kp

            # State machine (same as sign version)
            if state == IDLE:
                if face_detected and motion_val > MOTION_THRESHOLD:
                    state = CAPTURING
                    sequence = [sample]
                    stillness_counter = 0
                    print("▶  Capturing...")

            elif state == CAPTURING:
                sequence.append(sample)
                stillness_counter = 0 if motion_val >= MOTION_THRESHOLD else stillness_counter + 1

                buffer_full = len(sequence) >= SEQUENCE_LENGTH
                signer_stopped = (stillness_counter >= STILLNESS_FRAMES and
                                  len(sequence) >= MIN_CAPTURE_FRAMES)

                if buffer_full or signer_stopped:
                    state = PREDICTING

            elif state == PREDICTING:
                seq_arr = np.array(sequence, dtype=np.float32)
                if len(seq_arr) < SEQUENCE_LENGTH:
                    pad = np.repeat(seq_arr[-1][np.newaxis, ...], SEQUENCE_LENGTH - len(seq_arr), axis=0)
                    seq_arr = np.concatenate([seq_arr, pad], axis=0)
                else:
                    seq_arr = seq_arr[:SEQUENCE_LENGTH]

                if arch == "3dcnn":
                    inp = prepare_model_input(seq_arr, model)
                else:
                    inp = prepare_model_input(scale_sequence(seq_arr, scaler), model)
                preds = model.predict(inp, verbose=0)[0]
                last_probs = preds
                pred_class = int(np.argmax(preds))
                confidence = float(preds[pred_class])

                if confidence >= THRESHOLD:
                    current_prediction = actions[pred_class]
                    current_confidence = confidence
                    last_pred_time = now
                    print(f"✅  {current_prediction}  ({confidence*100:.1f}%)")
                else:
                    print(f"⚠️  Low conf {confidence*100:.1f}% — discarded")

                sequence = []
                stillness_counter = 0
                state = COOLDOWN

            elif state == COOLDOWN:
                if now - last_pred_time >= COOLDOWN_SECONDS:
                    state = IDLE
                    print("⏸  Ready")

            # UI
            face_color = (0, 220, 80) if face_detected else (0, 50, 220)
            cv2.putText(frame, f"Face: {'✓' if face_detected else '✗'}",
                        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

            # Motion meter (optional)
            mx, my = frame.shape[1] - 165, 55
            fill = min(int(150 * motion_val / (MOTION_THRESHOLD * 3)), 150)
            color = (0, 220, 80) if motion_val > MOTION_THRESHOLD else (80, 80, 200)
            cv2.rectangle(frame, (mx, my), (mx+150, my+14), (40,40,40), -1)
            cv2.rectangle(frame, (mx, my), (mx+fill, my+14), color, -1)
            cv2.line(frame, (mx+50, my), (mx+50, my+14), (0,220,220), 2)
            cv2.putText(frame, f"Motion {motion_val:.3f}", (mx, my-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)

            draw_state_bar(frame, state, len(sequence))
            if last_probs is not None:
                draw_top_probs(frame, last_probs, actions)
            if current_prediction:
                draw_prediction_box(frame, current_prediction, current_confidence)

            fps_times.append(time.time() - t0)
            fps = 1.0 / np.mean(fps_times) if fps_times else 0
            cv2.putText(frame, f"FPS: {fps:.1f}  [{state}]",
                        (frame.shape[1]-230, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋  Quit")
                break
            elif key == ord('r'):
                state = IDLE
                sequence = []
                current_prediction = ""
                last_probs = None
                stillness_counter = 0
                print("🔄  Reset → IDLE")
            elif key == ord('s'):
                show_landmarks = not show_landmarks
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                try:
                    cv2.setWindowProperty(WINDOW_NAME,
                                          cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
                except cv2.error:
                    pass

    cap.release()
    cv2.destroyAllWindows()
    print("✅  Session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Lip Reading Test")
    parser.add_argument("--arch", default="lstm", help="Model architecture: lstm, bilstm, bilstem, 3dcnn")
    args = parser.parse_args()
    main(arch=args.arch)