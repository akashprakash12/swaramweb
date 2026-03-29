"""
STEP 3 — DESKTOP LIVE TEST
============================
Motion-gated real-time inference using webcam.
Tests model.h5 + scaler.pkl before deploying to Android.

STATE MACHINE
  IDLE       → wait for hand motion
  CAPTURING  → collect frames until stillness or 30 frames
  PREDICTING → run model once on clean window
  COOLDOWN   → pause 1.2s before listening again

USAGE
  python step3_test_desktop.py

CONTROLS
  Q — quit
  R — reset to IDLE
  S — toggle landmark overlay
    F — toggle fullscreen
"""

import json
import os
import time
from collections import deque

import cv2
import joblib
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "Incompatible mediapipe package detected. This project requires the classic "
        "MediaPipe Solutions API (mp.solutions).*\n"
        "Use this project environment instead:\n"
        "  ./signenv/bin/python step3_test_desktop.py\n"
        "Or reinstall pinned dependencies:\n"
        "  ./signenv/bin/pip install --upgrade --force-reinstall -r requirements.txt"
    )

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH  = "model.h5"
SCALER_PATH = "scaler.pkl"
LABELS_PATH = "labels.json"

SEQUENCE_LENGTH = 30
THRESHOLD       = 0.85        # minimum confidence to accept prediction

# Motion gating
MOTION_THRESHOLD   = 0.025    # raise if triggering on stillness
STILLNESS_FRAMES   = 8        # frames of no-motion before capture ends
MIN_CAPTURE_FRAMES = 15       # reject captures shorter than this
COOLDOWN_SECONDS   = 1.2

MP_COMPLEXITY = 0             # 0 = faster; 1 = more accurate
MIN_CONF      = 0.5
WINDOW_NAME   = "Step 3 — Live Test"
START_FULLSCREEN = True
# ─────────────────────────────────────────────

# States
IDLE       = "IDLE"
CAPTURING  = "CAPTURING"
PREDICTING = "PREDICTING"
COOLDOWN   = "COOLDOWN"

STATE_COLORS = {
    IDLE:       (100, 100, 100),
    CAPTURING:  (0, 165, 255),
    PREDICTING: (0, 220, 220),
    COOLDOWN:   (180, 80, 240),
}

mp_hands          = mp.solutions.hands
mp_pose           = mp.solutions.pose
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ═══════════════════════════════════════════
# NORMALIZATION  — identical to step2_train.py
# ═══════════════════════════════════════════
def normalize_hand(hand_flat: np.ndarray) -> np.ndarray:
    pts   = hand_flat.reshape(21, 3).copy()
    wrist = pts[0].copy()
    pts  -= wrist
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().astype(np.float32)


def normalize_pose(pose_flat: np.ndarray) -> np.ndarray:
    pts            = pose_flat.reshape(33, 3).copy()
    l_sh           = pts[11].copy()
    r_sh           = pts[12].copy()
    midpoint       = (l_sh + r_sh) / 2.0
    pts           -= midpoint
    width          = np.linalg.norm(l_sh - r_sh)
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
            raw   = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                             dtype=np.float32).flatten()
            label = handedness.classification[0].label
            norm  = normalize_hand(raw)
            if label == "Left":
                lh = norm
            else:
                rh = norm

    pose = np.zeros(99, dtype=np.float32)
    if pose_results.pose_landmarks:
        raw_pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        pose = normalize_pose(raw_pose)

    return np.concatenate([lh, rh, pose])  # (225,)


def scale_sequence(seq: np.ndarray, scaler) -> np.ndarray:
    """(30, 225) → (1, 30, 225) scaled"""
    flat   = seq.reshape(-1, 225)
    scaled = scaler.transform(flat)
    return scaled.reshape(1, 30, 225)


def compute_motion(prev: np.ndarray | None, curr: np.ndarray) -> float:
    if prev is None:
        return 0.0
    return float(np.linalg.norm(curr[:126] - prev[:126]))  # hands only


# ═══════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════
def draw_state_bar(frame, state, capture_len):
    h, w  = frame.shape[:2]
    color = STATE_COLORS[state]
    label = {
        IDLE:       "IDLE — move your hands to begin",
        CAPTURING:  f"CAPTURING  {capture_len}/{SEQUENCE_LENGTH}",
        PREDICTING: "PREDICTING...",
        COOLDOWN:   "COOLDOWN",
    }[state]

    cv2.rectangle(frame, (0, h - 100), (w, h - 80), (0, 0, 0), -1)
    cv2.putText(frame, label, (12, h - 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    if state == CAPTURING and capture_len > 0:
        prog = capture_len / SEQUENCE_LENGTH
        cv2.rectangle(frame, (0, h - 80), (w, h - 72), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 80), (int(w * prog), h - 72), color, -1)


def draw_prediction_box(frame, prediction, confidence):
    h, w  = frame.shape[:2]
    color = (0, 220, 80) if confidence > 0.90 else (0, 165, 255)
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h - 8), (int(w * confidence), h), color, -1)
    cv2.putText(frame, prediction.upper(),
                (12, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (w - 130, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def draw_top_probs(frame, probs, actions, top_n=3):
    top_idx = np.argsort(probs)[::-1][:top_n]
    cv2.rectangle(frame, (0, 0), (260, 30 + top_n * 35), (0, 0, 0), -1)
    cv2.putText(frame, "Top Predictions:", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    for i, idx in enumerate(top_idx):
        y     = 50 + i * 35
        prob  = probs[idx]
        label = actions[idx]
        cv2.rectangle(frame, (5, y - 15), (255, y + 8), (50, 50, 50), -1)
        bar   = int(250 * prob)
        c     = (0, 220, 80) if i == 0 else (0, 180, 200) if i == 1 else (100, 100, 255)
        cv2.rectangle(frame, (5, y - 15), (5 + bar, y + 8), c, -1)
        cv2.putText(frame, f"{label}: {prob*100:.1f}%",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def draw_motion_meter(frame, motion_val):
    h, w    = frame.shape[:2]
    mx, my  = w - 165, 55
    fill    = min(int(150 * motion_val / (MOTION_THRESHOLD * 3)), 150)
    color   = (0, 220, 80) if motion_val > MOTION_THRESHOLD else (80, 80, 200)
    cv2.rectangle(frame, (mx, my), (mx + 150, my + 14), (40, 40, 40), -1)
    cv2.rectangle(frame, (mx, my), (mx + fill, my + 14), color, -1)
    thresh_x = mx + 50
    cv2.line(frame, (thresh_x, my), (thresh_x, my + 14), (0, 220, 220), 2)
    cv2.putText(frame, f"Motion {motion_val:.3f}", (mx, my - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


def init_preview_window() -> bool:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    fullscreen = False
    if START_FULLSCREEN:
        try:
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            fullscreen = True
        except cv2.error:
            try:
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL,
                )
            except cv2.error:
                pass

    return fullscreen


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    # ── load artifacts ──────────────────────
    for path, name in [(MODEL_PATH, "model.h5"), (SCALER_PATH, "scaler.pkl"),
                       (LABELS_PATH, "labels.json")]:
        if not os.path.exists(path):
            print(f"❌  Missing: {path}")
            print(f"   Run step2_train.py first.")
            return

    model   = load_model(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        actions = json.load(f)
    actions = np.array(actions)

    print("=" * 60)
    print("📷  DESKTOP LIVE TEST — Motion-Gated")
    print("=" * 60)
    print(f"   Model   : {MODEL_PATH}")
    print(f"   Classes : {list(actions)}")
    print(f"   Threshold: {THRESHOLD}")
    print(f"\n   Q: quit  |  R: reset  |  S: toggle landmarks  |  F: fullscreen")
    print("=" * 60 + "\n")

    # ── state machine vars ──────────────────
    state              = IDLE
    sequence: list     = []
    prev_kp            = None
    stillness_counter  = 0
    last_pred_time     = 0.0
    current_prediction = ""
    current_confidence = 0.0
    last_probs         = None
    motion_val         = 0.0
    show_landmarks     = True
    fps_times          = deque(maxlen=30)
    is_fullscreen      = init_preview_window()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(
        model_complexity=MP_COMPLEXITY,
        min_detection_confidence=MIN_CONF,
        min_tracking_confidence=MIN_CONF,
        max_num_hands=2,
    ) as hands, mp_pose.Pose(
        model_complexity=MP_COMPLEXITY,
        min_detection_confidence=MIN_CONF,
        min_tracking_confidence=MIN_CONF,
    ) as pose:

        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            hr = hands.process(img_rgb)
            pr = pose.process(img_rgb)
            img_rgb.flags.writeable = True

            if show_landmarks:
                if hr.multi_hand_landmarks:
                    for hl in hr.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hl, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
                if pr.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

            kp         = extract_keypoints(hr, pr)
            motion_val = compute_motion(prev_kp, kp)
            prev_kp    = kp.copy()
            hands_det  = hr.multi_hand_landmarks is not None
            now        = time.time()

            # ── state machine ───────────────
            if state == IDLE:
                if hands_det and motion_val > MOTION_THRESHOLD:
                    state             = CAPTURING
                    sequence          = [kp]
                    stillness_counter = 0
                    print("▶  Capturing...")

            elif state == CAPTURING:
                sequence.append(kp)
                stillness_counter = 0 if motion_val >= MOTION_THRESHOLD else stillness_counter + 1

                buffer_full    = len(sequence) >= SEQUENCE_LENGTH
                signer_stopped = (stillness_counter >= STILLNESS_FRAMES
                                  and len(sequence) >= MIN_CAPTURE_FRAMES)

                if buffer_full or signer_stopped:
                    state = PREDICTING

            elif state == PREDICTING:
                seq_arr = np.array(sequence, dtype=np.float32)
                if len(seq_arr) < SEQUENCE_LENGTH:
                    pad     = np.tile(seq_arr[-1], (SEQUENCE_LENGTH - len(seq_arr), 1))
                    seq_arr = np.vstack([seq_arr, pad])
                else:
                    seq_arr = seq_arr[:SEQUENCE_LENGTH]

                inp         = scale_sequence(seq_arr, scaler)
                preds       = model.predict(inp, verbose=0)[0]
                last_probs  = preds
                pred_class  = int(np.argmax(preds))
                confidence  = float(preds[pred_class])

                if confidence >= THRESHOLD:
                    current_prediction = actions[pred_class]
                    current_confidence = confidence
                    last_pred_time     = now
                    print(f"✅  {current_prediction}  ({confidence*100:.1f}%)")
                else:
                    print(f"⚠️  Low conf {confidence*100:.1f}% — discarded "
                          f"[top: {actions[pred_class]}]")

                sequence          = []
                stillness_counter = 0
                state             = COOLDOWN

            elif state == COOLDOWN:
                if now - last_pred_time >= COOLDOWN_SECONDS:
                    state = IDLE
                    print("⏸  Ready")

            # ── UI ──────────────────────────
            h_color = (0, 220, 80) if hands_det else (0, 50, 220)
            cv2.putText(frame,
                        f"Hands: {'✓' if hands_det else '✗'}  "
                        f"Pose: {'✓' if pr.pose_landmarks else '✗'}",
                        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_color, 2)

            draw_motion_meter(frame, motion_val)
            draw_state_bar(frame, state, len(sequence))
            if last_probs is not None:
                draw_top_probs(frame, last_probs, actions)
            if current_prediction:
                draw_prediction_box(frame, current_prediction, current_confidence)

            fps_times.append(time.time() - t0)
            fps = 1.0 / np.mean(fps_times) if fps_times else 0
            cv2.putText(frame, f"FPS: {fps:.1f}  [{state}]",
                        (frame.shape[1] - 230, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n👋  Quit")
                break
            elif key == ord("r"):
                state = IDLE; sequence = []; current_prediction = ""
                last_probs = None; stillness_counter = 0
                print("🔄  Reset → IDLE")
            elif key == ord("s"):
                show_landmarks = not show_landmarks
            elif key == ord("f"):
                is_fullscreen = not is_fullscreen
                try:
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
                    )
                except cv2.error:
                    pass

    cap.release()
    cv2.destroyAllWindows()
    print("✅  Session ended.")
    print("\n▶   If accuracy is good → deploy model.tflite + scaler.json + labels.json to Android\n")


if __name__ == "__main__":
    main()
