"""
STEP 1 — LIP READING DATA COLLECTION  (English — Enhanced)
=============================================================
Collects DUAL data per recording:
  • Landmark file  : (30, 90)  .npy  — 30 lip landmarks × (x,y,z)
  • Pixel-crop file: (30, 64, 96, 1) .npy — grayscale lip-region video
    → feeds 3D-CNN branch; landmark file feeds BiLSTM branch.

NEW vs original
───────────────
  ✦ Dual output (landmarks + pixel crops) for 3D CNN + BiLSTM pipeline
  ✦ Larger English word list (common + AAC / assistive phrases)
  ✦ 100 samples per word target (was 30)
  ✦ Talking-head guide overlay (nose-tip alignment box)
  ✦ Mouth-openness meter — warns if lips not moving enough
  ✦ Per-word speaking-speed variation prompt (slow / normal / fast)
  ✦ Live quality score shown during recording
  ✦ Auto-advance when word hits target
  ✦ Session CSV log (timestamp, word, quality, speed_label)
  ✦ Dataset integrity checker (I key)
  ✦ Brightness / contrast warning
  ✦ All original controls preserved (SPACE N P R Q)

USAGE
  python lip_collect.py
  python lip_collect.py --words "Hello,Goodbye,Thanks" --samples 60
  python lip_collect.py --cam 1 --samples 100

CONTROLS
  SPACE : start recording
  N     : next word
  P     : previous word
  R     : delete last recording
  I     : integrity check (print dataset stats)
  Q     : quit
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────────────────────
# LIP LANDMARK INDICES  (MediaPipe FaceMesh — 30 points)
# ─────────────────────────────────────────────────────────────
LIP_INDICES = [
    # outer lips (15)
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
    # inner lips (15)
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312,
]  # 30 points total

# Indices used for mouth-openness  (top-inner, bottom-inner)
MOUTH_TOP_IDX    = 13   # upper inner lip centre  (MediaPipe full mesh idx)
MOUTH_BOTTOM_IDX = 14   # lower inner lip centre

# ─────────────────────────────────────────────────────────────
# ENGLISH WORD LIST
# ─────────────────────────────────────────────────────────────
DEFAULT_WORDS = [
    # Greetings & basic social
    "Hello", "Goodbye", "Thanks", "Please", "Sorry", "Welcome",
    "Yes", "No", "Okay",
    # Basic needs
    "Water", "Food", "Help", "Stop", "Come", "Go",
    "More", "Less", "Hot", "Cold",
    # Common adjectives
    "Good", "Bad", "Big", "Small", "Fast", "Slow",
    # Numbers (visually distinct)
    "One", "Two", "Three", "Four", "Five",
    # High-value phrases (single-word proxies)
    "Bathroom", "Doctor", "Pain", "Medicine", "Family",
]

# Speed labels cycled per recording to force variety
SPEED_LABELS  = ["normal", "slow", "fast"]
SPEED_PROMPTS = {
    "normal": "Normal speed",
    "slow":   "Speak  S - L - O - W - L - Y",
    "fast":   "Speak quickly",
}

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR       = "lip_dataset"
SAMPLES_PER_WORD = 30
SEQUENCE_LENGTH  = 30           # frames per recording
COUNTDOWN_SEC    = 3

# Pixel-crop dimensions (for 3D-CNN)
CROP_H = 64
CROP_W = 96
CROP_PAD = 22                   # pixels of padding around lip bbox

# Quality gates
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5
MIN_QUALITY_RATE         = 0.70   # raised from 0.60
MIN_MOUTH_OPENNESS       = 0.010  # normalised; warn if avg below this

# Brightness / contrast
BRIGHTNESS_LOW   = 60
BRIGHTNESS_HIGH  = 200
CONTRAST_LOW     = 25

# Camera
CAM_INDEX    = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
WINDOW_NAME  = "Step 1 — Collect Lip Data (Enhanced)"

# CSV session log
SESSION_LOG = "lip_collection_log.csv"
# ─────────────────────────────────────────────────────────────

mp_face_mesh      = mp.solutions.face_mesh
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_lip_landmarks(face_landmarks) -> np.ndarray:
    """Returns (90,) array: 30 lip landmarks × (x,y,z), or zeros."""
    if face_landmarks is None:
        return np.zeros(90, dtype=np.float32)
    pts = []
    for idx in LIP_INDICES:
        lm = face_landmarks.landmark[idx]
        pts.extend([lm.x, lm.y, lm.z])
    return np.array(pts, dtype=np.float32)


def extract_lip_crop(frame: np.ndarray,
                     face_landmarks,
                     img_w: int,
                     img_h: int) -> np.ndarray:
    """
    Crops lip region from frame using landmark bounding box.
    Returns (CROP_H, CROP_W, 1) uint8 grayscale, or zeros on failure.
    """
    empty = np.zeros((CROP_H, CROP_W, 1), dtype=np.uint8)
    if face_landmarks is None:
        return empty

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
    resized = cv2.resize(gray, (CROP_W, CROP_H))
    return resized[:, :, np.newaxis]   # (H, W, 1)


def mouth_openness(face_landmarks, img_h: int) -> float:
    """
    Returns vertical gap between inner top/bottom lip (normalised by img_h).
    Used to warn if the mouth is not moving enough during recording.
    """
    if face_landmarks is None:
        return 0.0
    top    = face_landmarks.landmark[MOUTH_TOP_IDX].y    * img_h
    bottom = face_landmarks.landmark[MOUTH_BOTTOM_IDX].y * img_h
    return abs(bottom - top) / img_h


def image_quality(frame: np.ndarray) -> tuple[float, float]:
    """Returns (mean_brightness, std_brightness) of the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


# ═══════════════════════════════════════════════════════════════
# COLLECTOR
# ═══════════════════════════════════════════════════════════════

class LipCollector:
    def __init__(self, words: list, samples: int, cam_index: int = CAM_INDEX):
        self.words   = words
        self.samples = samples
        self.word_idx = 0
        self.cam_index = cam_index

        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for w in words:
            os.makedirs(os.path.join(OUTPUT_DIR, w), exist_ok=True)

        # Open camera
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera.")

        # Recording state
        self.recording        = False
        self.countdown_end    = 0.0
        self.recorded_lm      = []    # list of (90,)
        self.recorded_crop    = []    # list of (CROP_H, CROP_W, 1)
        self.recorded_open    = []    # mouth-openness per frame
        self.detection_frames = 0
        self.total_frames     = 0

        # Speed cycling
        self.speed_cycle_idx  = 0

        # Session CSV
        self._init_csv()

        # Cached quality info for display
        self._last_brightness = 128.0
        self._last_contrast   = 50.0
        self._last_openness   = 0.0

    # ── helpers ────────────────────────────────────────────────

    @property
    def current_word(self) -> str:
        return self.words[self.word_idx]

    @property
    def current_speed(self) -> str:
        return SPEED_LABELS[self.speed_cycle_idx % len(SPEED_LABELS)]

    def existing_count(self, word: str = None) -> int:
        word = word or self.current_word
        path = os.path.join(OUTPUT_DIR, word)
        # Count only landmark files (each recording produces two files)
        return len([f for f in os.listdir(path) if f.endswith("_lm.npy")])

    def _init_csv(self):
        if not os.path.exists(SESSION_LOG):
            with open(SESSION_LOG, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "word", "sample_n",
                     "quality_pct", "speed", "openness_avg"]
                )

    def _log_csv(self, word: str, sample_n: int,
                 quality: float, speed: str, openness: float):
        with open(SESSION_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                word, sample_n,
                f"{quality*100:.1f}", speed, f"{openness:.4f}",
            ])

    # ── recording control ───────────────────────────────────────

    def start_recording(self):
        self.recording        = True
        self.countdown_end    = time.time() + COUNTDOWN_SEC
        self.recorded_lm      = []
        self.recorded_crop    = []
        self.recorded_open    = []
        self.detection_frames = 0
        self.total_frames     = 0
        speed = self.current_speed
        print(f"\n⏳  Countdown {COUNTDOWN_SEC}s → '{self.current_word}'  [{speed}]  "
              f"— {SPEED_PROMPTS[speed]}")

    def delete_last(self):
        path  = os.path.join(OUTPUT_DIR, self.current_word)
        files = sorted(f for f in os.listdir(path) if f.endswith("_lm.npy"))
        if not files:
            print("  Nothing to delete.")
            return
        base = files[-1].replace("_lm.npy", "")
        for suffix in ("_lm.npy", "_crop.npy", "_preview.jpg"):
            fp = os.path.join(path, base + suffix)
            if os.path.exists(fp):
                os.remove(fp)
        print(f"  🗑  Deleted: {base}")

    def save_recording(self, frame_for_preview=None) -> bool:
        if not self.recorded_lm:
            return False

        # Quality gate
        rate = (self.detection_frames / self.total_frames
                if self.total_frames else 0)
        if rate < MIN_QUALITY_RATE:
            print(f"  ⚠️  Detection quality {rate*100:.0f}% < "
                  f"{MIN_QUALITY_RATE*100:.0f}% — NOT saved.")
            self.speed_cycle_idx += 1
            return False

        # Mouth-openness gate
        avg_open = float(np.mean(self.recorded_open)) if self.recorded_open else 0.0
        if avg_open < MIN_MOUTH_OPENNESS:
            print(f"  ⚠️  Mouth barely moved (openness {avg_open:.4f}) — NOT saved.")
            self.speed_cycle_idx += 1
            return False

        # ── Landmark sequence ──────────────────────────────────
        seq_lm = np.array(self.recorded_lm, dtype=np.float32)   # (T, 90)
        T = len(seq_lm)
        if T < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - T, 90), dtype=np.float32)
            seq_lm = np.vstack([seq_lm, pad])
        else:
            seq_lm = seq_lm[:SEQUENCE_LENGTH]
        assert seq_lm.shape == (SEQUENCE_LENGTH, 90)

        # ── Pixel-crop sequence ────────────────────────────────
        seq_crop = np.array(self.recorded_crop,
                            dtype=np.uint8)                      # (T, H, W, 1)
        if len(seq_crop) < SEQUENCE_LENGTH:
            pad_c = np.zeros(
                (SEQUENCE_LENGTH - len(seq_crop), CROP_H, CROP_W, 1),
                dtype=np.uint8)
            seq_crop = np.concatenate([seq_crop, pad_c], axis=0)
        else:
            seq_crop = seq_crop[:SEQUENCE_LENGTH]
        assert seq_crop.shape == (SEQUENCE_LENGTH, CROP_H, CROP_W, 1)

        # ── Save ──────────────────────────────────────────────
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        n    = self.existing_count() + 1
        base = f"{self.current_word}_{n:03d}_{ts}"
        path = os.path.join(OUTPUT_DIR, self.current_word)

        np.save(os.path.join(path, base + "_lm.npy"),   seq_lm)
        np.save(os.path.join(path, base + "_crop.npy"), seq_crop)

        if frame_for_preview is not None:
            cv2.imwrite(
                os.path.join(path, base + "_preview.jpg"),
                frame_for_preview
            )

        self._log_csv(self.current_word, n, rate,
                      self.current_speed, avg_open)
        self.speed_cycle_idx += 1

        print(f"  ✅  Saved: {base}  |  "
              f"quality {rate*100:.0f}%  |  "
              f"openness {avg_open:.3f}  |  "
              f"speed: {self.current_speed}  |  "
              f"{n}/{self.samples}")
        return True

    # ── integrity check ─────────────────────────────────────────

    def integrity_check(self):
        print("\n" + "=" * 55)
        print("🔍  DATASET INTEGRITY CHECK")
        print("=" * 55)
        total_lm = total_crop = corrupt = 0
        for w in self.words:
            path  = os.path.join(OUTPUT_DIR, w)
            lm_files   = sorted(f for f in os.listdir(path) if f.endswith("_lm.npy"))
            crop_files  = set(f.replace("_lm.npy","_crop.npy") for f in lm_files)
            actual_crop = set(f for f in os.listdir(path) if f.endswith("_crop.npy"))
            missing = crop_files - actual_crop
            ok = len(lm_files)
            total_lm += ok
            total_crop += len(actual_crop)
            for fname in lm_files:
                fp = os.path.join(path, fname)
                try:
                    arr = np.load(fp)
                    if arr.shape != (SEQUENCE_LENGTH, 90):
                        raise ValueError(f"bad shape {arr.shape}")
                except Exception as e:
                    print(f"  ⚠️  {fname}: {e}")
                    corrupt += 1
            status = "✅" if ok >= self.samples and not missing else "⚠️ "
            print(f"  {status}  {w:<18s}: {ok:3d} lm | "
                  f"{len(actual_crop):3d} crop | "
                  f"{len(missing)} missing crop")
        print(f"\n  Total lm   : {total_lm}")
        print(f"  Total crop : {total_crop}")
        print(f"  Corrupt    : {corrupt}")
        print("=" * 55 + "\n")

    # ── drawing ──────────────────────────────────────────────────

    def _brightness_warning(self, frame, brightness, contrast):
        h, w = frame.shape[:2]
        msgs = []
        if brightness < BRIGHTNESS_LOW:
            msgs.append("⚠️ TOO DARK")
        elif brightness > BRIGHTNESS_HIGH:
            msgs.append("⚠️ TOO BRIGHT")
        if contrast < CONTRAST_LOW:
            msgs.append("⚠️ LOW CONTRAST")
        for i, m in enumerate(msgs):
            cv2.putText(frame, m,
                        (w // 2 - 100, 60 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 220), 2, cv2.LINE_AA)

    def _draw_alignment_box(self, frame, face_landmarks):
        """Draw a guide rectangle showing where to position the face."""
        h, w = frame.shape[:2]
        # Suggested face region: centre third horizontally, middle vertically
        bx1, by1 = w // 3, h // 5
        bx2, by2 = 2 * w // 3, 4 * h // 5
        color = (0, 200, 255)

        if face_landmarks:
            # Check nose tip is roughly in the box
            nose = face_landmarks.landmark[1]
            nx, ny = int(nose.x * w), int(nose.y * h)
            in_box = bx1 < nx < bx2 and by1 < ny < by2
            color = (0, 220, 80) if in_box else (0, 60, 220)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, "Keep face here",
                    (bx1 + 5, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)

    def _draw_openness_meter(self, frame, openness: float):
        h, w = frame.shape[:2]
        mx, my = w - 175, 90
        bar_len = 150
        fill = min(int(bar_len * openness / 0.05), bar_len)
        ok   = openness >= MIN_MOUTH_OPENNESS
        col  = (0, 220, 80) if ok else (0, 80, 220)
        cv2.rectangle(frame, (mx, my), (mx + bar_len, my + 14), (40, 40, 40), -1)
        cv2.rectangle(frame, (mx, my), (mx + fill,    my + 14), col, -1)
        # Minimum threshold line
        thresh_x = mx + int(bar_len * MIN_MOUTH_OPENNESS / 0.05)
        cv2.line(frame, (thresh_x, my), (thresh_x, my + 14), (0, 220, 220), 2)
        cv2.putText(frame, f"Mouth {openness:.3f}",
                    (mx, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    def draw_ui(self, frame, face_detected, face_landmarks):
        h, w = frame.shape[:2]
        word  = self.current_word
        exist = self.existing_count()
        now   = time.time()

        # Top panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 160), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Word + progress
        cv2.putText(frame,
                    f"Word: {word}  ({exist}/{self.samples})",
                    (18, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        bar_x, bar_y, bar_w, bar_h = 18, 55, 420, 24
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill = int(bar_w * min(exist / self.samples, 1.0))
        col  = (0, 220, 120) if exist >= self.samples else (0, 160, 255)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h), col, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        cv2.putText(frame,
                    f"[{self.word_idx+1}/{len(self.words)}]",
                    (18, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

        # Speed prompt
        speed_text = SPEED_PROMPTS[self.current_speed]
        cv2.putText(frame, speed_text,
                    (18, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        # Face indicator
        fc = (0, 220, 80) if face_detected else (0, 50, 220)
        cv2.putText(frame,
                    "✓ FACE" if face_detected else "✗ FACE",
                    (w - 155, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2)

        # Recording overlay
        if self.recording:
            if now < self.countdown_end:
                left = int(self.countdown_end - now) + 1
                cv2.putText(frame, str(left),
                            (w // 2 - 40, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 6,
                            (0, 220, 220), 12)
            else:
                n_rec = len(self.recorded_lm)
                cv2.circle(frame, (w - 45, 175), 18, (0, 0, 220), -1)
                cv2.putText(frame, "REC",
                            (w - 95, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)
                cv2.putText(frame, f"{n_rec}/{SEQUENCE_LENGTH}",
                            (w - 110, 215),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                # live quality
                if self.total_frames > 0:
                    q = self.detection_frames / self.total_frames
                    qc = (0, 220, 80) if q >= MIN_QUALITY_RATE else (0, 60, 220)
                    cv2.putText(frame, f"Q:{q*100:.0f}%",
                                (w - 100, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, qc, 1)

        # Bottom instructions
        bot = frame.copy()
        cv2.rectangle(bot, (0, h - 100), (w, h), (25, 25, 25), -1)
        cv2.addWeighted(bot, 0.75, frame, 0.25, 0, frame)
        lines = [
            "SPACE: Record  |  N: Next  |  P: Prev  |  R: Redo last  |  I: Integrity  |  Q: Quit",
            "Vary speed each recording.  Stay in the guide box.  Speak clearly.",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt,
                        (12, h - 72 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

        # Extra meters
        self._draw_alignment_box(frame, face_landmarks)
        self._draw_openness_meter(frame, self._last_openness)
        self._brightness_warning(frame,
                                 self._last_brightness,
                                 self._last_contrast)

    # ── main loop ────────────────────────────────────────────────

    def run(self):
        print("\n" + "=" * 68)
        print("👄  LIP READING DATA COLLECTOR  —  English Enhanced")
        print("=" * 68)
        print(f"   Words   : {self.words}")
        print(f"   Target  : {self.samples} samples each")
        print(f"   Output  : {os.path.abspath(OUTPUT_DIR)}/")
        print(f"   Dual save: *_lm.npy (90)  +  *_crop.npy ({SEQUENCE_LENGTH},{CROP_H},{CROP_W},1)")
        print("=" * 68 + "\n")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

        mid_frame = None

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as face_mesh:

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                # Image-quality stats (sampled every frame, cheap)
                self._last_brightness, self._last_contrast = image_quality(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                rgb.flags.writeable = True

                face_detected  = results.multi_face_landmarks is not None
                face_landmarks = (results.multi_face_landmarks[0]
                                  if face_detected else None)

                # Mouth openness
                self._last_openness = mouth_openness(face_landmarks, h)

                # Draw lips only (no full face mesh)
                if face_detected:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())

                # ── Recording logic ──────────────────────────────
                if self.recording:
                    now = time.time()
                    if now >= self.countdown_end:
                        lm   = extract_lip_landmarks(face_landmarks)
                        crop = extract_lip_crop(frame, face_landmarks, w, h)

                        self.recorded_lm.append(lm)
                        self.recorded_crop.append(crop)
                        self.recorded_open.append(
                            self._last_openness if face_detected else 0.0
                        )
                        self.total_frames += 1
                        if face_detected:
                            self.detection_frames += 1

                        n = len(self.recorded_lm)
                        if n == SEQUENCE_LENGTH // 2:
                            mid_frame = frame.copy()
                        if n >= SEQUENCE_LENGTH:
                            self.recording = False
                            saved = self.save_recording(mid_frame)
                            mid_frame = None
                            if saved and self.existing_count() >= self.samples:
                                print(f"\n🎉  '{self.current_word}' complete!")
                                if self.word_idx + 1 < len(self.words):
                                    self.word_idx += 1
                                    self.speed_cycle_idx = 0
                                    print(f"➡️   Next: '{self.current_word}' "
                                          f"({self.existing_count()}/{self.samples} existing)")
                                else:
                                    print("\n🏆  ALL WORDS COMPLETE!")

                self.draw_ui(frame, face_detected, face_landmarks)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" ") and not self.recording:
                    self.start_recording()
                elif key == ord("n") and not self.recording:
                    self.word_idx = (self.word_idx + 1) % len(self.words)
                    self.speed_cycle_idx = 0
                    print(f"  ➡️  {self.current_word}  "
                          f"({self.existing_count()}/{self.samples})")
                elif key == ord("p") and not self.recording:
                    self.word_idx = (self.word_idx - 1) % len(self.words)
                    self.speed_cycle_idx = 0
                    print(f"  ⬅️  {self.current_word}  "
                          f"({self.existing_count()}/{self.samples})")
                elif key == ord("r") and not self.recording:
                    self.delete_last()
                elif key == ord("i") and not self.recording:
                    self.integrity_check()

        self.cap.release()
        cv2.destroyAllWindows()
        self._summary()

    # ── summary ──────────────────────────────────────────────────

    def _summary(self):
        print("\n" + "=" * 68)
        print("📊  COLLECTION SUMMARY")
        print("=" * 68)
        total = 0
        for w in self.words:
            n = self.existing_count(w)
            total += n
            status = "✅" if n >= self.samples else f"⚠️  {n}/{self.samples}"
            print(f"   {status:<10}  {w}")
        print(f"\n   Total recordings : {total}")
        print(f"   Session log      : {os.path.abspath(SESSION_LOG)}")
        print(f"   Saved to         : {os.path.abspath(OUTPUT_DIR)}/")
        print("=" * 68)
        print("\n▶  Next step: python lip_train.py --arch bilstm\n")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Lip Reading Data Collector — English Enhanced"
    )
    parser.add_argument("--words",   default="",
                        help="Comma-separated word list (default: built-in 30 words)")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_WORD,
                        help=f"Recordings per word (default: {SAMPLES_PER_WORD})")
    parser.add_argument("--cam",     type=int, default=CAM_INDEX,
                        help="Camera index (default: 0)")
    args = parser.parse_args()

    words = (
        [w.strip() for w in args.words.split(",") if w.strip()]
        if args.words else DEFAULT_WORDS
    )

    collector = LipCollector(
        words=words, samples=args.samples, cam_index=args.cam
    )
    collector.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted.")
    except Exception as e:
        import traceback
        print(f"\n❌  Error: {e}")
        traceback.print_exc()