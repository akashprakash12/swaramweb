"""
STEP 3 — DESKTOP LIVE TEST
============================
Motion-gated real-time inference using webcam.
Tests model.h5 + scaler.pkl before deploying to Android.

Pipeline:
  Sign words → Gemini English sentence → Gemini Malayalam → gTTS audio
  + Every confident prediction is spoken in Malayalam immediately via gTTS

SETUP
  pip install google-generativeai gtts pygame Pillow
  sudo apt install fonts-noto-core        # Linux Malayalam font
  export GEMINI_API_KEY="your-key-here"

CONTROLS
  Q — quit          R — reset to IDLE
  S — landmarks     F — fullscreen
  G — manual Gemini trigger
  C — clear buffer  M — mute / unmute
"""

import json, os, io, time, threading, tempfile
from collections import deque

import cv2
import joblib
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ── Gemini (new google.genai SDK) ─────────────
try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  pip install google-genai")

# ── gTTS ─────────────────────────────────────
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  pip install gtts")

# ── pygame ────────────────────────────────────
try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception as e:
    PYGAME_AVAILABLE = False
    print(f"⚠️  pygame init failed: {e}  — pip install pygame")

# ── Pillow (Malayalam Unicode rendering) ──────
try:
    from PIL import ImageFont, ImageDraw, Image as PILImage
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("⚠️  pip install Pillow")

if not hasattr(mp, "solutions"):
    raise RuntimeError("Incompatible mediapipe — reinstall from requirements.txt")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH  = "model.h5"
SCALER_PATH = "scaler.pkl"
LABELS_PATH = "labels.json"

SEQUENCE_LENGTH     = 30
THRESHOLD           = 0.85
MOTION_THRESHOLD    = 0.045
STILLNESS_THRESHOLD = MOTION_THRESHOLD * 0.4
STILLNESS_FRAMES    = 12
MIN_CAPTURE_FRAMES  = 15
COOLDOWN_SECONDS    = 1.2
MP_COMPLEXITY       = 0
MIN_CONF            = 0.5
WINDOW_NAME         = "Step 3 — Live Test"
START_FULLSCREEN    = True

GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL        = "gemini-2.5-flash"   # new SDK uses this name directly
GEMINI_AUTO_TRIGGER = 5      # fewer API calls = less quota burn
GEMINI_BUFFER_SIZE  = 10

# ── Malayalam font search ─────────────────────
_FONT_CANDIDATES = [
    os.environ.get("MALAYALAM_FONT_PATH", ""),
    "/usr/share/fonts/truetype/noto/NotoSansMalayalam-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifMalayalam-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansMalayalam-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    r"C:\Windows\Fonts\Kartika.ttf",
    r"C:\Windows\Fonts\AnjaliOldLipi.ttf",
    "/Library/Fonts/AnjaliOldLipi.ttf",
]
MALAYALAM_FONT_PATH = next((p for p in _FONT_CANDIDATES if p and os.path.exists(p)), "")
if MALAYALAM_FONT_PATH:
    print(f"✅  Malayalam font: {MALAYALAM_FONT_PATH}")
else:
    print("⚠️  No Malayalam font found — run: sudo apt install fonts-noto-core")

# ─────────────────────────────────────────────
IDLE, CAPTURING, PREDICTING, COOLDOWN = "IDLE", "CAPTURING", "PREDICTING", "COOLDOWN"
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

# ── shared mute flag (set in main, read in threads) ──
_MUTED = False


# ═══════════════════════════════════════════
# AUDIO
# ═══════════════════════════════════════════
_tts_lock = threading.Lock()   # only one playback at a time

def speak(text: str, lang: str = "ml"):
    """Non-blocking TTS — fires in a daemon thread."""
    if _MUTED or not TTS_AVAILABLE or not PYGAME_AVAILABLE or not text.strip():
        return
    threading.Thread(target=_speak_blocking, args=(text, lang), daemon=True).start()

def _speak_blocking(text: str, lang: str):
    with _tts_lock:
        try:
            tts = gTTS(text=text, lang=lang)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(buf.read())
                tmp_path = tmp.name
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            os.unlink(tmp_path)
        except Exception as e:
            print(f"❌  TTS error: {e}")


# ═══════════════════════════════════════════
# GEMINI PIPELINE
# ═══════════════════════════════════════════
class GeminiPipeline:
    """
    Background thread:
      word list → corrected English → Malayalam translation → speak Malayalam
    """
    def __init__(self):
        self.available = GEMINI_AVAILABLE and bool(GEMINI_API_KEY)
        self.client    = None
        self.english   = ""
        self.malayalam = ""
        self.status    = "idle"   # idle | thinking | done | error
        self._thread   = None

        if self.available:
            try:
                self.client = genai_new.Client(api_key=GEMINI_API_KEY)
                print(f"✅  Gemini ready ({GEMINI_MODEL})")
            except Exception as e:
                self.available = False
                print(f"⚠️  Gemini init failed: {e}")
        else:
            print("⚠️  Gemini disabled — set GEMINI_API_KEY")

    def build(self, words: list):
        if not self.available or not words:
            return
        if self._thread and self._thread.is_alive():
            print("⏳  Gemini still running — skipping")
            return
        self.status    = "thinking"
        self.english   = ""
        self.malayalam = ""
        self._thread   = threading.Thread(target=self._run, args=(list(words),), daemon=True)
        self._thread.start()

    def clear(self):
        self.english = ""
        self.malayalam = ""
        self.status = "idle"

    def _gemini_call(self, prompt: str, label: str = "Gemini") -> str:
        """Call Gemini with exponential backoff on 429 rate-limit errors."""
        import re as _re
        delay = 5
        for attempt in range(4):
            try:
                resp = self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                return resp.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err:
                    m = _re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err)
                    wait = int(m.group(1)) + 2 if m else delay
                    print(f"\u23f3  Rate-limited \u2014 waiting {wait}s (attempt {attempt+1}/4)...")
                    time.sleep(wait)
                    delay = min(delay * 2, 120)
                else:
                    raise
        raise RuntimeError(f"{label}: max retries exceeded")

    def _run(self, words: list):
        # Single Gemini call returns BOTH English + Malayalam -> halves quota usage
        combined_prompt = (
            "You are a sign language interpreter.\n"
            "Input sign words (in order): " + ", ".join(words) + "\n\n"
            "Do TWO things:\n"
            "1. Write one natural English sentence from these signs.\n"
            "2. Translate that sentence into Malayalam script (NOT transliteration).\n\n"
            "Reply in EXACTLY this format (two lines only, nothing else):\n"
            "English: <sentence>\n"
            "Malayalam: <sentence>"
        )
        try:
            raw = self._gemini_call(combined_prompt)

            # ── DEBUG: always print raw Gemini response ──────────────
            print("─" * 55)
            print("🤖  RAW GEMINI RESPONSE:")
            print(raw)
            print("─" * 55)

            lines = raw.strip().splitlines()
            en, ml = "", ""
            for ln in lines:
                ln_lower = ln.lower()
                if ln_lower.startswith("english:"):
                    en = ln.split(":", 1)[1].strip().strip('"\'\' ')
                elif ln_lower.startswith("malayalam:"):
                    ml = ln.split(":", 1)[1].strip().strip('"\'\' ')
            # Fallback: if labels missing, take first two non-empty lines
            if not en or not ml:
                clean = [l.strip() for l in lines if l.strip()]
                if len(clean) >= 2:
                    en, ml = clean[0], clean[1]
            self.english   = en
            self.malayalam = ml
            self.status    = "done"

            # ── Rich terminal output ─────────────────────────────────
            print(f"✅  ENGLISH   : {self.english}")
            print(f"🌐  MALAYALAM : {self.malayalam}")
            print("─" * 55)

        except Exception as e:
            self.status = "error"
            print(f"❌  Gemini error: {e}")
            return

        if self.malayalam:
            speak(self.malayalam, lang="ml")
        else:
            print("⚠️  Malayalam empty — check raw response above")

# ═══════════════════════════════════════════
def normalize_hand(hand_flat):
    pts = hand_flat.reshape(21, 3).copy()
    pts -= pts[0].copy()
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().astype(np.float32)

def normalize_pose(pose_flat):
    pts = pose_flat.reshape(33, 3).copy()
    mid = (pts[11] + pts[12]) / 2.0
    pts -= mid
    w = np.linalg.norm(pts[11] - pts[12])
    if w > 1e-6:
        pts /= w
    return pts.flatten().astype(np.float32)

def extract_keypoints(hr, pr):
    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)
    if hr.multi_hand_landmarks and hr.multi_handedness:
        for lm, hd in zip(hr.multi_hand_landmarks, hr.multi_handedness):
            raw  = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
            norm = normalize_hand(raw)
            if hd.classification[0].label == "Left":
                lh = norm
            else:
                rh = norm
    pose = np.zeros(99, dtype=np.float32)
    if pr.pose_landmarks:
        raw  = np.array([[p.x, p.y, p.z] for p in pr.pose_landmarks.landmark],
                        dtype=np.float32).flatten()
        pose = normalize_pose(raw)
    return np.concatenate([lh, rh, pose])

def scale_sequence(seq, scaler):
    return scaler.transform(seq.reshape(-1, 225)).reshape(1, 30, 225)

def compute_motion(prev, curr):
    if prev is None:
        return 0.0
    return float(np.linalg.norm(curr[:126] - prev[:126]))


# ═══════════════════════════════════════════
# PILLOW TEXT RENDERER  (Malayalam Unicode)
# ═══════════════════════════════════════════
_font_cache: dict = {}

def _get_font(size: int):
    if size not in _font_cache:
        if PILLOW_AVAILABLE and MALAYALAM_FONT_PATH:
            try:
                _font_cache[size] = ImageFont.truetype(MALAYALAM_FONT_PATH, size)
                return _font_cache[size]
            except Exception:
                pass
        _font_cache[size] = None
    return _font_cache[size]

def put_unicode_text(frame, text: str, xy: tuple, size: int = 24,
                     color_bgr: tuple = (80, 255, 160)):
    """Render Unicode text onto a BGR OpenCV frame using Pillow."""
    font = _get_font(size)
    if font is None:
        # Fallback: ASCII replacement via OpenCV
        fb = text.encode("ascii", errors="replace").decode()
        cv2.putText(frame, fb, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color_bgr, 1, cv2.LINE_AA)
        return
    try:
        rgb   = (color_bgr[2], color_bgr[1], color_bgr[0])
        pil   = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw  = ImageDraw.Draw(pil)
        draw.text(xy, text, font=font, fill=rgb)
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        fb = text.encode("ascii", errors="replace").decode()
        cv2.putText(frame, fb, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color_bgr, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════
def wrap_text(text, max_chars):
    words, lines, line = text.split(), [], ""
    for w in words:
        candidate = (line + " " + w).strip()
        if len(candidate) <= max_chars:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines

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
        y = 50 + i * 35
        cv2.rectangle(frame, (5, y - 15), (255, y + 8), (50, 50, 50), -1)
        bar = int(250 * probs[idx])
        c   = (0, 220, 80) if i == 0 else (0, 180, 200) if i == 1 else (100, 100, 255)
        cv2.rectangle(frame, (5, y - 15), (5 + bar, y + 8), c, -1)
        cv2.putText(frame, f"{actions[idx]}: {probs[idx]*100:.1f}%",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

def draw_motion_meter(frame, motion_val):
    h, w   = frame.shape[:2]
    mx, my = w - 165, 55
    fill   = min(int(150 * motion_val / (MOTION_THRESHOLD * 3)), 150)
    color  = (0, 220, 80) if motion_val > MOTION_THRESHOLD else (80, 80, 200)
    cv2.rectangle(frame, (mx, my), (mx + 150, my + 14), (40, 40, 40), -1)
    cv2.rectangle(frame, (mx, my), (mx + fill, my + 14), color, -1)
    cv2.line(frame, (mx + 50, my), (mx + 50, my + 14), (0, 220, 220), 2)
    cv2.putText(frame, f"Motion {motion_val:.3f}", (mx, my - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

def draw_gemini_panel(frame, gemini: GeminiPipeline, word_buffer: list):
    """
    Draws a 3-row panel at the top of the frame:
      Row 1 — raw sign buffer
      Row 2 — corrected English sentence
      Row 3 — Malayalam translation (rendered with Pillow for proper Unicode)
    """
    h, w  = frame.shape[:2]
    py, ph, lh = 80, 175, 28
    mc = max(1, (w - 24) // 11)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, py), (w, py + ph), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    y = py + 22

    # ── Row 1: raw sign words ──────────────
    raw = "  ›  ".join(word_buffer[-GEMINI_BUFFER_SIZE:]) if word_buffer else "—"
    cv2.putText(frame, f"Signs: {raw}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    y += lh

    # ── Row 2: English sentence ────────────
    if gemini.status == "thinking":
        en_text, en_col = "Gemini thinking...", (0, 200, 255)
    elif gemini.status == "error":
        en_text, en_col = "Error — check API key / network", (50, 80, 255)
    elif gemini.english:
        en_text, en_col = gemini.english, (210, 210, 210)
    else:
        en_text, en_col = "Press [G] to translate  |  [C] to clear", (70, 70, 70)

    cv2.putText(frame, f"EN: {wrap_text(en_text, mc - 4)[0]}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, en_col, 1, cv2.LINE_AA)
    y += lh

    # ── Row 3: Malayalam (Pillow rendered) ──
    cv2.rectangle(frame, (0, y - 20), (w, y + lh * 2 + 10), (0, 50, 28), -1)

    if gemini.malayalam:
        # Label
        cv2.putText(frame, "ML:", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 180, 120), 1, cv2.LINE_AA)
        # Malayalam text via Pillow — wraps up to 2 lines
        ml_max   = max(1, (w - 60) // 20)   # ~20px per Malayalam glyph at size 26
        ml_lines = wrap_text(gemini.malayalam, ml_max)
        for li, ml_line in enumerate(ml_lines[:2]):
            put_unicode_text(frame, ml_line,
                             xy=(50, y - 20 + li * (lh + 2)),
                             size=26, color_bgr=(60, 255, 150))
        y += lh * 2 + 4
    elif gemini.status == "thinking":
        cv2.putText(frame, "ML: translating...", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)
        y += lh
    elif gemini.status == "done":
        cv2.putText(frame, "ML: (empty — retrying may help)", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 180), 1, cv2.LINE_AA)
        y += lh
    else:
        cv2.putText(frame, "ML: —", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)
        y += lh

    # ── badges ─────────────────────────────
    muted_now  = _MUTED
    mute_label = "MUTED" if muted_now else "AUDIO ON"
    mute_col   = (50, 50, 200) if muted_now else (0, 180, 80)
    cv2.putText(frame, mute_label, (w - 115, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, mute_col, 1, cv2.LINE_AA)
    cv2.putText(frame, "G=build  C=clear  M=mute",
                (w - 215, py + ph - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 70), 1, cv2.LINE_AA)

def init_preview_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    fullscreen = False
    if START_FULLSCREEN:
        try:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            fullscreen = True
        except cv2.error:
            pass
    return fullscreen


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    global _MUTED

    for path in [MODEL_PATH, SCALER_PATH, LABELS_PATH]:
        if not os.path.exists(path):
            print(f"❌  Missing: {path}  — run step2_train.py first.")
            return

    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        actions = np.array(json.load(f))

    gemini             = GeminiPipeline()
    word_buffer        = []
    preds_since_gemini = 0

    print("=" * 60)
    print("📷  STEP 3 — Motion-Gated + Gemini + Malayalam TTS")
    print("=" * 60)
    print(f"   Classes : {list(actions)}")
    print(f"   Gemini  : {'✅' if gemini.available else '⚠️  disabled'}")
    print(f"   TTS     : {'✅' if TTS_AVAILABLE and PYGAME_AVAILABLE else '⚠️  disabled'}")
    print(f"   Font    : {'✅ ' + MALAYALAM_FONT_PATH if MALAYALAM_FONT_PATH else '⚠️  missing'}")
    print()
    print("   Q=quit  R=reset  S=landmarks  F=fullscreen")
    print("   G=translate+speak  C=clear  M=mute")
    print("=" * 60 + "\n")

    state                = IDLE
    sequence             = []
    prev_kp              = None
    stillness_counter    = 0
    motion_trigger_count = 0
    last_pred_time       = 0.0
    current_prediction   = ""
    current_confidence   = 0.0
    last_probs           = None
    motion_val           = 0.0
    motion_history       = deque(maxlen=5)
    show_landmarks       = True
    fps_times            = deque(maxlen=30)
    is_fullscreen        = init_preview_window()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
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

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hr = hands.process(rgb)
            pr = pose.process(rgb)
            rgb.flags.writeable = True

            if show_landmarks:
                if hr.multi_hand_landmarks:
                    for hl in hr.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hl, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                if pr.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, pr.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing_styles.get_default_pose_landmarks_style())

            kp = extract_keypoints(hr, pr)
            motion_history.append(compute_motion(prev_kp, kp))
            motion_val = float(np.mean(motion_history))
            prev_kp    = kp.copy()

            hands_det = hr.multi_hand_landmarks is not None
            now       = time.time()

            # ── STATE MACHINE ─────────────────
            if state == IDLE:
                if hands_det and motion_val > MOTION_THRESHOLD:
                    motion_trigger_count += 1
                    if motion_trigger_count >= 3:
                        state = CAPTURING; sequence = [kp]
                        stillness_counter = 0; motion_trigger_count = 0
                        print("▶  Capturing...")
                else:
                    motion_trigger_count = 0

            elif state == CAPTURING:
                sequence.append(kp)
                stillness_counter = (0 if motion_val > STILLNESS_THRESHOLD
                                     else stillness_counter + 1)
                if (len(sequence) >= SEQUENCE_LENGTH or
                        (stillness_counter >= STILLNESS_FRAMES
                         and len(sequence) >= MIN_CAPTURE_FRAMES)):
                    state = PREDICTING

            elif state == PREDICTING:
                seq_arr = np.array(sequence, dtype=np.float32)
                if len(seq_arr) < SEQUENCE_LENGTH:
                    pad     = np.tile(seq_arr[-1], (SEQUENCE_LENGTH - len(seq_arr), 1))
                    seq_arr = np.vstack([seq_arr, pad])
                else:
                    seq_arr = seq_arr[:SEQUENCE_LENGTH]

                preds      = model.predict(scale_sequence(seq_arr, scaler), verbose=0)[0]
                last_probs = preds
                pred_class = int(np.argmax(preds))
                confidence = float(preds[pred_class])

                if confidence >= THRESHOLD:
                    current_prediction = actions[pred_class]
                    current_confidence = confidence
                    last_pred_time     = now
                    print(f"✅  PREDICTED : {current_prediction}  ({confidence*100:.1f}%)")
                    print(f"   Buffer now: {word_buffer + [current_prediction]}")

                    # ── speak predicted word immediately in Malayalam ──
                    speak(current_prediction, lang="ml")

                    word_buffer.append(current_prediction)
                    if len(word_buffer) > GEMINI_BUFFER_SIZE:
                        word_buffer.pop(0)

                    preds_since_gemini += 1
                    if GEMINI_AUTO_TRIGGER > 0 and preds_since_gemini >= GEMINI_AUTO_TRIGGER:
                        print(f"🧠  Auto Gemini: {word_buffer}")
                        gemini.build(word_buffer)
                        preds_since_gemini = 0
                else:
                    print(f"⚠️  Low conf {confidence*100:.1f}% [{actions[pred_class]}]")

                sequence = []; stillness_counter = 0; state = COOLDOWN

            elif state == COOLDOWN:
                if now - last_pred_time >= COOLDOWN_SECONDS:
                    state = IDLE; print("⏸  Ready")

            # ── DRAW ──────────────────────────
            cv2.putText(frame,
                        f"Hands: {'✓' if hands_det else '✗'}  "
                        f"Pose: {'✓' if pr.pose_landmarks else '✗'}",
                        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 80) if hands_det else (0, 50, 220), 2)

            draw_motion_meter(frame, motion_val)
            draw_state_bar(frame, state, len(sequence))
            draw_gemini_panel(frame, gemini, word_buffer)
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
                print("\n👋  Quit"); break
            elif key == ord("r"):
                state = IDLE; sequence = []; current_prediction = ""
                last_probs = None; stillness_counter = 0; motion_trigger_count = 0
                word_buffer.clear(); gemini.clear(); preds_since_gemini = 0
                print("🔄  Reset → IDLE")
            elif key == ord("s"):
                show_landmarks = not show_landmarks
            elif key == ord("f"):
                is_fullscreen = not is_fullscreen
                try:
                    cv2.setWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
                except cv2.error:
                    pass
            elif key == ord("g"):
                if word_buffer:
                    print(f"🧠  Manual Gemini: {word_buffer}")
                    gemini.build(word_buffer); preds_since_gemini = 0
                else:
                    print("⚠️  No words buffered yet")
            elif key == ord("c"):
                word_buffer.clear(); gemini.clear(); preds_since_gemini = 0
                print("🗑  Buffer cleared")
            elif key == ord("m"):
                _MUTED = not _MUTED
                print("🔇 Muted" if _MUTED else "🔊 Unmuted")

    cap.release()
    cv2.destroyAllWindows()
    if PYGAME_AVAILABLE:
        pygame.mixer.quit()
    print("✅  Session ended.")

if __name__ == "__main__":
    main()