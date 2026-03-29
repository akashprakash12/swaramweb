"""
STEP 3 — DESKTOP LIVE TEST  (Malayalam Audio Output)
=====================================================
Pipeline: Signs → LLM Malayalam sentence → spoken aloud

FREE API OPTIONS (set ONE of these env vars):
─────────────────────────────────────────────
  GROQ_API_KEY      → https://console.groq.com          (FREE: 14400 req/day) ⭐
  GEMINI_API_KEY    → https://aistudio.google.com        (FREE: 1500 req/day)
  OPENROUTER_API_KEY→ https://openrouter.ai              (FREE models available)
  COHERE_API_KEY    → https://dashboard.cohere.com       (FREE: 1000 req/month)

SETUP
    pip install groq gtts pygame mediapipe tensorflow joblib opencv-python
    export GROQ_API_KEY="your_key_here"

Audio fallback chain:
  pygame → playsound → mpg123 → ffplay → pydub → afplay → winsound

CONTROLS  Q=quit  R=reset  G=speak now  C=clear  M=mute  S=landmarks
"""

import json, os, io, time, threading, tempfile, subprocess, sys, warnings
from collections import deque

import cv2
import joblib
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

# ── gTTS ─────────────────────────────────────
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  pip install gtts")


# ═══════════════════════════════════════════
# FREE LLM PROVIDER DETECTION
# Order: Groq → Gemini (new SDK) → Gemini (old SDK) → OpenRouter → Cohere → None
# ═══════════════════════════════════════════

GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "").strip()
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
COHERE_API_KEY     = os.environ.get("COHERE_API_KEY", "").strip()

LLM_PROVIDER = None  # set during detection below

# ── Try Groq ──────────────────────────────
_groq_client = None
try:
    from groq import Groq
    if GROQ_API_KEY:
        _groq_client = Groq(api_key=GROQ_API_KEY)
        LLM_PROVIDER = "groq"
        print(f"✅  LLM provider: Groq  (llama-3.1-8b-instant, FREE 14400 req/day)")
    else:
        print("ℹ️   Groq SDK found but GROQ_API_KEY not set — skipping")
except ImportError:
    if GROQ_API_KEY:
        print("⚠️  pip install groq   (key found but SDK missing)")

# ── Try Gemini (new SDK) ──────────────────
_gemini_new_client = None
if LLM_PROVIDER is None:
    try:
        from google import genai as genai_new
        if GEMINI_API_KEY:
            _gemini_new_client = genai_new.Client(api_key=GEMINI_API_KEY)
            LLM_PROVIDER = "gemini_new"
            print(f"✅  LLM provider: Gemini (google-genai SDK, FREE 1500 req/day)")
    except Exception as e:
        if GEMINI_API_KEY:
            print(f"ℹ️   google-genai SDK issue: {e}")

# ── Try Gemini (old SDK) ──────────────────
_gemini_old_model = None
if LLM_PROVIDER is None:
    try:
        import google.generativeai as genai_old
        if GEMINI_API_KEY:
            genai_old.configure(api_key=GEMINI_API_KEY)
            _gemini_old_model = genai_old.GenerativeModel("gemini-1.5-flash")
            LLM_PROVIDER = "gemini_old"
            print(f"✅  LLM provider: Gemini (google-generativeai SDK, FREE 1500 req/day)")
    except Exception as e:
        if GEMINI_API_KEY:
            print(f"ℹ️   google-generativeai issue: {e}")

# ── Try OpenRouter ────────────────────────
if LLM_PROVIDER is None and OPENROUTER_API_KEY:
    try:
        import openai as _oa
        _openrouter_client = _oa.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        LLM_PROVIDER = "openrouter"
        print(f"✅  LLM provider: OpenRouter  (FREE models available)")
    except ImportError:
        print("⚠️  pip install openai  (for OpenRouter support)")

# ── Try Cohere ────────────────────────────
_cohere_client = None
if LLM_PROVIDER is None and COHERE_API_KEY:
    try:
        import cohere
        _cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
        LLM_PROVIDER = "cohere"
        print(f"✅  LLM provider: Cohere  (FREE 1000 req/month)")
    except ImportError:
        print("⚠️  pip install cohere  (for Cohere support)")

if LLM_PROVIDER is None:
    print("⚠️  No LLM provider found. Local word-map fallback only.")
    print("    Quick fix: export GROQ_API_KEY=<your_key>  (free at console.groq.com)")

LLM_AVAILABLE = LLM_PROVIDER is not None


def _call_llm(prompt: str) -> str:
    """Call whichever LLM provider is configured. Returns raw text."""
    if LLM_PROVIDER == "groq":
        resp = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()

    elif LLM_PROVIDER == "gemini_new":
        return _gemini_new_client.models.generate_content(
            model="gemini-1.5-flash", contents=prompt
        ).text.strip()

    elif LLM_PROVIDER == "gemini_old":
        return _gemini_old_model.generate_content(prompt).text.strip()

    elif LLM_PROVIDER == "openrouter":
        resp = _openrouter_client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()

    elif LLM_PROVIDER == "cohere":
        resp = _cohere_client.chat(
            model="command-r",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return resp.message.content[0].text.strip()

    raise RuntimeError("No LLM provider configured")


# ─────────────────────────────────────────────
# AUDIO ENGINE — auto-detects best method
# ─────────────────────────────────────────────
_AUDIO_ENGINE = None
_tts_lock     = threading.Lock()
_MUTED        = False

def _detect_audio_engine():
    global _AUDIO_ENGINE

    try:
        import pygame
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.mixer.init()
        if pygame.mixer.get_init():
            _AUDIO_ENGINE = "pygame"
            print("🔊  Audio engine: pygame")
            return
    except Exception as e:
        print(f"   pygame failed: {e}")

    try:
        import playsound
        _AUDIO_ENGINE = "playsound"
        print("🔊  Audio engine: playsound")
        return
    except Exception:
        pass

    if subprocess.run(["which", "mpg123"], capture_output=True).returncode == 0:
        _AUDIO_ENGINE = "mpg123"
        print("🔊  Audio engine: mpg123")
        return

    if subprocess.run(["which", "ffplay"], capture_output=True).returncode == 0:
        _AUDIO_ENGINE = "ffplay"
        print("🔊  Audio engine: ffplay")
        return

    try:
        from pydub import AudioSegment
        from pydub.playback import play
        _AUDIO_ENGINE = "pydub"
        print("🔊  Audio engine: pydub")
        return
    except Exception:
        pass

    if sys.platform == "darwin":
        _AUDIO_ENGINE = "afplay"
        print("🔊  Audio engine: afplay (macOS)")
        return

    if sys.platform == "win32":
        _AUDIO_ENGINE = "winsound"
        print("🔊  Audio engine: winsound (Windows)")
        return

    print("❌  No audio engine found!")
    print("    Fix: pip install pygame   OR   sudo apt install mpg123")
    _AUDIO_ENGINE = None

_detect_audio_engine()


def _play_mp3(path: str):
    if _AUDIO_ENGINE == "pygame":
        import pygame
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    elif _AUDIO_ENGINE == "playsound":
        import playsound
        playsound.playsound(path, block=True)

    elif _AUDIO_ENGINE == "mpg123":
        subprocess.run(["mpg123", "-q", path], check=True)

    elif _AUDIO_ENGINE == "ffplay":
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            check=True
        )

    elif _AUDIO_ENGINE == "pydub":
        from pydub import AudioSegment
        from pydub.playback import play
        play(AudioSegment.from_mp3(path))

    elif _AUDIO_ENGINE == "afplay":
        subprocess.run(["afplay", path], check=True)

    elif _AUDIO_ENGINE == "winsound":
        wav = path.replace(".mp3", ".wav")
        subprocess.run(["ffmpeg", "-y", "-i", path, wav], capture_output=True)
        import winsound
        winsound.PlaySound(wav, winsound.SND_FILENAME)
        os.unlink(wav)


def speak(text: str, lang: str = "ml"):
    if _MUTED or not TTS_AVAILABLE or not _AUDIO_ENGINE or not text.strip():
        if not TTS_AVAILABLE:
            print(f"[TTS disabled] would say: {text}")
        if not _AUDIO_ENGINE:
            print(f"[No audio engine] would say: {text}")
        return
    threading.Thread(target=_speak_blocking, args=(text, lang), daemon=True).start()

def _speak_blocking(text: str, lang: str):
    with _tts_lock:
        tmp_path = None
        try:
            buf = io.BytesIO()
            gTTS(text=text, lang=lang).write_to_fp(buf)
            buf.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(buf.read())
                tmp_path = tmp.name
            _play_mp3(tmp_path)
        except Exception as e:
            print(f"❌  TTS/audio error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════
MODEL_PATH  = "model.h5"
SCALER_PATH = "scaler.pkl"
LABELS_PATH = "labels.json"

SEQUENCE_LENGTH          = 30
THRESHOLD                = 0.85
MOTION_THRESHOLD         = 0.045
STILLNESS_THRESHOLD      = MOTION_THRESHOLD * 0.4
STILLNESS_FRAMES         = 12
MIN_CAPTURE_FRAMES       = 15
COOLDOWN_SECONDS         = 1.2
MIN_CAPTURE_PEAK_MOTION  = MOTION_THRESHOLD * 0.95
MIN_CAPTURE_MEAN_MOTION  = STILLNESS_THRESHOLD * 1.10
MP_COMPLEXITY            = 0
MIN_CONF                 = 0.5
WINDOW_NAME              = "Sign → Malayalam Audio"

LLM_BUFFER_SIZE          = 10
SILENCE_TRIGGER_SEC      = 2.0
# Trigger LLM even for a single recognized sign so conversion happens consistently.
MIN_WORDS_FOR_AUTO_LLM   = 1
LLM_MAX_RETRIES          = 1

IDLE, CAPTURING, PREDICTING, COOLDOWN = "IDLE", "CAPTURING", "PREDICTING", "COOLDOWN"

mp_hands          = mp.solutions.hands
mp_pose           = mp.solutions.pose
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ═══════════════════════════════════════════
# LLM PIPELINE  (provider-agnostic)
# ═══════════════════════════════════════════
class LLMPipeline:
    """Wraps any configured LLM to produce English + Malayalam sentences."""

    WORD_MAP = {
        "Big": "വലിയത്", "Cat": "പൂച്ച", "Dog": "നായ",
        "Drink": "കുടിക്കുക", "Eat": "തിന്നുക", "Fish": "മത്സ്യം",
        "Food": "ഭക്ഷണം", "Give": "തരുക", "Go": "പോകുക",
        "Happy": "സന്തോഷം", "Hello": "നമസ്കാരം", "Help": "സഹായം",
        "House": "വീട്", "I": "ഞാൻ", "Lake": "തടാകം",
        "Play": "കളിക്കുക", "Run": "ഓടുക", "See": "കാണുക",
        "Sorry": "ക്ഷമിക്കണം", "Take": "എടുക്കുക", "Thanks": "നന്ദി",
        "Water": "വെള്ളം", "You": "നീ",
    }

    def __init__(self):
        self.available  = LLM_AVAILABLE
        self.status     = "idle"
        self.last_en    = ""
        self.last_ml    = ""
        self._thread    = None
        self._queue     = deque()
        self._lock      = threading.Lock()
        self._quota_block_until = 0.0
        provider_name   = LLM_PROVIDER.upper() if LLM_PROVIDER else "NONE"
        print(f"✅  LLMPipeline ready — provider: {provider_name}")

    def _fallback_sentence(self, words: list):
        en = " ".join(words)
        ml = " ".join(self.WORD_MAP.get(w, w) for w in words)
        return en, ml

    def build(self, words: list):
        if not words:
            return False
        snapshot = list(words)
        with self._lock:
            was_busy = self._thread is not None and self._thread.is_alive()
            self._queue.append(snapshot)
        if was_busy:
            print(f"⏳  LLM busy — queued: {snapshot}")
        self._start_next_if_idle()
        return True

    def pending_count(self) -> int:
        with self._lock:
            running = 1 if (self._thread is not None and self._thread.is_alive()) else 0
            return running + len(self._queue)

    def clear(self):
        with self._lock:
            self._queue.clear()
        self.status  = "idle"
        self.last_en = ""
        self.last_ml = ""

    def _start_next_if_idle(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if not self._queue:
                if self.status == "thinking":
                    self.status = "idle"
                return
            next_words   = self._queue.popleft()
            self.status  = "thinking"
            use_fallback = (not self.available) or (time.time() < self._quota_block_until)
            target       = self._run_fallback if use_fallback else self._run
            self._thread = threading.Thread(target=target, args=(next_words,), daemon=True)
            self._thread.start()

    def _run_fallback(self, words: list):
        en, ml = self._fallback_sentence(words)
        self.last_en = en
        self.last_ml = ml
        self.status  = "done"
        print("⚠️  Using local word-map fallback.")
        if ml:
            speak(ml, lang="ml")
        self._start_next_if_idle()

    def _run(self, words: list):
        prompt = (
            "You are a sign language interpreter.\n"
            "Input sign words (in order): " + ", ".join(words) + "\n\n"
            "1. Write one natural English sentence from these signs.\n"
            "2. Translate it into Malayalam script (NOT transliteration).\n\n"
            "Reply EXACTLY in this format:\n"
            "English: <sentence>\n"
            "Malayalam: <sentence>"
        )
        try:
            raw = ""
            for attempt in range(LLM_MAX_RETRIES):
                try:
                    raw = _call_llm(prompt)
                    break
                except Exception as e:
                    msg = str(e)
                    # Rate-limit handling (429 or similar)
                    if any(x in msg for x in ("429", "rate_limit", "quota", "RateLimitError")):
                        wait = self._extract_retry_seconds(msg)
                        self._quota_block_until = time.time() + max(wait, 30)
                        print(f"⚠️  Rate limited. Fallback for ~{max(wait,30)}s.")
                        raise
                    raise

            en = ml = ""
            for ln in raw.splitlines():
                low = ln.lower()
                if low.startswith("english:"):
                    en = ln.split(":", 1)[1].strip().strip('"\'')
                elif low.startswith("malayalam:"):
                    ml = ln.split(":", 1)[1].strip().strip('"\'')
            if not en or not ml:
                clean = [l.strip() for l in raw.splitlines() if l.strip()]
                if len(clean) >= 2:
                    en, ml = clean[0], clean[1]

            if not en or not ml:
                en, ml = self._fallback_sentence(words)

            self.last_en = en
            self.last_ml = ml
            self.status  = "done"
            print(f"✅  EN : {en}")
            
            if ml:
                speak(ml, lang="ml")
            else:
                print("⚠️  Malayalam empty")

        except Exception as e:
            en, ml       = self._fallback_sentence(words)
            self.last_en = en
            self.last_ml = ml
            self.status  = "error"
            if ml:
                speak(ml, lang="ml")
            short = str(e).splitlines()[0][:180]
            print(f"❌  LLM error: {short}")
        finally:
            self._start_next_if_idle()

    @staticmethod
    def _extract_retry_seconds(err_text: str) -> int:
        import re as _re
        for pat in [
            r"retry_delay\s*\{\s*seconds:\s*(\d+)",
            r"retryDelay'\s*:\s*'(\d+)s'",
            r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
            r"Please try again in ([0-9]+(?:\.[0-9]+)?)s",
        ]:
            m = _re.search(pat, err_text, flags=_re.IGNORECASE)
            if m:
                try:
                    return max(1, int(float(m.group(1))))
                except Exception:
                    continue
        return 0


# ═══════════════════════════════════════════
# KEYPOINT HELPERS
# ═══════════════════════════════════════════
def normalize_hand(h):
    p = h.reshape(21, 3).copy(); p -= p[0]
    s = np.linalg.norm(p[9])
    if s > 1e-6: p /= s
    return p.flatten().astype(np.float32)

def normalize_pose(p_flat):
    p   = p_flat.reshape(33, 3).copy()
    mid = (p[11] + p[12]) / 2; p -= mid
    w   = np.linalg.norm(p[11] - p[12])
    if w > 1e-6: p /= w
    return p.flatten().astype(np.float32)

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
        raw  = np.array([[p.x, p.y, p.z] for p in pr.pose_landmarks.landmark], dtype=np.float32).flatten()
        pose = normalize_pose(raw)
    return np.concatenate([lh, rh, pose])

def scale_seq(seq, scaler):
    return scaler.transform(seq.reshape(-1, 225)).reshape(1, 30, 225)

def motion(prev, curr):
    if prev is None: return 0.0
    return float(np.linalg.norm(curr[:126] - prev[:126]))


# ═══════════════════════════════════════════
# HUD
# ═══════════════════════════════════════════
SC = {IDLE: (100,100,100), CAPTURING: (0,165,255),
      PREDICTING: (0,220,220), COOLDOWN: (180,80,240)}

def draw_hud(frame, state, words, g_status, pred, conf, mv, fps,
             last_en="", last_ml=""):
    h, w = frame.shape[:2]
    c    = SC[state]

    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    provider_tag = f"[{(LLM_PROVIDER or 'none').upper()}]"
    cv2.putText(frame,
                f"[{state}]  FPS:{fps:.0f}  Motion:{mv:.3f}  "
                f"Audio:{_AUDIO_ENGINE}  LLM:{provider_tag}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    mute_col = (50, 50, 220) if _MUTED else (0, 200, 80)
    cv2.putText(frame, "MUTED" if _MUTED else "AUDIO ON",
                (w - 130, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mute_col, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h - 90), (w, h - 64), (0, 0, 0), -1)
    cv2.putText(frame, "Words: " + (" | ".join(words) if words else "—"),
                (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 60), 1, cv2.LINE_AA)

    gc = {"idle": (70,70,70), "thinking": (0,200,255), "done": (0,180,80),
          "error": (50,50,220)}.get(g_status, (120,120,120))
    cv2.putText(frame,
                f"LLM:{g_status}  G=speak C=clear M=mute R=reset Q=quit",
                (10, h - 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42, gc, 1, cv2.LINE_AA)

    
    if last_en:
        cv2.putText(frame, f"EN: {last_en[:70]}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    if pred:
        col = (0, 220, 80) if conf > 0.90 else (0, 165, 255)
        cv2.rectangle(frame, (0, h - 64), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"{pred.upper()}  {conf*100:.1f}%",
                    (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3, cv2.LINE_AA)
        cv2.rectangle(frame, (0, h - 6), (int(w * conf), h), col, -1)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    global _MUTED

    for p in [MODEL_PATH, SCALER_PATH, LABELS_PATH]:
        if not os.path.exists(p):
            print(f"❌  Missing: {p}"); return

    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        actions = np.array(json.load(f))

    llm               = LLMPipeline()
    words             = []
    last_sign_time    = 0.0
    last_hand_seen_time = 0.0
    silence_triggered = False

    print("=" * 60)
    print("🎤  Sign Language → Malayalam AUDIO")
    print("=" * 60)
    print(f"   Classes : {list(actions)}")
    print(f"   LLM     : {LLM_PROVIDER or '❌  none — local fallback only'}")
    print(f"   TTS     : {'✅' if TTS_AVAILABLE else '❌  pip install gtts'}")
    print(f"   Audio   : {_AUDIO_ENGINE or '❌  pip install pygame'}")
    print("=" * 60)

    state   = IDLE
    seq     = []
    prev_kp = None
    sc_cnt  = 0
    mt_cnt  = 0
    lpt     = 0.0
    pred    = ""
    conf    = 0.0
    mv      = 0.0
    mhist   = deque(maxlen=5)
    fps_q   = deque(maxlen=30)
    show_lm = True
    cap_peak = cap_sum = cap_cnt = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Camera not found"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(model_complexity=MP_COMPLEXITY,
                        min_detection_confidence=MIN_CONF,
                        min_tracking_confidence=MIN_CONF,
                        max_num_hands=2) as hands, \
         mp_pose.Pose(model_complexity=MP_COMPLEXITY,
                      min_detection_confidence=MIN_CONF,
                      min_tracking_confidence=MIN_CONF) as pose_mp:

        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hr = hands.process(rgb)
            pr = pose_mp.process(rgb)
            rgb.flags.writeable = True

            if show_lm:
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
            mhist.append(motion(prev_kp, kp))
            mv      = float(np.mean(mhist))
            prev_kp = kp.copy()
            hdet    = hr.multi_hand_landmarks is not None
            now     = time.time()

            if hdet:
                last_hand_seen_time = now

            if state == IDLE:
                if hdet and mv > MOTION_THRESHOLD:
                    mt_cnt += 1
                    if mt_cnt >= 3:
                        state = CAPTURING; seq = [kp]; sc_cnt = mt_cnt = 0
                        cap_peak = cap_sum = mv; cap_cnt = 1
                        print("▶  Capturing...")
                else:
                    mt_cnt = 0

            elif state == CAPTURING:
                seq.append(kp)
                cap_peak  = max(cap_peak, mv)
                cap_sum  += mv; cap_cnt += 1
                sc_cnt    = 0 if mv > STILLNESS_THRESHOLD else sc_cnt + 1
                if len(seq) >= SEQUENCE_LENGTH or (sc_cnt >= STILLNESS_FRAMES and len(seq) >= MIN_CAPTURE_FRAMES):
                    state = PREDICTING

            elif state == PREDICTING:
                cap_mean = cap_sum / cap_cnt if cap_cnt > 0 else 0.0
                if cap_peak < MIN_CAPTURE_PEAK_MOTION or cap_mean < MIN_CAPTURE_MEAN_MOTION:
                    pred = ""; conf = 0.0
                    seq = []; sc_cnt = 0
                    cap_peak = cap_sum = cap_cnt = 0.0
                    state = IDLE
                    continue

                arr = np.array(seq, dtype=np.float32)
                if len(arr) < SEQUENCE_LENGTH:
                    arr = np.vstack([arr, np.tile(arr[-1], (SEQUENCE_LENGTH - len(arr), 1))])
                else:
                    arr = arr[:SEQUENCE_LENGTH]

                probs = model.predict(scale_seq(arr, scaler), verbose=0)[0]
                pc    = int(np.argmax(probs))
                conf  = float(probs[pc])

                if conf >= THRESHOLD:
                    pred = actions[pc]; lpt = now
                    print(f"✅  {pred}  ({conf*100:.1f}%)")
                    words.append(pred)
                    if len(words) > LLM_BUFFER_SIZE: words.pop(0)
                    last_sign_time    = now
                    silence_triggered = False
                else:
                    print(f"⚠️  Low conf {conf*100:.1f}%")
                    pred = ""; conf = 0.0

                seq = []; sc_cnt = 0; state = COOLDOWN
                cap_peak = cap_sum = cap_cnt = 0.0

            elif state == COOLDOWN:
                if now - lpt >= COOLDOWN_SECONDS:
                    state = IDLE; pred = ""; conf = 0.0

            # Auto-trigger only when hands are idle for SILENCE_TRIGGER_SEC.
            if (
                state in (IDLE, COOLDOWN)
                and len(words) >= MIN_WORDS_FOR_AUTO_LLM
                and llm.status != "thinking"
                and not hdet
                and last_hand_seen_time > 0
                and (now - last_hand_seen_time) >= SILENCE_TRIGGER_SEC
                and not silence_triggered
            ):
                print(f"🧠  Hands idle → LLM: {words}")
                if llm.build(words):
                    words.clear()
                    silence_triggered = True

            fps_q.append(time.time() - t0)
            fps = 1.0 / np.mean(fps_q) if fps_q else 0
            draw_hud(frame, state, words, llm.status, pred, conf, mv, fps,
                     llm.last_en, llm.last_ml)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quit"); break
            elif key == ord('r'):
                state = IDLE; seq = []; pred = ""; sc_cnt = mt_cnt = 0
                words.clear(); llm.clear()
                last_sign_time = 0.0; silence_triggered = False
                print("🔄 Reset")
            elif key == ord('s'):
                show_lm = not show_lm
            elif key == ord('g'):
                if words:
                    pending_before = llm.pending_count()
                    if llm.build(words):
                        words.clear(); silence_triggered = True
                        print("⏳  Queued" if pending_before > 0 else "🧠  Sent to LLM")
                else:
                    p = llm.pending_count()
                    print(f"⏳  {p} request(s) pending" if p > 0 else "⚠️  Buffer empty")
            elif key == ord('c'):
                words.clear(); llm.clear()
                last_sign_time = 0.0; silence_triggered = False
                print("🗑 Cleared")
            elif key == ord('m'):
                _MUTED = not _MUTED
                print("🔇 Muted" if _MUTED else "🔊 Unmuted")

    cap.release()
    cv2.destroyAllWindows()
    try:
        import pygame; pygame.mixer.quit()
    except Exception:
        pass
    print("✅  Done.")


if __name__ == "__main__":
    main()