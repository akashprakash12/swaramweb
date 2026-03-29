"""
Malayalam Sign Language Interpreter — Web App
==============================================
Pipeline: Webcam frames → MediaPipe keypoints → TFLite model
         → Recognized signs → Groq LLM → Malayalam sentence → gTTS audio

Set API keys via environment variables:
  GROQ_API_KEY
  GEMINI_API_KEY

Endpoints:
  GET  /           → Main UI
  POST /predict    → { client_id, image } → prediction JSON
  POST /llm        → { words } → { english, malayalam, audio_b64 }
  POST /reset      → { client_id } → reset frame buffer
  POST /tts        → { text, lang } → { audio_b64 }
"""

import base64
import io
import json
import os
import re
import tempfile
import threading
import time
import warnings
from collections import deque
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore", message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*")


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from .env into process environment if unset."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Do not crash app startup due to .env parsing issues.
        pass


_load_dotenv()

# ═══════════════════════════════════════════
# MODEL / SCALER / LABELS PATHS
# ═══════════════════════════════════════════
MODEL_PATH  = "model.tflite"
SCALER_JSON = "scaler.json"
LABELS_JSON = "labels.json"
SEQUENCE_LEN = 30
N_FEATURES   = 225
THRESHOLD    = 0.85

# ═══════════════════════════════════════════
# API KEYS (from environment only)
# ═══════════════════════════════════════════
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "").strip()
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
COHERE_API_KEY     = os.environ.get("COHERE_API_KEY",     "").strip()

LLM_BUFFER_SIZE        = 10
LLM_MAX_RETRIES        = 2
SILENCE_TRIGGER_SEC    = 2.0
MIN_WORDS_FOR_AUTO_LLM = 1

# ═══════════════════════
# LLM PROVIDERS
# Groq   → English sentence generation
# Gemini → Malayalam translation (superior Indic quality)
# ═══════════════════════
LLM_PROVIDER        = None
_groq_client        = None
_gemini_translate   = None
_GEMINI_ML_PROVIDER = None
_openrouter_client  = None
_cohere_client      = None

try:
    from groq import Groq
    if GROQ_API_KEY:
        _groq_client = Groq(api_key=GROQ_API_KEY)
        LLM_PROVIDER = "groq"
        print("✅  English LLM : Groq llama-3.1-8b-instant")
except ImportError:
    if GROQ_API_KEY:
        print("⚠️  pip install groq")

try:
    from google import genai as _genai_new
    if GEMINI_API_KEY:
        _gemini_translate   = _genai_new.Client(api_key=GEMINI_API_KEY)
        _GEMINI_ML_PROVIDER = "new"
        print("✅  Malayalam LLM: Gemini gemini-2.0-flash (google-genai SDK)")
        if LLM_PROVIDER is None:
            LLM_PROVIDER = "gemini_new"
except Exception as e:
    if GEMINI_API_KEY:
        print(f"ℹ️  google-genai SDK: {e}")

if _GEMINI_ML_PROVIDER is None:
    try:
        import google.generativeai as _genai_old
        if GEMINI_API_KEY:
            _genai_old.configure(api_key=GEMINI_API_KEY)
            _gemini_translate   = _genai_old.GenerativeModel("gemini-3-flash-preview")
            _GEMINI_ML_PROVIDER = "old"
            print("✅  Malayalam LLM: Gemini gemini-2.0-flash (google-generativeai SDK)")
            if LLM_PROVIDER is None:
                LLM_PROVIDER = "gemini_old"
    except Exception as e:
        if GEMINI_API_KEY:
            print(f"ℹ️  google-generativeai: {e}")

if LLM_PROVIDER is None and OPENROUTER_API_KEY:
    try:
        import openai as _oa
        _openrouter_client = _oa.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        LLM_PROVIDER = "openrouter"
        print("✅  LLM provider: OpenRouter")
    except ImportError:
        print("⚠️  pip install openai")

if LLM_PROVIDER is None and COHERE_API_KEY:
    try:
        import cohere
        _cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
        LLM_PROVIDER = "cohere"
        print("✅  LLM provider: Cohere")
    except ImportError:
        print("⚠️  pip install cohere")

if LLM_PROVIDER is None:
    print("⚠️  No LLM provider. Local word-map fallback only.")

LLM_AVAILABLE = LLM_PROVIDER is not None


def _call_llm(prompt: str) -> str:
    """English generation — Groq preferred."""
    if _groq_client:
        resp = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    return _call_gemini_translate(prompt)


def _call_gemini_translate(prompt: str) -> str:
    """Malayalam translation — Gemini 2.0 Flash preferred."""
    if _GEMINI_ML_PROVIDER == "new":
        return _gemini_translate.models.generate_content(
            model="gemini-3-flash-preview", contents=prompt
        ).text.strip()
    if _GEMINI_ML_PROVIDER == "old":
        return _gemini_translate.generate_content(prompt).text.strip()
    if _groq_client:
        resp = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    if _openrouter_client:
        resp = _openrouter_client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    if _cohere_client:
        resp = _cohere_client.chat(
            model="command-r",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return resp.message.content[0].text.strip()
    raise RuntimeError("No translation provider configured")

def _extract_retry_seconds(err_text: str) -> int:
    for pat in [
        r"retry_delay\s*\{\s*seconds:\s*(\d+)",
        r"retryDelay'\s*:\s*'(\d+)s'",
        r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"Please try again in ([0-9]+(?:\.[0-9]+)?)s",
    ]:
        m = re.search(pat, err_text, flags=re.IGNORECASE)
        if m:
            try:
                return max(1, int(float(m.group(1))))
            except Exception:
                continue
    return 0


# ═══════════════════════════════════════════
# gTTS — Malayalam / English speech synthesis
# ═══════════════════════════════════════════
TTS_AVAILABLE = False
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    print("✅  gTTS available")
except ImportError:
    print("⚠️  pip install gtts  (TTS disabled)")


def synthesize_speech(text: str, lang: str = "ml") -> str | None:
    """Return base64-encoded MP3 bytes, or None on failure."""
    if not TTS_AVAILABLE or not text.strip():
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"❌  TTS error: {e}")
        return None


# ═══════════════════════════════════════════
# MEDIAPIPE + TFLITE SETUP
# ═══════════════════════════════════════════
with open(SCALER_JSON, "r", encoding="utf-8") as f:
    _sc = json.load(f)
_mean  = np.array(_sc["mean"],  dtype=np.float32)
_scale = np.array(_sc["scale"], dtype=np.float32)
_scale = np.where(_scale == 0.0, 1.0, _scale)

with open(LABELS_JSON, "r", encoding="utf-8") as f:
    labels = np.array(json.load(f))

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose

hands = mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5,
    min_tracking_confidence=0.5, max_num_hands=2,
)
pose = mp_pose.Pose(
    model_complexity=0, min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

client_buffers: Dict[str, deque] = {}
# Stores the previous keypoint vector per client for motion delta calculation
client_prev_kp: Dict[str, np.ndarray] = {}


def compute_motion(prev: np.ndarray, curr: np.ndarray) -> float:
    """L2 norm of hand keypoint delta (first 126 values = both hands)."""
    if prev is None:
        return 0.0
    return float(np.linalg.norm(curr[:126] - prev[:126]))


def normalize_hand(hand_flat: np.ndarray) -> np.ndarray:
    pts = hand_flat.reshape(21, 3).copy()
    pts -= pts[0]
    s = np.linalg.norm(pts[9])
    if s > 1e-6:
        pts /= s
    return pts.flatten().astype(np.float32)


def normalize_pose(pose_flat: np.ndarray) -> np.ndarray:
    pts = pose_flat.reshape(33, 3).copy()
    mid = (pts[11] + pts[12]) / 2.0
    pts -= mid
    w = np.linalg.norm(pts[11] - pts[12])
    if w > 1e-6:
        pts /= w
    return pts.flatten().astype(np.float32)


def extract_keypoints(hand_results, pose_results) -> np.ndarray:
    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for lm, hd in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            raw  = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
            norm = normalize_hand(raw)
            if hd.classification[0].label == "Left":
                lh = norm
            else:
                rh = norm
    pv = np.zeros(99, dtype=np.float32)
    if pose_results.pose_landmarks:
        raw = np.array([[p.x, p.y, p.z] for p in pose_results.pose_landmarks.landmark], dtype=np.float32).flatten()
        pv = normalize_pose(raw)
    return np.concatenate([lh, rh, pv])


def decode_image(data_url: str) -> np.ndarray:
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Invalid image data")
    return frame_bgr


# ═══════════════════════════════════════════
# LLM PIPELINE  (thread-safe, queue-based)
# ═══════════════════════════════════════════
class LLMPipeline:
    def __init__(self):
        self.status   = "idle"
        self.last_en  = ""
        self.last_ml  = ""
        self.last_audio = None   # base64 mp3
        self._thread  = None
        self._queue: deque = deque()
        self._lock    = threading.Lock()
        self._quota_block_until = 0.0

    def _fallback(self, words):
        en = " ".join(words)
        ml = " ".join(WORD_MAP.get(w, w) for w in words)
        return en, ml

    def build(self, words: list) -> bool:
        if not words:
            return False
        with self._lock:
            self._queue.append(list(words))
        self._start_next()
        return True

    def pending_count(self) -> int:
        with self._lock:
            running = 1 if (self._thread and self._thread.is_alive()) else 0
            return running + len(self._queue)

    def clear(self):
        with self._lock:
            self._queue.clear()
        self.status = "idle"
        self.last_en = self.last_ml = ""
        self.last_audio = None

    def _start_next(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if not self._queue:
                if self.status == "thinking":
                    self.status = "idle"
                return
            words = self._queue.popleft()
            self.status  = "thinking"
            use_fallback = (not LLM_AVAILABLE) or (time.time() < self._quota_block_until)
            fn = self._run_fallback if use_fallback else self._run
            self._thread = threading.Thread(target=fn, args=(words,), daemon=True)
            self._thread.start()

    def _run_fallback(self, words):
        en, ml = self._fallback(words)
        self._finish(en, ml, "done")

    def _run(self, words):
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
                    if any(x in msg for x in ("429", "rate_limit", "quota", "RateLimitError")):
                        wait = _extract_retry_seconds(msg)
                        self._quota_block_until = time.time() + max(wait, 30)
                        raise
                    raise

            en = ml = ""
            for ln in raw.splitlines():
                low = ln.lower()
                if low.startswith("english:"):
                    en = ln.split(":", 1)[1].strip().strip("\"'")
                elif low.startswith("malayalam:"):
                    ml = ln.split(":", 1)[1].strip().strip("\"'")
            if not en or not ml:
                clean = [l.strip() for l in raw.splitlines() if l.strip()]
                if len(clean) >= 2:
                    en, ml = clean[0], clean[1]
            if not en or not ml:
                en, ml = self._fallback(words)

            self._finish(en, ml, "done")
        except Exception as e:
            en, ml = self._fallback(words)
            self._finish(en, ml, "error")
            print(f"❌  LLM error: {str(e)[:200]}")

    def _finish(self, en, ml, status):
        self.last_en = en
        self.last_ml = ml
        self.status  = status
        self.last_audio = synthesize_speech(ml, lang="ml")
        self._start_next()

    def snapshot(self):
        return {
            "status":   self.status,
            "english":  self.last_en,
            "malayalam": self.last_ml,
            "audio_b64": self.last_audio,
        }


# Shared pipeline instance per session (keyed by client_id)
_pipelines: Dict[str, LLMPipeline] = {}
_pipeline_lock = threading.Lock()

def get_pipeline(client_id: str) -> LLMPipeline:
    with _pipeline_lock:
        if client_id not in _pipelines:
            _pipelines[client_id] = LLMPipeline()
        return _pipelines[client_id]


# ═══════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Swaram· Malayalam Sign Interpreter</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+Malayalam:wght@400;600;700&family=Space+Mono:wght@400;700&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:        #0b0c10;
    --surface:   #13151b;
    --surface2:  #1c1f2a;
    --border:    #2a2d3a;
    --accent:    #f4a237;
    --accent2:   #e05c5c;
    --green:     #3ddc84;
    --blue:      #4fc3f7;
    --text:      #e8eaf0;
    --muted:     #6b7080;
    --radius:    14px;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'Instrument Serif', serif;
    --font-ml:   'Noto Serif Malayalam', serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
  }

  /* ── NOISE OVERLAY ── */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
  }

  /* ── HEADER ── */
  header {
    position: relative; z-index: 10;
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(11,12,16,0.92);
    backdrop-filter: blur(12px);
  }
  .logo {
    display: flex; align-items: baseline; gap: 10px;
  }
  .logo-mark {
    font-family: var(--font-ml);
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
  }
  .logo-sub {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }
  .header-pills { display: flex; gap: 8px; flex-wrap: wrap; }
  .pill {
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .pill.ok { border-color: var(--green); color: var(--green); }
  .pill.warn { border-color: var(--accent); color: var(--accent); }
  .pill.err  { border-color: var(--accent2); color: var(--accent2); }

  /* ── LAYOUT ── */
  main {
    position: relative; z-index: 1;
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 380px;
    grid-template-rows: auto 1fr;
    gap: 20px;
    padding: 24px 24px 24px 24px;
    max-width: 1400px;
    width: 100%;
    margin: 0 auto;
  }

  /* ── CAMERA PANEL ── */
  .cam-panel {
    grid-column: 1; grid-row: 1 / 3;
    display: flex; flex-direction: column; gap: 16px;
  }
  .cam-wrap {
    position: relative;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
    background: #000;
    aspect-ratio: 16/9;
  }
  #video {
    width: 100%; height: 100%;
    object-fit: cover;
    transform: scaleX(-1);
    display: block;
  }
  #canvas { display: none; }

  /* Overlay HUD inside camera */
  .hud {
    position: absolute; inset: 0;
    pointer-events: none;
  }
  .hud-top {
    position: absolute; top: 12px; left: 12px; right: 12px;
    display: flex; gap: 8px; align-items: flex-start;
  }
  .hud-state {
    padding: 5px 14px;
    border-radius: 6px;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    background: rgba(0,0,0,0.7);
    border: 1px solid var(--border);
    transition: all 0.2s;
  }
  .hud-state.idle       { color: var(--muted); }
  .hud-state.collecting { color: var(--blue); border-color: var(--blue); }
  .hud-state.active     { color: var(--green); border-color: var(--green); }
  .hud-state.ready      { color: var(--green); border-color: var(--green); }
  .hud-state.error      { color: var(--accent2); border-color: var(--accent2); }

  .hud-idle-count {
    font-size: 0.6rem;
    color: var(--accent);
    background: rgba(0,0,0,0.7);
    padding: 5px 10px; border-radius: 6px;
    border: 1px solid var(--accent);
    letter-spacing: 0.06em;
    min-width: 110px;
    text-align: center;
    transition: opacity 0.2s;
  }
  .hud-motion {
    font-size: 0.58rem;
    color: var(--muted);
    background: rgba(0,0,0,0.6);
    padding: 4px 8px; border-radius: 6px;
  }

  .hud-frames {
    margin-left: auto;
    font-size: 0.6rem; color: var(--muted);
    background: rgba(0,0,0,0.7);
    padding: 5px 10px; border-radius: 6px;
    border: 1px solid var(--border);
  }

  /* Prediction badge at bottom of camera */
  .hud-pred {
    position: absolute; bottom: 0; left: 0; right: 0;
    padding: 10px 16px;
    background: linear-gradient(transparent, rgba(0,0,0,0.92));
    display: flex; align-items: center; gap: 12px;
  }
  #predLabel {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: var(--font-ml);
    color: var(--green);
    letter-spacing: 1px;
  }
  #predConf {
    font-size: 0.75rem;
    color: var(--muted);
  }
  /* Confidence bar */
  .conf-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  #confBar {
    height: 100%;
    width: 0%;
    background: var(--green);
    border-radius: 2px;
    transition: width 0.3s, background 0.3s;
  }

  /* ── CONTROLS BAR ── */
  .controls {
    display: flex; gap: 10px; flex-wrap: wrap;
  }
  .btn {
    padding: 10px 20px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: all 0.18s;
    text-transform: uppercase;
  }
  .btn:hover  { background: var(--surface2); border-color: var(--muted); }
  .btn.primary { background: var(--accent); border-color: var(--accent); color: #000; font-weight: 700; }
  .btn.primary:hover { filter: brightness(1.1); }
  .btn.danger  { border-color: var(--accent2); color: var(--accent2); }
  .btn.danger:hover { background: rgba(224,92,92,0.12); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ── WORDS STRIP ── */
  .words-strip {
    display: flex; gap: 8px; flex-wrap: wrap;
    min-height: 44px; align-items: center;
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
  }
  .word-chip {
    padding: 4px 14px;
    border-radius: 100px;
    background: var(--surface2);
    border: 1px solid var(--border);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: var(--accent);
    animation: chipIn 0.25s ease;
  }
  @keyframes chipIn {
    from { opacity: 0; transform: scale(0.8) translateY(4px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
  }
  .words-empty { color: var(--muted); font-size: 0.7rem; letter-spacing: 0.08em; }

  /* ── RIGHT PANEL ── */
  .right-panel {
    grid-column: 2; grid-row: 1 / 3;
    display: flex; flex-direction: column; gap: 16px;
  }

  /* ── CARD ── */
  .card {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    overflow: hidden;
  }
  .card-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .card-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
    transition: background 0.3s;
  }
  .card-dot.active { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .card-dot.busy   { background: var(--blue);  box-shadow: 0 0 6px var(--blue); animation: pulse 1s infinite; }
  .card-dot.err    { background: var(--accent2); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .card-body { padding: 16px; }

  /* ── SENTENCE DISPLAY ── */
  .sentence-en {
    font-family: var(--font-body);
    font-style: italic;
    font-size: 1.05rem;
    color: var(--text);
    line-height: 1.5;
    min-height: 32px;
  }
  .sentence-ml {
    font-family: var(--font-ml);
    font-size: 1.5rem;
    color: var(--accent);
    margin-top: 12px;
    line-height: 1.6;
    min-height: 40px;
  }
  .sentence-placeholder {
    color: var(--muted);
    font-size: 0.75rem;
    font-style: italic;
  }

  /* ── AUDIO PLAYER ── */
  #audioPlayer { display: none; }
  .audio-btn {
    margin-top: 14px;
    display: flex; align-items: center; gap: 10px;
  }
  .play-btn {
    width: 40px; height: 40px;
    border-radius: 50%;
    border: 1px solid var(--accent);
    background: transparent;
    color: var(--accent);
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    transition: all 0.2s;
  }
  .play-btn:hover { background: rgba(244,162,55,0.15); }
  .play-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .audio-label { font-size: 0.65rem; color: var(--muted); letter-spacing: 0.08em; }

  /* ── LLM STATUS ── */
  .llm-status {
    padding: 10px 16px;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    border-top: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
  }
  .spinner {
    width: 10px; height: 10px;
    border: 2px solid var(--border);
    border-top-color: var(--blue);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── TOP PREDICTIONS ── */
  .top-preds { display: flex; flex-direction: column; gap: 8px; }
  .top-pred-row {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.7rem;
  }
  .top-pred-label { width: 90px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .top-pred-bar-bg {
    flex: 1; height: 6px; border-radius: 3px;
    background: var(--surface2);
    overflow: hidden;
  }
  .top-pred-bar { height: 100%; border-radius: 3px; background: var(--accent); transition: width 0.4s; }
  .top-pred-pct { width: 40px; text-align: right; color: var(--muted); font-size: 0.62rem; }

  /* ── LOG ── */
  .log {
    font-size: 0.62rem;
    color: var(--muted);
    line-height: 1.8;
    max-height: 160px;
    overflow-y: auto;
    padding: 12px 16px;
  }
  .log::-webkit-scrollbar { width: 4px; }
  .log::-webkit-scrollbar-track { background: transparent; }
  .log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .log-entry { border-bottom: 1px solid var(--border); padding: 4px 0; }
  .log-entry:last-child { border-bottom: none; }
  .log-ts { color: var(--blue); margin-right: 6px; }
  .log-ok   { color: var(--green); }
  .log-warn { color: var(--accent); }
  .log-err  { color: var(--accent2); }

  /* ── SETTINGS PANEL ── */
  .settings-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
    padding: 14px 16px;
  }
  .setting-item { display: flex; flex-direction: column; gap: 4px; }
  .setting-label { font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
  .setting-value { font-size: 0.75rem; color: var(--text); }

  /* ── FOOTER ── */
  footer {
    position: relative; z-index: 1;
    padding: 14px 32px;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.58rem;
    color: var(--muted);
    letter-spacing: 0.08em;
  }

  /* ── RESPONSIVE ── */
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; grid-template-rows: auto; }
    .cam-panel { grid-column: 1; grid-row: 1; }
    .right-panel { grid-column: 1; grid-row: 2; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <span class="logo-mark">Swaram</span>
    <span class="logo-sub">Malayalam Sign Interpreter</span>
  </div>
  <div class="header-pills">
    <span class="pill" id="pillLlm">LLM: —</span>
    <span class="pill" id="pillTts">TTS: —</span>
    <span class="pill" id="pillModel">Model: Loading…</span>
  </div>
</header>

<main>
  <!-- ── CAMERA PANEL ── -->
  <div class="cam-panel">
    <div class="cam-wrap">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="canvas"></canvas>
      <div class="hud">
        <div class="hud-top">
          <div class="hud-state idle" id="hudState">IDLE</div>
          <div class="hud-idle-count" id="idleCountdown" style="display:none"></div>
          <div class="hud-frames" id="hudFrames">0 / 30 frames</div>
          <div class="hud-motion" id="hudMotion">motion: 0.000</div>
        </div>
        <div class="hud-pred">
          <span id="predLabel">—</span>
          <span id="predConf"></span>
          <div class="conf-bar-bg"><div id="confBar"></div></div>
        </div>
      </div>
    </div>

    <!-- Controls -->
    <div class="controls">
      <button class="btn primary" id="btnStart">▶ Start Camera</button>
      <button class="btn" id="btnSpeak" disabled>🔊 Speak Now</button>
      <button class="btn danger" id="btnClear" disabled>✕ Clear</button>
      <button class="btn" id="btnReset">↺ Reset Buffer</button>
    </div>

    <!-- Collected words -->
    <div class="words-strip" id="wordsStrip">
      <span class="words-empty">Recognized signs will appear here…</span>
    </div>
  </div>

  <!-- ── RIGHT PANEL ── -->
  <div class="right-panel">

    <!-- Translation card -->
    <div class="card">
      <div class="card-header">
        <div class="card-dot" id="llmDot"></div>
        Translation
      </div>
      <div class="card-body">
        <div class="sentence-en" id="sentEn">
          <span class="sentence-placeholder">Waiting for signs…</span>
        </div>
        <div class="sentence-ml" id="sentMl"></div>
        <div class="audio-btn">
          <button class="play-btn" id="playBtn" disabled title="Play Malayalam audio">▶</button>
          <span class="audio-label" id="audioLabel">No audio yet</span>
        </div>
        <audio id="audioPlayer"></audio>
      </div>
      <div class="llm-status" id="llmStatus">
        <div class="spinner" id="spinner"></div>
        <span id="llmStatusText">Idle</span>
      </div>
    </div>

    <!-- Top predictions card -->
    <div class="card">
      <div class="card-header">
        <div class="card-dot" id="predDot"></div>
        Top Predictions
      </div>
      <div class="card-body">
        <div class="top-preds" id="topPreds">
          <div class="sentence-placeholder" style="font-size:0.72rem">No prediction yet</div>
        </div>
      </div>
    </div>

    <!-- System info card -->
    <div class="card">
      <div class="card-header">
        <div class="card-dot active"></div>
        System
      </div>
      <div class="settings-grid" id="sysGrid">
        <div class="setting-item">
          <span class="setting-label">LLM Provider</span>
          <span class="setting-value" id="sysLlm">—</span>
        </div>
        <div class="setting-item">
          <span class="setting-label">TTS</span>
          <span class="setting-value" id="sysTts">—</span>
        </div>
        <div class="setting-item">
          <span class="setting-label">Threshold</span>
          <span class="setting-value">85%</span>
        </div>
        <div class="setting-item">
          <span class="setting-label">Sequence</span>
          <span class="setting-value">30 frames</span>
        </div>
      </div>
    </div>

    <!-- Log card -->
    <div class="card" style="flex:1">
      <div class="card-header">
        <div class="card-dot active"></div>
        Activity Log
      </div>
      <div class="log" id="actLog">
        <div class="log-entry"><span class="log-ts">—</span>System ready</div>
      </div>
    </div>

  </div>
</main>

<footer>
  <span>Swaram · Malayalam Sign Language Interpreter</span>
  <span id="fpsDisplay">FPS: —</span>
</footer>

<script>
'use strict';
// ─── CONFIG ──────────────────────────────────────────────────────────────────
const CLIENT_ID       = 'client_' + Math.random().toString(36).slice(2, 9);
const SEND_EVERY      = 100;    // ms between frame sends
const AUTO_LLM_SEC    = 2.0;    // idle seconds before auto LLM trigger
const MIN_AUTO_LLM    = 1;      // minimum words before auto-trigger
const MOTION_THRESH   = 0.08;   // below this = no meaningful movement (raised from 0.03)
const IDLE_CONFIRM_MS = 500;    // ms of consecutive low-motion to confirm IDLE

// ─── STATE ───────────────────────────────────────────────────────────────────
let streaming        = false;
let sendTimer        = null;
let words            = [];
let lastSignTime     = 0;
let silenceTimer     = null;     // fires LLM after 2s idle
let idleCountTimer   = null;     // updates the on-screen countdown
let idleStartTime    = 0;        // when we first went idle (ms)
let isIdle           = true;     // current motion state
let idleLowCount     = 0;        // consecutive low-motion frames
let llmBusy          = false;
let silenceTriggered = false;    // prevent double-trigger per idle period
let currentAudio     = null;
let fpsHistory       = [];
let lastFpsTime      = performance.now();

// ─── DOM ─────────────────────────────────────────────────────────────────────
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const hudState    = document.getElementById('hudState');
const hudFrames   = document.getElementById('hudFrames');
const hudMotion   = document.getElementById('hudMotion');
const idleCountEl = document.getElementById('idleCountdown');
const predLabel   = document.getElementById('predLabel');
const predConf    = document.getElementById('predConf');
const confBar     = document.getElementById('confBar');
const wordsStrip  = document.getElementById('wordsStrip');
const btnStart    = document.getElementById('btnStart');
const btnSpeak    = document.getElementById('btnSpeak');
const btnClear    = document.getElementById('btnClear');
const btnReset    = document.getElementById('btnReset');
const sentEn      = document.getElementById('sentEn');
const sentMl      = document.getElementById('sentMl');
const playBtn     = document.getElementById('playBtn');
const audioLabel  = document.getElementById('audioLabel');
const audioEl     = document.getElementById('audioPlayer');
const llmDot      = document.getElementById('llmDot');
const llmStatus   = document.getElementById('llmStatusText');
const spinner     = document.getElementById('spinner');
const topPreds    = document.getElementById('topPreds');
const actLog      = document.getElementById('actLog');
const fpsDisplay  = document.getElementById('fpsDisplay');
const pillLlm     = document.getElementById('pillLlm');
const pillTts     = document.getElementById('pillTts');
const pillModel   = document.getElementById('pillModel');
const sysLlm      = document.getElementById('sysLlm');
const sysTts      = document.getElementById('sysTts');

// ─── INIT ─────────────────────────────────────────────────────────────────────
(async () => {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    pillLlm.textContent = 'LLM: ' + (d.llm_provider || 'none');
    pillLlm.className = 'pill ' + (d.llm_available ? 'ok' : 'warn');
    pillTts.textContent = 'TTS: ' + (d.tts_available ? 'gTTS' : 'off');
    pillTts.className = 'pill ' + (d.tts_available ? 'ok' : 'warn');
    pillModel.textContent = 'Model: ' + (d.model_loaded ? 'Ready' : 'Error');
    pillModel.className = 'pill ' + (d.model_loaded ? 'ok' : 'err');
    sysLlm.textContent = d.llm_provider || 'none';
    sysTts.textContent = d.tts_available ? 'gTTS (Malayalam)' : 'Unavailable';
    log('System initialized. LLM=' + (d.llm_provider||'none'), 'ok');
  } catch(e) { log('Status check failed: ' + e, 'err'); }
})();

// ─── CAMERA ──────────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  if (streaming) { stopStreaming(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    await video.play();
    streaming = true;
    btnStart.textContent = '⏹ Stop Camera';
    btnStart.className = 'btn danger';
    log('Camera started', 'ok');
    startSending();
    startPolling();
  } catch(e) {
    log('Camera error: ' + e.message, 'err');
    setHudState('error');
  }
});

function stopStreaming() {
  streaming = false;
  clearInterval(sendTimer);
  clearTimeout(silenceTimer);
  clearInterval(idleCountTimer);
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  btnStart.textContent = '▶ Start Camera';
  btnStart.className = 'btn primary';
  setHudState('idle');
  idleCountEl.style.display = 'none';
  log('Camera stopped', 'warn');
}

// ─── FRAME SENDING ────────────────────────────────────────────────────────────
function startSending() {
  sendTimer = setInterval(sendFrame, SEND_EVERY);
}

async function sendFrame() {
  if (!streaming) return;
  if (video.readyState < 2) return;

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  ctx.drawImage(video, 0, 0);
  const imageData = canvas.toDataURL('image/jpeg', 0.7);

  // FPS tracking
  const now = performance.now();
  fpsHistory.push(now - lastFpsTime);
  lastFpsTime = now;
  if (fpsHistory.length > 20) fpsHistory.shift();
  const avgMs = fpsHistory.reduce((a,b)=>a+b,0) / fpsHistory.length;
  fpsDisplay.textContent = 'FPS: ' + Math.round(1000 / avgMs);

  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: CLIENT_ID, image: imageData }),
    });
    const d = await r.json();
    handlePrediction(d);
  } catch(e) {
    // silent network errors
  }
}

// ─── MOTION + IDLE STATE MACHINE ─────────────────────────────────────────────
function handlePrediction(d) {
  if (d.error) { setHudState('error'); return; }

  const handSeen = d.hand_detected ?? false;
  const motion   = d.motion ?? 0;

  if (hudMotion) hudMotion.textContent = (handSeen ? '\u270b ' : '\u00b7 ') + 'motion: ' + motion.toFixed(3);

  if (handSeen) {
    // HAND PRESENT -> cancel idle, mark active
    idleLowCount = 0;
    if (isIdle) {
      isIdle = false;
      silenceTriggered = false;
      clearTimeout(silenceTimer);
      clearInterval(idleCountTimer);
      idleCountEl.style.display = 'none';
    }
  } else {
    // NO HAND -> count frames toward idle threshold
    idleLowCount++;
    if (!isIdle && idleLowCount * SEND_EVERY >= IDLE_CONFIRM_MS) {
      // Confirmed idle — flip state and start countdown
      isIdle = true;
      setHudState('idle');
      predLabel.textContent = '\u2014';
      predConf.textContent  = '';
      confBar.style.width   = '0%';
      hudFrames.textContent = '0 / 30 frames';
      topPreds.innerHTML    = '<div class="sentence-placeholder" style="font-size:0.72rem">No prediction yet</div>';
      document.getElementById('predDot').className = 'card-dot';
      if (words.length >= MIN_AUTO_LLM && !llmBusy && !silenceTriggered) {
        startIdleCountdown();
      }
    }
    return; // no hand -> nothing more to process
  }

  // Hand is present: process prediction
  if (d.status === 'collecting') {
    setHudState('collecting');
    hudFrames.textContent = d.frames + ' / ' + d.needed + ' frames';
    return;
  }

  if (d.status === 'ok') {
    hudFrames.textContent = '30 / 30 frames';
    setHudState('active');
    renderTopPreds(d.top_predictions || []);
    document.getElementById('predDot').className = 'card-dot active';

    if (d.accepted) {
      const lbl  = d.prediction;
      const conf = d.confidence;
      predLabel.textContent = lbl;
      predConf.textContent  = (conf * 100).toFixed(1) + '%';
      confBar.style.width   = (conf * 100) + '%';
      confBar.style.background = conf > 0.9 ? 'var(--green)' : 'var(--accent)';

      if (words.length === 0 || words[words.length - 1] !== lbl) {
        addWord(lbl);
        lastSignTime     = Date.now();
        silenceTriggered = false;
        clearTimeout(silenceTimer);
        clearInterval(idleCountTimer);
        idleCountEl.style.display = 'none';
      }
    }
  }
}

// ─── IDLE COUNTDOWN → LLM TRIGGER ────────────────────────────────────────────
function startIdleCountdown() {
  clearTimeout(silenceTimer);
  clearInterval(idleCountTimer);

  const deadline = Date.now() + AUTO_LLM_SEC * 1000;
  idleCountEl.style.display = 'block';

  // Tick every 100ms to update countdown display
  idleCountTimer = setInterval(() => {
    const remaining = (deadline - Date.now()) / 1000;
    if (remaining <= 0) {
      clearInterval(idleCountTimer);
      idleCountEl.style.display = 'none';
    } else {
      idleCountEl.textContent = 'sending in ' + remaining.toFixed(1) + 's…';
    }
  }, 100);

  silenceTimer = setTimeout(async () => {
    clearInterval(idleCountTimer);
    idleCountEl.style.display = 'none';
    if (words.length >= MIN_AUTO_LLM && !llmBusy && !silenceTriggered) {
      silenceTriggered = true;
      log('Idle 2s → auto LLM: ' + words.join(', '), 'ok');
      triggerLLM();
    }
    // Always reset the frame buffer after a pause so stale frames don't
    // contaminate the next sign sequence
    try {
      await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: CLIENT_ID }),
      });
      log('Frame buffer cleared after pause', 'warn');
    } catch(_) {}
  }, AUTO_LLM_SEC * 1000);
}

// ─── WORDS ───────────────────────────────────────────────────────────────────
function addWord(w) {
  words.push(w);
  if (words.length > 10) words.shift();
  renderWords();
  btnSpeak.disabled = false;
  btnClear.disabled = false;
  log('Recognized: ' + w, 'ok');
}

function renderWords() {
  if (words.length === 0) {
    wordsStrip.innerHTML = '<span class="words-empty">Recognized signs will appear here…</span>';
    btnSpeak.disabled = true;
    btnClear.disabled = true;
    return;
  }
  wordsStrip.innerHTML = words.map(w =>
    `<span class="word-chip">${w}</span>`
  ).join('');
}

// ─── LLM ─────────────────────────────────────────────────────────────────────
function triggerLLM() {
  if (!words.length) return;
  // Snapshot current words for the request, then immediately clear
  const snapshot = [...words];
  words = [];
  renderWords();
  silenceTriggered = false;
  llmBusy = true;
  setLlmStatus('thinking');
  fetch('/llm', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ client_id: CLIENT_ID, words: snapshot }),
  })
  .then(r => r.text().then(text => {
    try { return JSON.parse(text); }
    catch(_) { throw new Error('Server ' + r.status + ': non-JSON response'); }
  }))
  .then(d => {
    llmBusy = false;
    if (d.error) { setLlmStatus('error'); log('LLM error: ' + d.error, 'err'); return; }
    setLlmStatus('done');
    sentEn.innerHTML = d.english  || '<span class="sentence-placeholder">No result</span>';
    sentMl.textContent = d.malayalam || '';
    log('EN: ' + (d.english || ''), 'ok');
    log('ML: ' + (d.malayalam || ''), 'ok');
    if (d.audio_b64) {
      currentAudio = d.audio_b64;
      playBtn.disabled = false;
      audioLabel.textContent = 'Malayalam audio ready';
      playAudio(d.audio_b64);
    }
  })
  .catch(e => {
    llmBusy = false;
    setLlmStatus('error');
    log('LLM failed: ' + e.message, 'err');
  });
}

function playAudio(b64) {
  const src = 'data:audio/mpeg;base64,' + b64;
  audioEl.src = src;
  audioEl.play().catch(e => log('Audio play error: ' + e, 'warn'));
}

// ─── LLM STATUS POLLING ───────────────────────────────────────────────────────
function startPolling() {
  // No server-side state needed since LLM is synchronous per request
}

// ─── BUTTONS ─────────────────────────────────────────────────────────────────
btnSpeak.addEventListener('click', () => {
  if (words.length) { triggerLLM(); log('Manual speak triggered', 'ok'); }
});

btnClear.addEventListener('click', () => {
  words = [];
  renderWords();
  predLabel.textContent = '—';
  predConf.textContent  = '';
  confBar.style.width   = '0%';
  clearTimeout(silenceTimer);
  clearInterval(idleCountTimer);
  idleCountEl.style.display = 'none';
  silenceTriggered = false;
  log('Cleared word buffer', 'warn');
});

btnReset.addEventListener('click', async () => {
  words = [];
  renderWords();
  clearTimeout(silenceTimer);
  clearInterval(idleCountTimer);
  idleCountEl.style.display = 'none';
  silenceTriggered = false;
  await fetch('/reset', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ client_id: CLIENT_ID }),
  });
  log('Frame buffer reset', 'warn');
});

playBtn.addEventListener('click', () => {
  if (currentAudio) playAudio(currentAudio);
});

function setHudState(s) {
  hudState.className = 'hud-state ' + s;
  const labels = { idle:'IDLE', collecting:'COLLECTING', active:'ACTIVE', ready:'READY', error:'ERROR' };
  hudState.textContent = labels[s] || s.toUpperCase();
}

// ─── LLM STATUS POLLING ─────────────────────────────────────────────────────
function startPolling() { /* LLM is synchronous per request — no polling needed */ }

function setLlmStatus(s) {
  const labels = { idle:'Idle', thinking:'Processing…', done:'Done', error:'Error' };
  llmStatus.textContent = labels[s] || s;
  spinner.style.display = s === 'thinking' ? 'block' : 'none';
  llmDot.className = 'card-dot' + (s==='thinking'?' busy': s==='done'?' active': s==='error'?' err':'');
}

function renderTopPreds(preds) {
  if (!preds || !preds.length) return;
  topPreds.innerHTML = preds.map(p => {
    const pct = (p.confidence * 100).toFixed(1);
    return `
      <div class="top-pred-row">
        <span class="top-pred-label">${p.label}</span>
        <div class="top-pred-bar-bg">
          <div class="top-pred-bar" style="width:${pct}%"></div>
        </div>
        <span class="top-pred-pct">${pct}%</span>
      </div>`;
  }).join('');
}

function log(msg, type) {
  const ts = new Date().toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  const cls = type === 'ok' ? 'log-ok' : type === 'warn' ? 'log-warn' : type === 'err' ? 'log-err' : '';
  const el = document.createElement('div');
  el.className = 'log-entry';
  el.innerHTML = `<span class="log-ts">${ts}</span><span class="${cls}">${msg}</span>`;
  actLog.prepend(el);
  // Keep max 40 entries
  while (actLog.children.length > 40) actLog.removeChild(actLog.lastChild);
}
</script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/status")
def status():
    return jsonify({
        "llm_provider":  LLM_PROVIDER or "none",
        "llm_available": LLM_AVAILABLE,
        "tts_available": TTS_AVAILABLE,
        "model_loaded":  True,
        "labels":        labels.tolist(),
    })


@app.post("/predict")
def predict():
    payload   = request.get_json(silent=True) or {}
    client_id = str(payload.get("client_id", "default"))
    image_data = payload.get("image")

    if not image_data or not isinstance(image_data, str) or len(image_data) < 100:
        return jsonify({"error": "Missing or invalid image"}), 400

    try:
        frame_bgr = decode_image(image_data)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        hr = hands.process(frame_rgb)
        pr = pose.process(frame_rgb)
        kp = extract_keypoints(hr, pr)

        # Motion detection
        hand_detected = hr.multi_hand_landmarks is not None
        if client_id not in client_buffers:
            client_buffers[client_id] = deque(maxlen=SEQUENCE_LEN)

        # No hand in frame: don't emit prediction output and clear stale sequence.
        if not hand_detected:
            client_buffers[client_id].clear()
            client_prev_kp.pop(client_id, None)
            return jsonify({
                "status": "collecting",
                "frames": 0,
                "needed": SEQUENCE_LEN,
                "hand_detected": False,
                "motion": 0.0,
            })

        motion_val = compute_motion(client_prev_kp.get(client_id), kp)
        client_prev_kp[client_id] = kp.copy()
        client_buffers[client_id].append(kp)

        buf = client_buffers[client_id]
        if len(buf) < SEQUENCE_LEN:
            return jsonify({
                "status": "collecting",
                "frames": len(buf),
                "needed": SEQUENCE_LEN,
                "hand_detected": hand_detected,
                "motion": round(motion_val, 4),
            })

        seq    = np.stack(buf, axis=0).astype(np.float32)
        flat   = seq.reshape(-1, N_FEATURES)
        scaled = (flat - _mean) / _scale
        inp    = scaled.reshape(1, SEQUENCE_LEN, N_FEATURES).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        pred_idx   = int(np.argmax(output))
        confidence = float(output[pred_idx])

        top_idx = np.argsort(output)[::-1][:3]
        top_preds = [
            {"label": str(labels[i]), "confidence": float(output[i])}
            for i in top_idx
        ]

        return jsonify({
            "status":          "ok",
            "prediction":      str(labels[pred_idx]),
            "confidence":      confidence,
            "accepted":        confidence >= THRESHOLD,
            "top_predictions": top_preds,
            "hand_detected":   hand_detected,
            "motion":          round(motion_val, 4),
        })
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/llm")
def llm_endpoint():
    """Groq → English, Gemini → Malayalam."""
    try:
        payload   = request.get_json(silent=True) or {}
        client_id = str(payload.get("client_id", "default"))
        words_in  = payload.get("words", [])
        if not words_in:
            return jsonify({"error": "No words provided"}), 400

        prompt_en = (
            "You are a sign language interpreter.\n"
            "Input sign words (in order): " + ", ".join(words_in) + "\n\n"
            "Write ONE natural, fluent English sentence from these signs.\n"
            "Reply with ONLY the sentence \u2014 no labels, no explanation."
        )

        def prompt_ml(en_s):
            return (
                "You are a professional Malayalam translator.\n\n"
                "Translate the following English sentence into natural, grammatically "
                "correct Malayalam script (Unicode).\n"
                "Rules:\n"
                "  \u2022 Malayalam Unicode script ONLY \u2014 no Roman letters, no transliteration\n"
                "  \u2022 Natural spoken Malayalam, not literal word-for-word\n"
                "  \u2022 Correct SOV grammar and verb conjugation\n\n"
                "English: " + en_s + "\n\n"
                "Reply with ONLY the Malayalam sentence \u2014 nothing else."
            )

        en = ml = ""
        try:
            en = _call_llm(prompt_en).strip().strip("\'\"")
            if not en:
                raise ValueError("Empty English response")
            ml = _call_gemini_translate(prompt_ml(en)).strip().strip("\'\"")
            if not ml or all(ord(c) < 128 for c in ml if c.isalpha()):
                ml = " ".join(WORD_MAP.get(w, w) for w in words_in)
                print("\u26a0\ufe0f  Malayalam ASCII \u2014 word map fallback")
        except Exception as e:
            en = en or " ".join(words_in)
            ml = " ".join(WORD_MAP.get(w, w) for w in words_in)
            print(f"\u274c  LLM error: {e}")

        audio_b64 = synthesize_speech(ml, lang="ml")
        return jsonify({"english": en, "malayalam": ml, "audio_b64": audio_b64})

    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

@app.post("/tts")
def tts_endpoint():
    payload = request.get_json(silent=True) or {}
    text    = payload.get("text", "")
    lang    = payload.get("lang", "ml")
    audio   = synthesize_speech(text, lang=lang)
    if audio is None:
        return jsonify({"error": "TTS unavailable or empty text"}), 400
    return jsonify({"audio_b64": audio})


@app.post("/reset")
def reset():
    payload   = request.get_json(silent=True) or {}
    client_id = str(payload.get("client_id", "default"))
    client_buffers[client_id] = deque(maxlen=SEQUENCE_LEN)
    client_prev_kp.pop(client_id, None)
    if client_id in _pipelines:
        _pipelines[client_id].clear()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"🚀  Starting ചിഹ്നം on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)