"""
STEP 1 — DATA COLLECTION
=========================
Collects sign language keypoints directly (no video files needed).
Saves (30, 225) .npy files: left_hand(63) + right_hand(63) + pose(99)

USAGE
  python step1_collect.py
  python step1_collect.py --signs "Hello,Thanks,Help" --samples 30

CONTROLS (during recording window)
  SPACE : start recording
  N     : next sign
  P     : previous sign
  R     : redo / delete last recording
  Q     : quit and show summary
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "Incompatible mediapipe package detected. This project requires the classic "
        "MediaPipe Solutions API (mp.solutions).*\n"
        "Use this project environment instead:\n"
        "  ./signenv/bin/python step1_collect.py\n"
        "Or reinstall pinned dependencies:\n"
        "  ./signenv/bin/pip install --upgrade --force-reinstall -r requirements.txt"
    )

# ─────────────────────────────────────────────
# CONFIGURATION  (edit these for your project)
# ─────────────────────────────────────────────
DEFAULT_SIGNS = [
    "Hello", "Thanks", "Help",
    "Water", "Food", "Happy",
    "Sorry", "I", "You",
    # additional vocabulary requested by user
    "House", "Cat", "Dog", "Big", "Eat", "Drink",
    "Run", "Play", "Happy", "Give", "Take",
    "See", "Go", "Lake", "Fish",
]

OUTPUT_DIR       = "dataset"          # folder where .npy files are saved
SAMPLES_PER_SIGN = 30                 # recordings per sign
SEQUENCE_LENGTH  = 30                 # frames per recording
COUNTDOWN_SEC    = 3                  # countdown before recording starts

# MediaPipe settings
MP_COMPLEXITY        = 1
MIN_DETECT_CONF      = 0.5
MIN_TRACK_CONF       = 0.5
MIN_QUALITY_RATE     = 0.60           # reject recordings below 60% detection

# Camera
CAM_INDEX    = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
WINDOW_NAME  = "Step 1 — Collect Data"
# ─────────────────────────────────────────────

mp_hands          = mp.solutions.hands
mp_pose           = mp.solutions.pose
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ═══════════════════════════════════════════
# KEYPOINT EXTRACTION
# ═══════════════════════════════════════════
def extract_keypoints(hand_results, pose_results) -> np.ndarray:
    """
    Returns (225,) array:
      [0:63]   left hand  — 21 landmarks × 3 (x,y,z)
      [63:126] right hand — 21 landmarks × 3
      [126:]   pose       — 33 landmarks × 3
    Handedness is assigned from MediaPipe's classification label,
    so Left/Right is always correct regardless of camera flip.
    """
    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for landmarks, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness,
        ):
            arr   = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                             dtype=np.float32).flatten()
            label = handedness.classification[0].label  # 'Left' or 'Right'
            if label == "Left":
                lh = arr
            else:
                rh = arr

    pose = np.zeros(99, dtype=np.float32)
    if pose_results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()

    return np.concatenate([lh, rh, pose])  # (225,)


# ═══════════════════════════════════════════
# COLLECTOR CLASS
# ═══════════════════════════════════════════
class Collector:
    def __init__(self, signs: list[str], samples: int, cam_index: int = CAM_INDEX):
        self.signs   = signs
        self.samples = samples
        self.sign_idx = 0
        self.cam_index = cam_index

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for s in signs:
            os.makedirs(os.path.join(OUTPUT_DIR, s), exist_ok=True)

        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.cam_index}. "
                "Try: --cam 1 (or 2), and close apps that are using webcam."
            )

        # Recording state
        self.recording        = False
        self.countdown_end    = 0.0
        self.recorded_kp      = []
        self.detection_frames = 0
        self.total_frames     = 0

    # ── helpers ──────────────────────────────
    @property
    def current_sign(self) -> str:
        return self.signs[self.sign_idx]

    def existing_count(self, sign: str | None = None) -> int:
        sign = sign or self.current_sign
        path = os.path.join(OUTPUT_DIR, sign)
        return len([f for f in os.listdir(path) if f.endswith(".npy")])

    def start_recording(self):
        self.recording        = True
        self.countdown_end    = time.time() + COUNTDOWN_SEC
        self.recorded_kp      = []
        self.detection_frames = 0
        self.total_frames     = 0
        print(f"\n⏳  Countdown {COUNTDOWN_SEC}s → recording '{self.current_sign}'")

    def delete_last(self):
        path  = os.path.join(OUTPUT_DIR, self.current_sign)
        files = sorted(f for f in os.listdir(path) if f.endswith(".npy"))
        if not files:
            print("  Nothing to delete.")
            return
        fp = os.path.join(path, files[-1])
        os.remove(fp)
        # also remove preview jpg if present
        jpg = fp.replace(".npy", "_preview.jpg")
        if os.path.exists(jpg):
            os.remove(jpg)
        print(f"  🗑  Deleted: {files[-1]}")

    def save_recording(self, frame_for_preview=None) -> bool:
        if not self.recorded_kp:
            return False

        rate = (self.detection_frames / self.total_frames) if self.total_frames else 0
        if rate < MIN_QUALITY_RATE:
            print(f"  ⚠️  Quality {rate*100:.0f}% < {MIN_QUALITY_RATE*100:.0f}% — NOT saved. Try again.")
            return False

        seq = np.array(self.recorded_kp, dtype=np.float32)
        # Pad or crop to SEQUENCE_LENGTH
        if len(seq) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(seq), 225), dtype=np.float32)
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:SEQUENCE_LENGTH]

        assert seq.shape == (30, 225), f"Unexpected shape {seq.shape}"

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        n    = self.existing_count() + 1
        name = f"{self.current_sign}_{n:03d}_{ts}.npy"
        fp   = os.path.join(OUTPUT_DIR, self.current_sign, name)
        np.save(fp, seq)

        if frame_for_preview is not None:
            cv2.imwrite(fp.replace(".npy", "_preview.jpg"), frame_for_preview)

        print(f"  ✅  Saved: {name}  |  quality {rate*100:.0f}%  |  "
              f"{n}/{self.samples}")
        return True

    # ── UI drawing ───────────────────────────
    def draw_ui(self, frame, hands_ok, pose_ok):
        h, w   = frame.shape[:2]
        sign   = self.current_sign
        exist  = self.existing_count()
        now    = time.time()

        # top bar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 145), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # sign name + progress
        cv2.putText(frame, f"Sign: {sign}  ({exist}/{self.samples})",
                    (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        bar_x, bar_y, bar_w, bar_h = 18, 55, 420, 24
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill = int(bar_w * min(exist / self.samples, 1.0))
        color = (0, 220, 120) if exist >= self.samples else (0, 160, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        # sign index
        cv2.putText(frame, f"[{self.sign_idx+1}/{len(self.signs)}]",
                    (18, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

        # detection indicators
        def ind(label, ok, x, y):
            c = (0, 220, 80) if ok else (0, 50, 220)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

        ind("✓ HANDS" if hands_ok else "✗ HANDS", hands_ok, w - 280, 40)
        ind("✓ POSE"  if pose_ok  else "✗ POSE",  pose_ok,  w - 280, 75)

        # recording state
        if self.recording:
            if now < self.countdown_end:
                left = int(self.countdown_end - now) + 1
                cv2.putText(frame, str(left),
                            (w // 2 - 40, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 220, 220), 12)
            else:
                n_rec = len(self.recorded_kp)
                cv2.circle(frame, (w - 45, 160), 18, (0, 0, 220), -1)
                cv2.putText(frame, "REC", (w - 95, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)
                cv2.putText(frame, f"{n_rec}/{SEQUENCE_LENGTH}",
                            (w - 110, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # bottom instructions
        bot = frame.copy()
        cv2.rectangle(bot, (0, h - 90), (w, h), (25, 25, 25), -1)
        cv2.addWeighted(bot, 0.75, frame, 0.25, 0, frame)
        for i, txt in enumerate([
            "SPACE: Record  |  N: Next  |  P: Prev  |  R: Redo last  |  Q: Quit",
            "Stand 3-4 ft back · keep hands & shoulders visible · vary speed/position",
        ]):
            cv2.putText(frame, txt, (15, h - 62 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)

    # ── main loop ────────────────────────────
    def run(self):
        print("\n" + "=" * 65)
        print("🎬  SIGN LANGUAGE DATA COLLECTOR")
        print("=" * 65)
        print(f"   Signs   : {self.signs}")
        print(f"   Target  : {self.samples} samples each")
        print(f"   Output  : {os.path.abspath(OUTPUT_DIR)}/")
        print(f"   Camera  : index {self.cam_index}")
        print("=" * 65 + "\n")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

        mid_frame = None   # saved for preview

        with mp_hands.Hands(
            model_complexity=MP_COMPLEXITY,
            min_detection_confidence=MIN_DETECT_CONF,
            min_tracking_confidence=MIN_TRACK_CONF,
            max_num_hands=2,
        ) as hands, mp_pose.Pose(
            model_complexity=MP_COMPLEXITY,
            min_detection_confidence=MIN_DETECT_CONF,
            min_tracking_confidence=MIN_TRACK_CONF,
        ) as pose:

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame   = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                hr = hands.process(img_rgb)
                pr = pose.process(img_rgb)
                img_rgb.flags.writeable = True

                hands_ok = hr.multi_hand_landmarks is not None
                pose_ok  = pr.pose_landmarks is not None

                # draw landmarks
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

                # recording logic
                if self.recording:
                    now = time.time()
                    if now >= self.countdown_end:
                        kp = extract_keypoints(hr, pr)
                        self.recorded_kp.append(kp)
                        self.total_frames += 1
                        if hands_ok and pose_ok:
                            self.detection_frames += 1

                        n = len(self.recorded_kp)
                        if n == SEQUENCE_LENGTH // 2:
                            mid_frame = frame.copy()
                        if n >= SEQUENCE_LENGTH:
                            self.recording = False
                            saved = self.save_recording(mid_frame)
                            mid_frame = None
                            if saved and self.existing_count() >= self.samples:
                                print(f"\n🎉  '{self.current_sign}' complete!")
                                if self.sign_idx + 1 < len(self.signs):
                                    self.sign_idx += 1
                                    print(f"➡️   Next: '{self.current_sign}'  "
                                          f"({self.existing_count()}/{self.samples} existing)")
                                else:
                                    print("\n🏆  ALL SIGNS COMPLETE!")

                self.draw_ui(frame, hands_ok, pose_ok)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" ") and not self.recording:
                    self.start_recording()
                elif key == ord("n") and not self.recording:
                    self.sign_idx = (self.sign_idx + 1) % len(self.signs)
                    print(f"  ➡️  {self.current_sign}  ({self.existing_count()}/{self.samples})")
                elif key == ord("p") and not self.recording:
                    self.sign_idx = (self.sign_idx - 1) % len(self.signs)
                    print(f"  ⬅️  {self.current_sign}  ({self.existing_count()}/{self.samples})")
                elif key == ord("r") and not self.recording:
                    self.delete_last()

        self.cap.release()
        cv2.destroyAllWindows()
        self._summary()

    def _summary(self):
        print("\n" + "=" * 65)
        print("📊  COLLECTION SUMMARY")
        print("=" * 65)
        total = 0
        for s in self.signs:
            n      = self.existing_count(s)
            total += n
            status = "✅" if n >= self.samples else f"⚠️  {n}/{self.samples}"
            print(f"   {status:<8}  {s}")
        print(f"\n   Total files: {total}")
        print(f"   Saved to   : {os.path.abspath(OUTPUT_DIR)}/")
        print("=" * 65)
        print("\n▶  Next step: python step2_train.py\n")


# ═══════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Sign Language Data Collector")
    parser.add_argument("--signs",   default="",  help="Comma-separated list of signs")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_SIGN)
    parser.add_argument("--cam", type=int, default=CAM_INDEX,
                        help="Camera index (0 default, try 1/2 if preview is black)")
    args = parser.parse_args()

    signs = [s.strip() for s in args.signs.split(",") if s.strip()] if args.signs else DEFAULT_SIGNS

    collector = Collector(signs=signs, samples=args.samples, cam_index=args.cam)
    collector.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as exc:
        import traceback
        print(f"\n❌  Error: {exc}")
        traceback.print_exc()
