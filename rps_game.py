import cv2
from ultralytics import YOLO
import mediapipe as mp
import pyautogui
import pyttsx3
import threading
import time
import csv
from datetime import datetime

MODEL_PATH = "Rock Paper Scissors.yolov8/runs/detect/train7/weights/best.pt"
HOLD_TIME = 1.5          
FINGER_TIPS = [4, 8, 12, 16, 20]
MIN_SCISSORS = 2
MAX_SCISSORS = 3
TRIGGER_DISPLAY_DURATION = 1.0          
LATENCY_WINDOW  = 30           
FONT = cv2.FONT_HERSHEY_SIMPLEX

POS_LATENCY = (10, 40)
POS_FPS = (10, 125)
POS_HINT = (10, 70)
POS_PROGRESS_TL = (10, 90)
POS_PROGRESS_BR = (310, 108)
POS_TRIGGERED = (10, 145)

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Could not load YOLO model: {e}")
    exit(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine_lock = threading.Lock()

def speak(text: str) -> None:
    """Run TTS in a daemon thread so the video loop never freezes."""
    def _speak():
        with engine_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()


def get_finger_states(hand_landmarks, handedness: str) -> list:
    """
    Return a list of 5 booleans (True = finger extended).
    Handedness is used to correctly evaluate the thumb direction.
    """
    fingers = []

    tip = hand_landmarks.landmark[FINGER_TIPS[0]]
    knuckle = hand_landmarks.landmark[FINGER_TIPS[0] - 1]
    if handedness == "Right":
        fingers.append(tip.x < knuckle.x)
    else:
        fingers.append(tip.x > knuckle.x)

    for tip_id in FINGER_TIPS[1:]:
        tip_y = hand_landmarks.landmark[tip_id].y
        pip_y = hand_landmarks.landmark[tip_id - 2].y
        fingers.append(tip_y < pip_y)

    return fingers


def get_move(hand_landmarks, handedness: str) -> str:
    """Classify gesture as Rock, Paper, Scissors, or Unknown."""
    fingers = get_finger_states(hand_landmarks, handedness)
    total   = sum(fingers)

    if total == 0:
        return "Rock"
    elif MIN_SCISSORS <= total <= MAX_SCISSORS:
        return "Scissors"
    elif total >= 4:
        return "Paper"
    else:
        return "Unknown"


GESTURE_ACTIONS = {
    "Rock":     ("space", "Play / Pause"),
    "Paper":    ("up",    "Volume Up"),
    "Scissors": ("down",  "Volume Down"),
}

last_action = None
hold_start = None
last_trigger_time = 0


def perform_action(move: str, gesture_counts: dict) -> None:
    """
    Trigger a system action only after the gesture has been held
    continuously for HOLD_TIME seconds, then reset the timer.
    """
    global last_action, hold_start, last_trigger_time

    if move == last_action:
        if hold_start is None:
            hold_start = time.time()
        elif time.time() - hold_start > HOLD_TIME:
            if move in GESTURE_ACTIONS:
                key, label = GESTURE_ACTIONS[move]
                pyautogui.press(key)
                speak(label)
                gesture_counts[move] = gesture_counts.get(move, 0) + 1
                last_trigger_time    = time.time()
            hold_start = None          
    else:
        hold_start = None              

    last_action = move


class LatencyTracker:
    """Keeps a rolling average of per-frame latency."""
    def __init__(self, window: int = LATENCY_WINDOW):
        self.window  = window
        self.samples: list = []

    def record(self, ms: float) -> None:
        self.samples.append(ms)
        if len(self.samples) > self.window:
            self.samples.pop(0)

    @property
    def average(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def total_frames(self) -> int:
        return len(self.samples)


def save_session_log(tracker: LatencyTracker, gesture_counts: dict) -> None:
    """Save per-session metrics to a timestamped CSV file."""
    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["avg_latency_ms", round(tracker.average, 2)])
            writer.writerow(["total_frames",   tracker.total_frames])
            for gesture, count in gesture_counts.items():
                writer.writerow([f"actions_{gesture.lower()}", count])
        print(f"[INFO] Session log saved → {filename}")
    except Exception as e:
        print(f"[WARNING] Could not save session log: {e}")


def draw_hud(frame, latency_ms: float, avg_latency: float,
             move: str, fps: float) -> None:
    """Render all on-screen overlays onto the frame."""

    cv2.putText(frame,
                f"Latency: {int(latency_ms)} ms  |  Avg: {int(avg_latency)} ms",
                POS_LATENCY, FONT, 0.65, (0, 255, 255), 2)

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                POS_FPS, FONT, 0.65, (0, 255, 255), 2)

    cv2.putText(frame,
                "Hold gesture 1.5s to trigger  |  Q to quit",
                POS_HINT, FONT, 0.55, (255, 255, 255), 2)

    if hold_start is not None and move in GESTURE_ACTIONS:
        elapsed  = time.time() - hold_start
        progress = min(elapsed / HOLD_TIME, 1.0)
        bar_x2   = POS_PROGRESS_TL[0] + int(300 * progress)
        cv2.rectangle(frame, POS_PROGRESS_TL, (bar_x2, POS_PROGRESS_BR[1]),
                      (0, 200, 100), -1)
        cv2.rectangle(frame, POS_PROGRESS_TL, POS_PROGRESS_BR,
                      (200, 200, 200), 1)

    if time.time() - last_trigger_time < TRIGGER_DISPLAY_DURATION:
        cv2.putText(frame, "Action Triggered!",
                    POS_TRIGGERED, FONT, 0.7, (0, 255, 100), 2)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        exit(1)

    tracker        = LatencyTracker()
    gesture_counts = {"Rock": 0, "Paper": 0, "Scissors": 0}

    print("[INFO] Assistive Gesture System running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame — retrying...")
            continue

        frame       = cv2.flip(frame, 1)
        frame_start = time.time()

        results = model(frame, verbose=False)
        move    = "None"

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf            = float(box.conf[0])

                hand_crop = frame[y1:y2, x1:x2]
                if hand_crop.size == 0:
                    continue

                rgb_crop  = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                mp_result = hands.process(rgb_crop)

                if mp_result.multi_hand_landmarks and mp_result.multi_handedness:
                    for handLms, hand_info in zip(
                        mp_result.multi_hand_landmarks,
                        mp_result.multi_handedness
                    ):
                        handedness = hand_info.classification[0].label
                        move       = get_move(handLms, handedness)

                        cv2.putText(frame,
                                    f"{move} ({handedness})  conf: {conf:.2f}",
                                    (x1, y1 - 10), FONT, 0.85, (0, 255, 0), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if move not in ("Unknown", "None"):
            perform_action(move, gesture_counts)

        latency_ms = (time.time() - frame_start) * 1000
        tracker.record(latency_ms)
        fps = 1000 / latency_ms if latency_ms > 0 else 0

        draw_hud(frame, latency_ms, tracker.average, move, fps)

        cv2.imshow("Assistive Gesture System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_session_log(tracker, gesture_counts)
    print(f"[INFO] Session ended.")
    print(f"       Avg latency : {int(tracker.average)} ms")
    print(f"       Avg FPS     : {int(1000 / tracker.average) if tracker.average > 0 else 0}")
    print(f"       Actions     : {gesture_counts}")


if __name__ == "__main__":
    main()