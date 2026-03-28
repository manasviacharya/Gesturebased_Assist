import cv2
import random
import time
from ultralytics import YOLO
import mediapipe as mp

# Load YOLO model
model = YOLO("Rock Paper Scissors.yolov8/runs/detect/train7/weights/best.pt")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]

# Game variables
player_score = 0
computer_score = 0
round_active = False
countdown_start = 0
COUNTDOWN = 3

def get_move(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in FINGER_TIPS[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total = sum(fingers)

    if total == 0:
        return "Rock"
    elif total == 2:
        return "Scissors"
    elif total >= 4:
        return "Paper"
    else:
        return "Unknown"

def get_winner(p, c):
    if p == c:
        return "Draw"
    elif (p=="Rock" and c=="Scissors") or (p=="Paper" and c=="Rock") or (p=="Scissors" and c=="Paper"):
        return "Player"
    else:
        return "Computer"

cap = cv2.VideoCapture(0)

player_move = "None"
computer_move = "None"
result = ""

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    detected_move = "Unknown"

    # YOLO detection
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🔥 BIG FIX: expand bounding box
            h, w, _ = frame.shape
            pad = 100
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            hand_crop = frame[y1:y2, x1:x2]

            if hand_crop.size == 0:
                continue

            # 🔥 Resize for MediaPipe
            hand_crop = cv2.resize(hand_crop, (300, 300))

            rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                for handLms in res.multi_hand_landmarks:
                    detected_move = get_move(handLms)

            # Draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    # 🔥 Fallback (VERY IMPORTANT)
    if detected_move == "Unknown":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                detected_move = get_move(handLms)
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    key = cv2.waitKey(1) & 0xFF

    # Start round
    if key == ord('s') and not round_active:
        round_active = True
        countdown_start = time.time()
        result = ""

    # Countdown logic
    if round_active:
        elapsed = time.time() - countdown_start
        remaining = COUNTDOWN - int(elapsed)

        if remaining > 0:
            cv2.putText(frame, str(remaining), (300,200),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)
        else:
            player_move = detected_move
            computer_move = random.choice(["Rock","Paper","Scissors"])

            if player_move != "Unknown":
                winner = get_winner(player_move, computer_move)

                if winner == "Player":
                    player_score += 1
                    result = "You Win!"
                elif winner == "Computer":
                    computer_score += 1
                    result = "Computer Wins!"
                else:
                    result = "Draw"
            else:
                result = "No hand detected!"

            round_active = False

    # UI
    cv2.putText(frame, f"You: {player_move}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Computer: {computer_move}", (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.putText(frame, f"Score {player_score}:{computer_score}", (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, result, (10,160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)

    cv2.putText(frame, "Press S to Play | Q to Quit", (10,200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("YOLO + MediaPipe RPS Game", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()