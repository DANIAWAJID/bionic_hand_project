import os
import cv2
import csv
import time
from datetime import datetime
import mediapipe as mp

# 1. Dataset Folders Setup
BASE_DIR = "hand_pixel_dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
CSV_FILE = os.path.join(BASE_DIR, "dataset.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# CSV Header Setup (Agar file pehle se nahi hai)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "timestamp", "status", "hand_size_px",
            "wrist_x", "wrist_y", "thumb_x", "thumb_y",
            "index_x", "index_y", "middle_x", "middle_y",
            "ring_x", "ring_y", "pinky_x", "pinky_y"
        ])

# 2. MediaPipe & Camera Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Settings
LIMIT = 200          # Size limit for RED alert
auto_save = False
last_save_time = 0
count = len(os.listdir(IMG_DIR))

print("=" * 50)
print("     HAND PIXEL DATASET GENERATOR")
print("=" * 50)
print(" Controls:")
print("   [S] -> Save 1 Image + All Pixel Data to CSV")
print("   [A] -> Auto-Save ON / OFF")
print("   [Q] -> Exit")
print("=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    raw_frame = frame.copy()
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    color = (0, 255, 0)
    status = "FAR (SAFE)"
    hand_size = 0
    current_data = None

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            # Measure all pixel coordinates
            x_pts = [int(p.x * w) for p in hand.landmark]
            y_pts = [int(p.y * h) for p in hand.landmark]

            hand_size = max(max(x_pts) - min(x_pts), max(y_pts) - min(y_pts))

            # Close vs Far condition
            if hand_size > LIMIT:
                color = (0, 0, 255)  # RED
                status = "TOO CLOSE"
            else:
                color = (0, 255, 0)  # GREEN
                status = "FAR (SAFE)"

            # Draw skeleton and lines
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Display pixel coordinates on fingertips and wrist
            key_points = [0, 4, 8, 12, 16, 20]
            for i in key_points:
                cv2.putText(frame, f"({x_pts[i]},{y_pts[i]})", 
                            (x_pts[i] + 5, y_pts[i] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Draw Bounding Box
            cv2.rectangle(frame, (min(x_pts) - 10, min(y_pts) - 10), 
                                 (max(x_pts) + 10, max(y_pts) + 10), color, 2)
            cv2.putText(frame, f"Size: {hand_size}px", (min(x_pts), min(y_pts) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Data to be saved into CSV
            current_data = [
                status, hand_size,
                x_pts[0], y_pts[0],   # Wrist
                x_pts[4], y_pts[4],   # Thumb
                x_pts[8], y_pts[8],   # Index
                x_pts[12], y_pts[12], # Middle
                x_pts[16], y_pts[16], # Ring
                x_pts[20], y_pts[20]  # Pinky
            ]

    # Save Function
    def save_sample():
        global count
        if current_data is None:
            return
        
        file_name = f"hand_{count:05d}.jpg"
        img_path = os.path.join(IMG_DIR, file_name)
        
        # 1. Clean image save karein
        cv2.imwrite(img_path, raw_frame)

        # 2. CSV file mein pixels aur status save karein
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [file_name, now] + current_data
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[SAVED] {file_name} -> Size: {current_data[1]}px | Status: {current_data[0]}")
        count += 1

    # On-Screen Dashboard
    cv2.putText(frame, f"STATUS: {status}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, f"Saved: {count} | Auto: {'ON' if auto_save else 'OFF'}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "[S]: Save | [A]: Auto | [Q]: Quit", (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Hand Pixel Dataset Generator", frame)
    key = cv2.waitKey(1) & 0xFF

    # Controls
    if key == ord('s'):
        save_sample()
    elif key == ord('a'):
        auto_save = not auto_save
        print(f"Auto-save Mode: {'ENABLED' if auto_save else 'DISABLED'}")
    elif key == ord('q'):
        break

    # Auto-save har 0.5 second baad
    if auto_save and (time.time() - last_save_time) >= 0.5:
        save_sample()
        last_save_time = time.time()

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print(f"Total {count} samples dataset mein save ho chuke hain!")
print(f"Images Folder: {os.path.abspath(IMG_DIR)}")
print(f"CSV File:      {os.path.abspath(CSV_FILE)}")
print("=" * 50)