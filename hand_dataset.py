import os
import time
import cv2

# 1. Dataset folders setup
DATASET_DIR = "hand_dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# 2. Camera Start
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Camera connect nahi hua!")
    exit()

# Bounding Box Settings (Screen ke center mein Green Box hoga)
box_w, box_h = 240, 240
count = len(os.listdir(IMG_DIR))
auto_save = False
last_save_time = 0

print("=" * 45)
print("     HAND YOLO DATASET CREATOR")
print("=" * 45)
print(" [S]     -> Ek Photo + Hand Label save karein")
print(" [A]     -> Auto-Capture ON/OFF (Lagaataar photo lega)")
print(" [W/S/A/D keys] -> Box ko chota/bada karein (+ / -)")
print(" [Q]     -> Quit / Band karein")
print("=" * 45)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror effect (selfie camera ki tarah)
    frame = cv2.flip(frame, 1)
    raw_frame = frame.copy()
    h_frame, w_frame, _ = frame.shape

    # Screen ke center mein box ke coordinates
    x1 = (w_frame - box_w) // 2
    y1 = (h_frame - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    # YOLO Format: class_id x_center y_center width height (Normalized 0 se 1)
    x_center = (x1 + box_w / 2.0) / w_frame
    y_center = (y1 + box_h / 2.0) / h_frame
    norm_w = box_w / float(w_frame)
    norm_h = box_h / float(h_frame)

    # Class 0 = Hand
    yolo_annotation = f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

    # Visual Display (Green Box jahan apna haath rakhna hai)
    box_color = (0, 0, 255) if auto_save else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(frame, "APNA HAATH YAHAN RAKHEIN", (x1 - 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Status Overlay
    status = f"Saved: {count} | Auto-Mode: {'ON' if auto_save else 'OFF'}"
    cv2.putText(frame, status, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Hand Dataset Maker - OpenCV", frame)
    key = cv2.waitKey(1) & 0xFF

    # Save function
    def save_hand_sample():
        global count
        file_name = f"hand_{count:05d}"
        img_path = os.path.join(IMG_DIR, f"{file_name}.jpg")
        txt_path = os.path.join(LBL_DIR, f"{file_name}.txt")

        # 1. Clean Photo save karein (bina green box ke)
        cv2.imwrite(img_path, raw_frame)

        # 2. YOLO Text file save karein (0 x_center y_center w h)
        with open(txt_path, "w") as f:
            f.write(yolo_annotation)

        print(f"[SAVED] {file_name}.jpg aur {file_name}.txt")
        count += 1

    # Key Controls
    if key == ord('s'):
        save_hand_sample()

    elif key == ord('a'):
        auto_save = not auto_save
        print(f"Auto Save Mode: {'ON' if auto_save else 'OFF'}")

    elif key == ord('+') or key == ord('='):
        box_w = min(box_w + 20, 400)
        box_h = min(box_h + 20, 400)

    elif key == ord('-'):
        box_w = max(box_w - 20, 100)
        box_h = max(box_h - 20, 100)

    elif key == ord('q'):
        break

    # Auto Save har 0.3 seconds baad (jab tak ON ho)
    if auto_save and (time.time() - last_save_time) >= 0.3:
        save_hand_sample()
        last_save_time = time.time()

cap.release()
cv2.destroyAllWindows()

# YOLO data.yaml file automatically generate karein
yaml_path = os.path.join(DATASET_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {os.path.abspath(DATASET_DIR)}\n")
    f.write("train: images\n")
    f.write("val: images\n")
    f.write("names:\n  0: hand\n")

print("\n" + "=" * 45)
print(f"Total {count} Hand images aur YOLO labels save ho gaye!")
print(f"Folder: {os.path.abspath(DATASET_DIR)}")
print("data.yaml bhi ban gaya hai (Ready for YOLO training).")
print("=" * 45)