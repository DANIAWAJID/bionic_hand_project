import cv2
import mediapipe as mp
import math

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# max_num_hands=2 taake dono haath detect hon
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Camera Start
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=" * 50)
print("     TWO HANDS PIXEL & DISTANCE MEASUREMENT")
print("=" * 50)
print("Dono haath camera ke samnay laayein.")
print("Band karne ke liye keyboard par [Q] dabayein.")
print("=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Selfie view (flip frame horizontally)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # RGB conversion for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_centers = []

    if results.multi_hand_landmarks:
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            # Skeleton draw karein
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Har landmark ke x, y pixels nikaalein
            x_coords = [int(lm.x * w) for lm in hand_lms.landmark]
            y_coords = [int(lm.y * h) for lm in hand_lms.landmark]

            # Hand Bounding Box (Pixels calculation)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            hand_w_px = x_max - x_min
            hand_h_px = y_max - y_min

            # Haath ka center point (Pixels)
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)
            hand_centers.append((cx, cy))

            # Bounding box draw karein
            cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)
            
            # Haath ka size screen par likhein
            cv2.putText(frame, f"Hand {idx+1}: {hand_w_px}x{hand_h_px} px",
                        (x_min, max(y_min - 20, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Center point par dot banayein
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

        # AGAR DONO HAATH DETECT HO JAYEIN
        if len(hand_centers) == 2:
            (x1, y1) = hand_centers[0]
            (x2, y2) = hand_centers[1]

            # Dono haathon ke darmiyan Euclidean Pixel Distance ka formula
            distance_px = math.hypot(x2 - x1, y2 - y1)

            # Dono ke darmayan line draw karein
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Center of the line par distance display karein
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)

            cv2.circle(frame, (mid_x, mid_y), 6, (255, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"Distance: {int(distance_px)} px", 
                        (mid_x - 80, mid_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # Info banner top par
    cv2.putText(frame, "Press 'Q' to Exit", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Two Hands Distance Measurement - OpenCV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()