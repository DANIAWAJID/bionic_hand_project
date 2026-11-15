import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Yahan apni naapi hui hatheli ki chauRai (cm) likhein
REAL_WIDTH_CM = {'Left': 8.0, 'Right': 8.0}
DISTANCES = [20, 30, 40, 50, 60]

saved = {'Left': {}, 'Right': {}}
idx = 0
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
      continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    target = DISTANCES[idx]
    current = {}

    if res.multi_hand_landmarks:
      for hl, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = hd.classification[0].label
        mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        x1 = hl.landmark[5].x * w
        y1 = hl.landmark[5].y * h
        x2 = hl.landmark[17].x * w
        y2 = hl.landmark[17].y * h
        pixel_width = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        focal = (pixel_width * target) / REAL_WIDTH_CM[label]
        current[label] = (pixel_width, focal)

        tx = max(10, int(hl.landmark[0].x * w) - 80)
        ty = min(h - 20, int(hl.landmark[0].y * h) + 35)
        cv2.putText(frame, f'{label}: {int(pixel_width)} px  F={focal:.0f}',
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f'Haath {target} cm par rakhein | S = save | N = next | Q = quit',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    y = 65
    for label in ['Left', 'Right']:
      for d, (p, f) in saved[label].items():
        cv2.putText(frame, f'{label} {d}cm: {int(p)}px  F={f:.0f}', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 22

    cv2.imshow('Hand Focal Length', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      break
    if key == ord('s'):
      for label, (p, f) in current.items():
        saved[label][target] = (p, f)
        print(f'{label} hand | {target} cm | {p:.1f} px | focal {f:.1f}')
    if key == ord('n'):
      idx += 1
      if idx >= len(DISTANCES):
        break

cap.release()
cv2.destroyAllWindows()

print('\n--- RESULT ---')
for label in ['Left', 'Right']:
  if saved[label]:
    print(f'\n{label} hand:')
    for d, (p, f) in saved[label].items():
      print(f'  {d} cm | {p:.1f} px | focal {f:.1f}')
    avg = sum(f for _, f in saved[label].values()) / len(saved[label])
    print(f'  AVERAGE FOCAL LENGTH ({label}) = {avg:.1f}')