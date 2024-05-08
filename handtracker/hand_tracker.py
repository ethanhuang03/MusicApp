import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.3)

cap = cv2.VideoCapture(1)  # 0 for webcam
while cap.isOpened():
    success, image = cap.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_max, y_max, x_min, y_min = 0, 0, image.shape[1], image.shape[0]
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                x_max = max(x, x_max)
                y_max = max(y, y_max)
                x_min = min(x, x_min)
                y_min = min(y, y_min)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



