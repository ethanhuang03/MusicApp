import cv2
import mediapipe as mp
from utils import LineSegment


class HandTracking:
    def __init__(self, capture_device=0, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.3):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.cap = cv2.VideoCapture(capture_device)  # 0 for webcam
        self.hand_bounds = []

    def process_frame(self):
        success, image = self.cap.read()
        if not success:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            self.hand_bounds = []
            for hand_landmarks in results.multi_hand_landmarks:
                x_max, y_max, x_min, y_min = 0, 0, image.shape[1], image.shape[0]
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                self.hand_bounds.append([LineSegment((x_min, y_min), (x_min, y_max)),
                                         LineSegment((x_max, y_min), (x_max, y_max))])
        else:
            self.hand_bounds = []
        return image

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
