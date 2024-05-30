import cv2
import mediapipe as mp
import threading

'''
# Example Code:
from handtracker.handtracker_threading import HandTracking
hand_tracking = HandTracking(capture_device=0,
                             show_camera=True,
                             max_num_hands=2,
                             key_bounds=None)
hand_tracking.running = True  # Start the thread
hand_tracking.start()  # Start the thread

while True:
    print(hand_tracking.hand_bounds)
    # hand_tracking.running = False  # Stop the thread
'''


class HandTracking(threading.Thread):
    def __init__(self, capture_device=0, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.3, show_camera=True, key_bounds=None):
        super(HandTracking, self).__init__()
        self.running = False  # For stopping the thread
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.cap = cv2.VideoCapture(capture_device)  # 0 for webcam

        self.show_camera = show_camera
        self.key_bounds = key_bounds
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
                self.hand_bounds.append([(x_min, y_min), (x_max, y_max)])
        else:
            self.hand_bounds = []
        return image

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        while self.cap.isOpened() and self.running:
            frame = self.process_frame()
            if self.show_camera:
                if self.key_bounds is not None:
                    for line in self.key_bounds:
                        cv2.line(frame, line[0], line[1], (255, 0, 0), 2)
                cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        self.release()
