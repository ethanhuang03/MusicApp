import cv2
import threading
import handtracker

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


class HandTrackingAsync(handtracker.HandTracking, threading.Thread):
    def __init__(self, capture_device=0, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.3, show_camera=True, key_bounds=None):
        handtracker.HandTracking.__init__(self, capture_device, static_image_mode, max_num_hands,
                                          min_detection_confidence, min_tracking_confidence)
        threading.Thread.__init__(self)
        self.running = False  # For stopping the thread
        self.show_camera = show_camera
        self.key_bounds = key_bounds

    def run(self):
        while self.cap.isOpened() and self.running:
            frame = self.process_frame()
            if self.show_camera:
                if self.key_bounds is not None:
                    for line in self.key_bounds:
                        cv2.line(frame, line.point1, line.point2, (255, 0, 0), 2)
                cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        self.release()
