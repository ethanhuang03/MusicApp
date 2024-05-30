import cv2
import time
from pianokeydetector.pianokeydetector import PianoKeyDetector
from handtracker.handtracker import HandTracking


def check_point_position(line_segment, point):
    (x1, y1) = line_segment[1] if line_segment[1][1] >= line_segment[0][1] else line_segment[0]
    (x2, y2) = line_segment[0] if line_segment[0][1] <= line_segment[1][1] else line_segment[1]
    (x3, y3) = point

    # Calculate the determinant (cross product) to determine the relative position
    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    if det > 0:
        return 1  # Point is to the right
    elif det < 0:
        return -1  # Point is to the left
    else:
        return 0  # Point is on the line


# Calibrate Range:
video_stream = "http://10.0.0.21:8080/video"
detector = PianoKeyDetector(cv2.VideoCapture(video_stream).read()[1])
detector.get_corner_points()
nkeys = int(input("How many white keys are selected: "))
lowest_note = input("What is the lowest note (e.g. C4): ")
detector.process_image(nkeys)
key_bounds = detector.key_bounds
key_bounds_gradient = (key_bounds[0][0][1] - key_bounds[1][0][1])/(key_bounds[0][0][0] - key_bounds[1][0][0] + 0.00001)

# Track Hands: 
hand_tracking = HandTracking(capture_device=video_stream,
                             show_camera=True,
                             max_num_hands=2,
                             key_bounds=key_bounds)
hand_tracking.running = True  # Start the thread
hand_tracking.start()  # Start the thread
while True:
    for hand in hand_tracking.hand_bounds:
        bottom_left, top_right = hand[0], hand[1]
        bottom_right = (top_right[0], bottom_left[1])
        top_left = (bottom_left[0], top_right[1])
        maximum_hand_range = []
        for i in range(len(key_bounds)-1):
            left_line = key_bounds[i]
            right_line = key_bounds[i+1]
            if key_bounds_gradient > 0:
                if check_point_position(left_line, top_left) == 1 and \
                        check_point_position(right_line, top_left) == -1:
                    maximum_hand_range.append(right_line)
                if check_point_position(left_line, bottom_right) == 1 and \
                        check_point_position(right_line, bottom_right) == -1:
                    maximum_hand_range.append(right_line)
                    break
            else:
                if check_point_position(left_line, bottom_left) == 1 and \
                        check_point_position(right_line, bottom_left) == -1:
                    maximum_hand_range.append(right_line)
                    pass
                if check_point_position(left_line, top_right) == 1 and \
                        check_point_position(right_line, top_right) == -1:
                    maximum_hand_range.append(right_line)
                    break

        print(maximum_hand_range)
        maximum_hand_range = []
        time.sleep(0.5)

    # print(hand_tracking.hand_bounds)
    # hand_tracking.running = False

    # hand_tracking.hand_bounds: [[(x_min, y_min), (x_max, y_max)], [(x_min, y_min), (x_max, y_max)]]
    # key_bounds = [[(x, y),(x, y)], [(x, y),(x, y)], [(x, y),(x, y)], [(x, y),(x, y)]]

