import cv2
import keyboard
import json

from pianokeydetector.pianokeydetector import PianoKeyDetector
from handtracker.handtracker import HandTracking
import utils


# Video Recording:
video_stream = """C:\\Users\\Ethan\\Videos\\rfiu\\rfiu.mp4"""
cap = cv2.VideoCapture(video_stream)
frame_counter = 0
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Calibrate Range:
detector = PianoKeyDetector(cap.read()[1])
detector.get_corner_points()
nkeys = int(input("How many white keys are selected: "))
lowest_note = input("What is the lowest note (e.g. C4): ")  # Only sharp keys
detector.process_image(nkeys)
key_bounds = detector.key_bounds

white_piano_notes = list(utils.white_piano_notes_frequencies.keys())
lowest_note_index = white_piano_notes.index(lowest_note)
notes_list = white_piano_notes[lowest_note_index-1:lowest_note_index + len(key_bounds)-1]  # A0 edge case
key_bounds_dict = dict(zip(key_bounds, notes_list))

# Hand Tracking:
max_num_hands = 2
hand_tracking = HandTracking(capture_device=video_stream, max_num_hands=max_num_hands)
show_camera = True

previous_hand_coverage = [None] * max_num_hands

# Prepare JSON data:
output_data = []

while True:
    frame = hand_tracking.process_frame()
    if frame is None:
        continue
    if show_camera:
        for line in key_bounds:
            cv2.line(frame, line.point1, line.point2, (255, 0, 0), 2)
        cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(5) & 0xFF == 27 or keyboard.is_pressed('esc'):  # esc to terminate / failsafe
        break

    hand_coverage = utils.calculate_hand_coverage(hand_tracking, key_bounds)
    hand_moved = False
    hand_ranges = []
    for i, hand in enumerate(hand_coverage):
        if hand != previous_hand_coverage[i]:
            hand_moved = True
        note_range = []
        for line in hand:
            note_range.append(key_bounds_dict[line])
        hand_ranges.append(note_range)
        previous_hand_coverage[i] = hand

    if hand_moved:
        timestamp = frame_counter / frame_rate * 1000
        output_data.append({
            "timestamp": timestamp,
            "hand_ranges": hand_ranges
        })
        print(f"Time {timestamp:.2f} ms - " + " | ".join(map(str, hand_ranges)))

    frame_counter += 1


hand_tracking.release()
with open('hand_tracker_data.json', 'w') as json_file:
    json.dump(output_data, json_file, indent=4)

