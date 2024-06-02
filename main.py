import cv2
import keyboard
import matplotlib.pyplot as plt

from pianokeydetector.pianokeydetector import PianoKeyDetector
from handtracker.handtracker import HandTracking
from realtimefft.stream_analyzer import Stream_Analyzer
import utils


# Calibrate Range:
video_stream = 1  #"http://10.0.0.21:8080/video"
detector = PianoKeyDetector(cv2.VideoCapture(video_stream).read()[1])
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
show_camera = False

# FFT
ear = Stream_Analyzer(
    device=2,  # Pyaudio (portaudio) device index, defaults to first mic input
    rate=None,  # Audio samplerate, None uses the default source settings
    FFT_window_size_ms=100,  # Window size used for the FFT transform
    updates_per_second=2000,  # How often to read the audio stream for new data
    smoothing_length_ms=50,  # Apply some temporal smoothing to reduce noisy features
    n_frequency_bins=600,  # The FFT features are grouped in bins
    visualize=0,  # Visualize the FFT features with PyGame
    verbose=0  # Print running statistics (latency, fps, ...)
)
plt.ion()
fig, ax = plt.subplots(1, max_num_hands)


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
    for i, hand in enumerate(hand_coverage):
        note_range = []
        for line in hand:
            note_range.append(key_bounds_dict[line])
        # print(f"Hand {i + 1}: {note_range}")

        # Filtered FFT Analysis
        fftx, fft = utils.filtered_fft(ear, (utils.white_piano_notes_frequencies[note_range[0]][0],
                                             utils.white_piano_notes_frequencies[note_range[-1]][-1]))
        # Plot FFT
        ax[i].bar(fftx, fft)
        ax[i].set_title(f"Hand {i + 1}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        ax[i].cla()

hand_tracking.release()
