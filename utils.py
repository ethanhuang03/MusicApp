import numpy as np


class LineSegment:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
        self.gradient = (point2[1] - point1[1]) / (point2[0] - point1[0] + 0.00001)

    def check_point_position(self, point):
        (x1, y1) = self.point2 if self.point2[1] >= self.point1[1] else self.point1
        (x2, y2) = self.point1 if self.point1[1] <= self.point2[1] else self.point2
        (x3, y3) = point

        # Calculate the determinant (cross product) to determine the relative position
        det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

        if det > 0:
            return 1  # Point is to the right
        elif det < 0:
            return -1  # Point is to the left
        else:
            return 0  # Point is on the line


def calculate_hand_coverage(hand_tracker, key_bound_line):
    hand_coverage = []
    for hand in hand_tracker.hand_bounds:
        bottom_left = hand[0].point1
        top_right = hand[1].point2
        bottom_right = hand[1].point1
        top_left = hand[0].point2
        maximum_hand_range = []
        for i in range(len(key_bound_line) - 1):
            left_line = key_bound_line[i]
            right_line = key_bound_line[i + 1]
            if left_line.gradient > 0:  # There's an edge case where hand is at edge of detection range where the following does not work. Need to fix this.
                if left_line.check_point_position(top_left) == 1 and \
                        right_line.check_point_position(top_left) == -1:
                    maximum_hand_range.append(right_line)
                if left_line.check_point_position(bottom_right) == 1 and \
                        right_line.check_point_position(bottom_right) == -1:
                    maximum_hand_range.append(right_line)
                    break
            else:
                if left_line.check_point_position(bottom_left) == 1 and \
                        right_line.check_point_position(bottom_left) == -1:
                    maximum_hand_range.append(right_line)
                if left_line.check_point_position(top_right) == 1 and \
                        right_line.check_point_position(top_right) == -1:
                    maximum_hand_range.append(right_line)
                    break

        hand_coverage.append(maximum_hand_range)
    return hand_coverage


def filtered_fft(ear, freq_bounds):  # freq_bounds are (lower freq, higher freq)
    raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()
    fft_indices = (np.argmin(np.abs(raw_fftx - freq_bounds[0])), np.argmin(np.abs(raw_fftx - freq_bounds[1])))
    fft = raw_fft[fft_indices[0]:fft_indices[1]]

    return raw_fftx[fft_indices[0]:fft_indices[1]], fft


piano_notes_frequencies = {
    "A0": (27.50, 28.29),
    "A#0": (28.29, 30.00),
    "B0": (30.00, 32.00),
    "C1": (32.00, 34.00),
    "C#1": (34.00, 36.50),
    "D1": (36.50, 38.50),
    "D#1": (38.50, 41.00),
    "E1": (41.00, 43.50),
    "F1": (43.50, 46.50),
    "F#1": (46.50, 49.50),
    "G1": (49.50, 52.00),
    "G#1": (52.00, 55.00),
    "A1": (55.00, 58.50),
    "A#1": (58.50, 61.50),
    "B1": (61.50, 65.50),
    "C2": (65.50, 69.00),
    "C#2": (69.00, 73.00),
    "D2": (73.00, 77.00),
    "D#2": (77.00, 82.00),
    "E2": (82.00, 87.00),
    "F2": (87.00, 92.50),
    "F#2": (92.50, 98.50),
    "G2": (98.50, 104.00),
    "G#2": (104.00, 110.00),
    "A2": (110.00, 116.50),
    "A#2": (116.50, 123.50),
    "B2": (123.50, 130.50),
    "C3": (130.50, 138.00),
    "C#3": (138.00, 146.50),
    "D3": (146.50, 155.50),
    "D#3": (155.50, 164.50),
    "E3": (164.50, 174.50),
    "F3": (174.50, 185.00),
    "F#3": (185.00, 196.00),
    "G3": (196.00, 207.50),
    "G#3": (207.50, 220.00),
    "A3": (220.00, 233.00),
    "A#3": (233.00, 246.50),
    "B3": (246.50, 261.00),
    "C4": (261.00, 277.00),
    "C#4": (277.00, 293.50),
    "D4": (293.50, 311.00),
    "D#4": (311.00, 329.50),
    "E4": (329.50, 349.00),
    "F4": (349.00, 369.50),
    "F#4": (369.50, 392.00),
    "G4": (392.00, 415.00),
    "G#4": (415.00, 440.00),
    "A4": (440.00, 466.00),
    "A#4": (466.00, 493.50),
    "B4": (493.50, 523.00),
    "C5": (523.00, 554.50),
    "C#5": (554.50, 587.00),
    "D5": (587.00, 622.00),
    "D#5": (622.00, 659.00),
    "E5": (659.00, 698.00),
    "F5": (698.00, 739.50),
    "F#5": (739.50, 784.00),
    "G5": (784.00, 830.50),
    "G#5": (830.50, 880.00),
    "A5": (880.00, 932.00),
    "A#5": (932.00, 987.50),
    "B5": (987.50, 1046.00),
    "C6": (1046.00, 1109.00),
    "C#6": (1109.00, 1175.00),
    "D6": (1175.00, 1244.00),
    "D#6": (1244.00, 1319.00),
    "E6": (1319.00, 1397.00),
    "F6": (1397.00, 1480.00),
    "F#6": (1480.00, 1568.00),
    "G6": (1568.00, 1661.00),
    "G#6": (1661.00, 1760.00),
    "A6": (1760.00, 1865.00),
    "A#6": (1865.00, 1976.00),
    "B6": (1976.00, 2093.00),
    "C7": (2093.00, 2217.50),
    "C#7": (2217.50, 2349.00),
    "D7": (2349.00, 2489.00),
    "D#7": (2489.00, 2637.00),
    "E7": (2637.00, 2794.00),
    "F7": (2794.00, 2960.00),
    "F#7": (2960.00, 3136.00),
    "G7": (3136.00, 3322.50),
    "G#7": (3322.50, 3520.00),
    "A7": (3520.00, 3729.50),
    "A#7": (3729.50, 3951.00),
    "B7": (3951.00, 4186.00),
    "C8": (4186.00, 4435.00)
}

white_piano_notes_frequencies = {
    "A0": (27.50, 28.29),
    "B0": (30.00, 32.00),
    "C1": (32.00, 34.00),
    "D1": (36.50, 38.50),
    "E1": (41.00, 43.50),
    "F1": (43.50, 46.50),
    "G1": (49.50, 52.00),
    "A1": (55.00, 58.50),
    "B1": (61.50, 65.50),
    "C2": (65.50, 69.00),
    "D2": (73.00, 77.00),
    "E2": (82.00, 87.00),
    "F2": (87.00, 92.50),
    "G2": (98.50, 104.00),
    "A2": (110.00, 116.50),
    "B2": (123.50, 130.50),
    "C3": (130.50, 138.00),
    "D3": (146.50, 155.50),
    "E3": (164.50, 174.50),
    "F3": (174.50, 185.00),
    "G3": (196.00, 207.50),
    "A3": (220.00, 233.00),
    "B3": (246.50, 261.00),
    "C4": (261.00, 277.00),
    "D4": (293.50, 311.00),
    "E4": (329.50, 349.00),
    "F4": (349.00, 369.50),
    "G4": (392.00, 415.00),
    "A4": (440.00, 466.00),
    "B4": (493.50, 523.00),
    "C5": (523.00, 554.50),
    "D5": (587.00, 622.00),
    "E5": (659.00, 698.00),
    "F5": (698.00, 739.50),
    "G5": (784.00, 830.50),
    "A5": (880.00, 932.00),
    "B5": (987.50, 1046.00),
    "C6": (1046.00, 1109.00),
    "D6": (1175.00, 1244.00),
    "E6": (1319.00, 1397.00),
    "F6": (1397.00, 1480.00),
    "G6": (1568.00, 1661.00),
    "A6": (1760.00, 1865.00),
    "B6": (1976.00, 2093.00),
    "C7": (2093.00, 2217.50),
    "D7": (2349.00, 2489.00),
    "E7": (2637.00, 2794.00),
    "F7": (2794.00, 2960.00),
    "G7": (3136.00, 3322.50),
    "A7": (3520.00, 3729.50),
    "B7": (3951.00, 4186.00),
    "C8": (4186.00, 4435.00)
}
