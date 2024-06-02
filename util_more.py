from realtimefft.stream_analyzer import Stream_Analyzer
from itertools import groupby
import time
import matplotlib.pyplot as plt
import numpy as np


def filtered_fft(ear: Stream_Analyzer, lh_freq_bounds: tuple,
                 rh_freq_bounds: tuple):  # freq_bounds are (lower freq, higher freq)
    raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()

    lh_indices = (np.argmin(np.abs(raw_fftx - lh_freq_bounds[0])), np.argmin(np.abs(raw_fftx - lh_freq_bounds[1])))
    rh_indices = (np.argmin(np.abs(raw_fftx - rh_freq_bounds[0])), np.argmin(np.abs(raw_fftx - rh_freq_bounds[1])))

    lh_fft = raw_fft[lh_indices[0]:lh_indices[1]]
    rh_fft = raw_fft[rh_indices[0]:rh_indices[1]]

    return raw_fftx[lh_indices[0]:lh_indices[1]], lh_fft, raw_fftx[rh_indices[0]:rh_indices[1]], rh_fft


def frequency_to_note(freq: float):
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    A4 = 440
    note_number = 12 * np.log2(freq / A4) + 49
    note_number = round(note_number)
    note = (note_number - 1) % len(notes)
    note = notes[note]
    octave = (note_number + 8) // len(notes)
    return note + str(octave)


def find_index_ranges(note_list, fft):
    start = 0
    average_amplitude = []
    notes = []
    for key, group in groupby(note_list):
        notes.append(key)
        group_len = len(list(group))
        average_amplitude.append(
            sum(fft[start:start + group_len]) / group_len)  # sum(fft[start:start+group_len])/group_len
        start += group_len
    return notes, average_amplitude


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
fig, ax = plt.subplots(1, 2)

fps = 60
last_update = time.time()
while True:
    if (time.time() - last_update) > (1. / fps):
        last_update = time.time()

        # Camera does scanning, get ranges. If range not detected, them use previous range
        lhx, lh_fft, rhx, rh_fft = filtered_fft(ear, (143, 180), (520, 670))  # D3-F3, C5-E5

        lhnotes = list(map(frequency_to_note, lhx))
        rhnotes = list(map(frequency_to_note, rhx))

        lhnotes, lh_amplitudes = find_index_ranges(lhnotes, lh_fft)
        rhnotes, rh_amplitudes = find_index_ranges(rhnotes, rh_fft)

        ax[0].bar(lhnotes, lh_amplitudes)
        ax[0].set_title("Left Hand")
        # ax[0].set_ylim(0, 10000000)
        ax[1].set_title("Right Hand")
        ax[1].bar(rhnotes, rh_amplitudes)
        # ax[1].set_ylim(0, 10000000)

        fig.canvas.draw()
        fig.canvas.flush_events()
        ax[0].cla()
        ax[1].cla()
