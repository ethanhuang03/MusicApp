from realtimefft.stream_analyzer import Stream_Analyzer
import time
import matplotlib.pyplot as plt
import numpy as np


def generate_piano_freqs(freq_bond=15):
    piano_freqs = np.array(
        [
            26.0, 27.5, 29.14, 30.87, 32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49, 51.91, 55, 58.27, 61.74, 65.41,
            69.3, 73.42, 77.78, 82.41, 87.31, 92.5, 98, 103.83, 110, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56,
            164.81, 174.61, 185, 196, 207.65, 220, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
            369.99, 392, 415.3, 440, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99,
            830.61, 880, 932.33, 987.77, 1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22,
            1760, 1864.66, 1975.53, 2093, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520,
            3729.31, 3951.07, 4186.01, 4434.92
        ])  # There are two extra frequencies (at beginning and end)
    bounded = []
    for i in range(1, len(piano_freqs)-1):
        bounded.append((max(0, piano_freqs[i]-freq_bond, piano_freqs[i]-(piano_freqs[i]-piano_freqs[i-1])/2),
                        min(piano_freqs[i]+freq_bond, piano_freqs[i]+(piano_freqs[i+1]-piano_freqs[i])/2)))
    return bounded


def filter_freq(fttx, fft, bounded_freq):
    result = np.array([], dtype=fttx.dtype)
    for i in bounded_freq:
        index = np.where((fttx >= i[0]) & (fttx <= i[1]))
        result = np.concatenate((result, index[0]))
    indices = np.unique(result).astype(int)

    return fttx[indices], fft[indices]


def find_area(fttx, fft, bounded_freq):
    area = []
    for i in bounded_freq:
        index = np.where((fttx >= i[0]) & (fttx <= i[1]))
        amplitudes = fft[index[0]]
        if len(amplitudes) <= 0:
            height = 0
        else:
            height = max(fft[index[0]])
        area.append((i[1]-i[0])*height/2)
    return np.array(area)


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
fig, ax = plt.subplots()

piano_freqs = generate_piano_freqs(freq_bond=15)
axis = [i for i in range(0, 88)]

fps = 60
last_update = time.time()
while True:
    if (time.time() - last_update) > (1. / fps):
        last_update = time.time()
        raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()

        highest_freq = np.argmin(np.abs(raw_fftx-piano_freqs[-1][1]))

        piano_fftx = raw_fftx[:highest_freq]
        piano_fft = raw_fft[:highest_freq]

        areas = find_area(piano_fftx, piano_fft, piano_freqs)
        note_percentage_composition = (areas/np.sum(areas)) * 100

        # Own plotting to understand everything
        # ax.plot(*filter_freq(piano_fftx, piano_fft, piano_freqs))
        ax.set_ylim(0, 100)
        ax.bar(axis, note_percentage_composition)
        fig.canvas.draw()
        fig.canvas.flush_events()
        ax.cla()
