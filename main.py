from realtimefft.stream_analyzer import Stream_Analyzer
import time
import matplotlib.pyplot as plt
import numpy as np


def generate_piano_freqs(freq_bond=50):
    piano_freqs = np.array(
        [16, 17, 18, 19, 21, 23, 24, 26, 28, 29, 31, 33, 35, 37, 39, 41, 44, 46, 49, 52, 55, 58, 62, 65,
         69, 73, 78, 82, 87, 92, 98, 104, 110, 117, 123, 131, 139, 147, 156, 165, 175, 185, 196, 208,
         220, 233, 247, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494, 523, 554, 587, 622, 659, 698,
         740, 784, 831, 880, 932, 988, 1047, 1109, 1175, 1245, 1319, 1397, 1480, 1568, 1661, 1760, 1865,
         1976, 2093, 2217, 2349, 2489, 2637, 2793, 2960, 3136, 3322, 3520, 3729, 3951, 4186])
    bounded = []
    for i in piano_freqs:
        bounded.append((max(0, i-freq_bond), i+freq_bond))

    return bounded


def filter_freq(fttx, fft, bounded_freq):
    result = np.array([], dtype=fttx.dtype)
    for i in bounded_freq:
        index = np.where((fttx >= i[0]) & (fttx <= i[1]))
        result = np.concatenate((result, index[0]))
    indices = np.unique(result).astype(int)

    return fttx[indices], fft[indices]


ear = Stream_Analyzer(
    device=2,  # Pyaudio (portaudio) device index, defaults to first mic input
    rate=None,  # Audio samplerate, None uses the default source settings
    FFT_window_size_ms=80,  # Window size used for the FFT transform
    updates_per_second=2000,  # How often to read the audio stream for new data
    smoothing_length_ms=50,  # Apply some temporal smoothing to reduce noisy features
    n_frequency_bins=600,  # The FFT features are grouped in bins
    visualize=0,  # Visualize the FFT features with PyGame
    verbose=0  # Print running statistics (latency, fps, ...)
)

plt.ion()
fig, ax = plt.subplots()

piano_freqs = generate_piano_freqs(freq_bond=5)

fps = 60
last_update = time.time()
while True:
    if (time.time() - last_update) > (1. / fps):
        last_update = time.time()
        raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()

        piano_fftx = raw_fftx[:400]
        piano_fft = raw_fft[:400]

        # Own plotting to understand everything
        ax.set_ylim(0, 10000000)
        ax.plot(*filter_freq(piano_fftx, piano_fft, piano_freqs))
        fig.canvas.draw()
        fig.canvas.flush_events()
        ax.cla()
