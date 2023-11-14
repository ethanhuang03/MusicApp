from realtimefft.stream_analyzer import Stream_Analyzer
import time

ear = Stream_Analyzer(
    device=2,  # Pyaudio (portaudio) device index, defaults to first mic input
    rate=None,  # Audio samplerate, None uses the default source settings
    FFT_window_size_ms=80,  # Window size used for the FFT transform
    updates_per_second=2000,  # How often to read the audio stream for new data
    smoothing_length_ms=50,  # Apply some temporal smoothing to reduce noisy features
    n_frequency_bins=600,  # The FFT features are grouped in bins
    visualize=1,  # Visualize the FFT features with PyGame
    verbose=0 # Print running statistics (latency, fps, ...)
)

fps = 60

last_update = time.time()
while True:
    if (time.time() - last_update) > (1./fps):
        last_update = time.time()
        raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()
