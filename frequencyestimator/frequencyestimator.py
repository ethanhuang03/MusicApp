import pathlib
import glob
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tqdm import tqdm
import pretty_midi
import utils


data_dir = pathlib.Path('E:\Documents\maestro-v3.0.0')
midi_filenames = glob.glob(str(data_dir/'**/*.mid*'))
wav_filenames = [os.path.splitext(midi_file)[0] + '.wav' for midi_file in midi_filenames]
filenames = dict(zip(wav_filenames, midi_filenames))


n_seconds = 200
df = pd.DataFrame()

plt.ion()
fig, ax = plt.subplots(1, 1)

for wav_filename, midi_filename in tqdm(filenames.items()):
    sample_rate, data = wav.read(wav_filename)
    total_duration = len(data) / sample_rate
    num_chunks = int(total_duration / n_seconds)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    for i in range(num_chunks):
        start = int(i * n_seconds * sample_rate)
        end = int((i + 1) * n_seconds * sample_rate)
        chunk = data[start:end]
        yf = np.fft.rfft(chunk)
        xf = np.fft.rfftfreq(chunk.size, 1.0 / sample_rate)
        yf = np.abs(yf[(xf >= 27.50) & (xf <= 4435.00)])
        xf = xf[(xf >= 27.50) & (xf <= 4435.00)]

        fractions = []
        for note, freq_range in utils.piano_notes_frequencies.items():
            yf_note = yf[(xf >= freq_range[0]) & (xf <= freq_range[1])]
            area_note = np.trapz(yf_note, dx=1.0 / sample_rate)
            fractions.append(area_note)  # np.mean(yf_note)
        total_area = np.trapz(yf, dx=1.0/sample_rate)  # np.sum(fractions)
        fractions = np.array([area / total_area for area in fractions])
        # print(np.sum(fractions))

        # Process MIDI
        graph = ax.bar(utils.piano_notes_frequencies.keys(), fractions)
        ax.set_title("RAW Notes")

        midi = pretty_midi.PrettyMIDI(midi_filename)
        notes = np.zeros(88)
        for note in midi.instruments[0].notes:
            start_time = max(note.start, start / sample_rate)
            end_time = min(note.end, end / sample_rate)
            duration = end_time - start_time
            if duration > 0:
                note_number = note.pitch - 21
                notes[note_number] = 1
                graph[note_number].set_color('r')

        fig.canvas.draw()
        fig.canvas.flush_events()
        ax.cla()
