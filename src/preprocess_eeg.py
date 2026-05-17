import matplotlib
matplotlib.use('TkAgg')

import mne
import matplotlib.pyplot as plt

# Load EEG file
file_path = "dataset/S001R04.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

print("Original Data Loaded")

# -----------------------------
# BANDPASS FILTER
# -----------------------------
# Keep frequencies between 8–30 Hz
# Important for motor imagery EEG

raw.filter(l_freq=8, h_freq=30)

print("Filtering Completed")

# -----------------------------
# REMOVE BAD CHANNELS (optional)
# -----------------------------
raw.pick_types(eeg=True)

# -----------------------------
# PLOT FILTERED SIGNAL
# -----------------------------
raw.plot(duration=5, n_channels=10)

plt.show()