import matplotlib
matplotlib.use('TkAgg')

import mne
import matplotlib.pyplot as plt

# EEG file path
file_path = "dataset/S001R04.edf"

# Load EEG data
raw = mne.io.read_raw_edf(file_path, preload=True)

# Print dataset info
print(raw)

# Print channel names
print("\nChannels:")
print(raw.ch_names)

# Plot EEG signals
raw.plot(duration=5, n_channels=10)

# Keep plot window open
plt.show()