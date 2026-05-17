import mne

# Load EEG file
file_path = "dataset/S001R04.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

# Filter EEG
raw.filter(8, 30)

# Extract events
events, event_dict = mne.events_from_annotations(raw)

# Select only T1 and T2
event_id = {
    'T1': 2,
    'T2': 3
}

# Create epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=0,
    tmax=4,
    baseline=None,
    preload=True
)

print(epochs)

print("\nEpoch Data Shape:")
print(epochs.get_data().shape)