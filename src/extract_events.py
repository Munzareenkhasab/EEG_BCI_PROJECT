import mne

# Load EEG data
file_path = "dataset/S001R04.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

# Apply filter
raw.filter(8, 30)

# Extract events
events, event_dict = mne.events_from_annotations(raw)

print("Event Dictionary:")
print(event_dict)

print("\nEvents Shape:")
print(events.shape)

print("\nFirst 10 Events:")
print(events[:10])