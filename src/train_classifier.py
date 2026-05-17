import mne
import numpy as np
import joblib
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from mne.decoding import Vectorizer
from sklearn.svm import SVC

# List of EEG files
files = [
    "dataset/S001R04.edf",
    "dataset/S001R08.edf",
    "dataset/S001R12.edf",

    "dataset/S002R04.edf",
    "dataset/S002R08.edf",
    "dataset/S002R12.edf",

    "dataset/S003R04.edf",
    "dataset/S003R08.edf",
    "dataset/S003R12.edf"
]


all_epochs = []
all_labels = []

for file_path in files:

    print(f"\nLoading: {file_path}")

    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Bandpass filter
    raw.filter(7., 35., fir_design='firwin')

    # Extract events
    events, event_id = mne.events_from_annotations(raw)

    # Keep only T1 and T2
    epochs = mne.Epochs(
        raw,
        events,
        event_id={'T1': 2, 'T2': 3},
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True
    )

    X = epochs.get_data()
    y = epochs.events[:, -1]

    all_epochs.append(X)
    all_labels.append(y)

# Combine all files
X = np.concatenate(all_epochs, axis=0)
y = np.concatenate(all_labels, axis=0)

print("\nFinal EEG Shape:", X.shape)
print("Labels Shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

import numpy as np

np.savez(
    "test_data.npz",
    X_test=X_test,
    y_test=y_test
)

# CSP + LDA pipeline
csp = CSP(n_components=8, log=True)

clf = Pipeline([
    ('CSP', csp),
    ('SVM', SVC(kernel='rbf'))
])


# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


joblib.dump(clf, "eeg_model.pkl")
print("Model saved successfully!")

import joblib

# Save trained model
joblib.dump(clf, "models/eeg_classifier.pkl")

print("Model saved successfully at models/eeg_classifier.pkl")

import numpy as np
import joblib

# Save test data for Streamlit demo
np.save("models/X_test.npy", X_test)
np.save("models/y_test.npy", y_test)

joblib.dump(clf, "models/eeg_classifier.pkl")

print("Model + test data saved")