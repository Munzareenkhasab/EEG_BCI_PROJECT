import joblib
import numpy as np

model = joblib.load("models/eeg_classifier.pkl")

def predict_eeg(sample):
    sample = np.array(sample)
    sample = sample.reshape(1, -1)
    return model.predict(sample)[0]