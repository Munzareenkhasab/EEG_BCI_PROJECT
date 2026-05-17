import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

model = joblib.load("eeg_model.pkl")

def evaluate_model(X_test, y_test):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return cm, report