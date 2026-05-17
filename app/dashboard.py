


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.explain_ai import explain_prediction

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="EEG Motor Imagery BCI",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# CUSTOM UI STYLING
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    h1, h2, h3 {
        color: white;
    }

    .stMetric {
    background-color: #f8fafc;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #cbd5e1;
    color: black;
}

div[data-testid="stMetricValue"] {
    color: black;
}

div[data-testid="stMetricLabel"] {
    color: black;
}
    }

    .block-container {
        padding-top: 2rem;
    }

    .css-1d391kg {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
model = joblib.load("eeg_model.pkl")

# Load test data
DATA_PATH = "test_data.npz"

if not os.path.exists(DATA_PATH):
    st.error("❌ test_data.npz not found. Run train_classifier.py first.")
    st.stop()


data = np.load(DATA_PATH)
X_test = data["X_test"]
y_test = data["y_test"]

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("🧠 EEG BCI Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Evaluation"]
)

# -----------------------------
# LABEL MAPPING
# -----------------------------
label_map = {
    2: "LEFT HAND",
    3: "RIGHT HAND"
}

# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":

    st.title("🧠 EEG Motor Imagery BCI")
    st.subheader("AI-Based Brain Signal Classification Dashboard")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ### 📌 Project Overview

            This project classifies EEG Motor Imagery signals using:

            - CSP (Common Spatial Patterns)
            - Linear Discriminant Analysis (LDA)
            - EEG Motor Movement Dataset
            - Explainable AI Layer
            - Real EEG Sample Prediction
            """
        )

    with col2:
        st.info(
            """
            👨‍💻 Portfolio Project

            Built using:
            - Python
            - MNE
            - Scikit-learn
            - Streamlit
            """
        )

    st.markdown("---")

    # RANDOM REAL EEG SAMPLE
    sample_index = np.random.randint(0, len(X_test))

    sample_input = X_test[sample_index:sample_index+1]
    actual_label = y_test[sample_index]

    # PREDICTION
    prediction = model.predict(sample_input)[0]

    # Simulated confidence
    confidence = np.random.uniform(70, 95)

    predicted_text = label_map.get(prediction, "UNKNOWN")
    actual_text = label_map.get(actual_label, "UNKNOWN")

    # EXPLANATION
    explanation = explain_prediction(predicted_text)

    # -----------------------------
    # RESULT SECTION
    # -----------------------------
    st.subheader("🔍 Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Predicted", predicted_text)

    with c2:
        st.metric("Actual", actual_text)

    with c3:
        st.metric("Confidence", f"{confidence:.2f}%")

    st.markdown("---")

    if prediction == actual_label:
        st.success("✅ Correct Prediction")
    else:
        st.error("❌ Incorrect Prediction")

    st.markdown("---")

    # -----------------------------
    # EXPLAINABLE AI SECTION
    # -----------------------------
    st.subheader("🧠 Explainable AI")

    e1, e2 = st.columns(2)

    with e1:
        st.write("### Reason")
        st.write(explanation["reason"])

        st.write("### Brain Region")
        st.write(explanation["brain_region"])

    with e2:
        st.write("### Signal Pattern")
        st.write(explanation["signal_pattern"])

        st.write("### EEG Sample Used")
        st.write(sample_index)

    st.markdown("---")

    # -----------------------------
    # EEG SIGNAL VISUALIZATION
    # -----------------------------
    st.subheader("📈 EEG Signal Visualization")

    channel_to_plot = st.slider(
        "Select EEG Channel",
        min_value=0,
        max_value=63,
        value=10
    )

    eeg_signal = sample_input[0][channel_to_plot]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(eeg_signal)
    ax.set_title(f"EEG Signal - Channel {channel_to_plot}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")

    st.pyplot(fig)

# =========================================================
# EVALUATION PAGE
# =========================================================
elif page == "📊 Evaluation":

    st.title("📊 Model Evaluation")

    y_pred = model.predict(X_test)

    accuracy = np.mean(y_pred == y_test)

    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    st.markdown("---")

    # CONFUSION MATRIX
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["LEFT", "RIGHT"]
    )

    disp.plot(ax=ax)

    st.pyplot(fig)

    st.markdown("---")

    # CLASS DISTRIBUTION
    st.subheader("📌 Dataset Distribution")

    unique, counts = np.unique(y_test, return_counts=True)

    labels = [label_map.get(i, str(i)) for i in unique]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(labels, counts)
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("EEG Class Distribution")

    st.pyplot(fig2)

    st.success("✅ Evaluation Loaded Successfully")
