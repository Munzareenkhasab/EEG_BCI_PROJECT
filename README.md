# 🧠 EEG Motor Imagery BCI

AI-Based EEG Brain Signal Classification using CSP + LDA with Explainable AI and Streamlit Dashboard.

---

## 🚀 Live Demo

🔗 Streamlit Dashboard:
(streamlit run app/dashboard.py)

---

## 📌 Project Overview

This project classifies EEG Motor Imagery signals into:

- LEFT HAND movement imagery
- RIGHT HAND movement imagery

using Brain-Computer Interface (BCI) techniques.

The system processes EEG signals, extracts spatial features using CSP (Common Spatial Patterns), and performs classification using Linear Discriminant Analysis (LDA).

---

## 🧠 Features

✅ EEG Signal Processing  
✅ CSP Feature Extraction  
✅ LDA Classification  
✅ Real EEG Sample Prediction  
✅ Explainable AI Layer  
✅ Confusion Matrix Visualization  
✅ Interactive Streamlit Dashboard  
✅ EEG Signal Visualization  
✅ GitHub + Cloud Deployment

---

## 🛠️ Technologies Used

- Python
- Streamlit
- MNE
- Scikit-learn
- NumPy
- Matplotlib
- Joblib

---

## 📂 Project Structure

```plaintext
EEG_BCI_Project/
│
├── app/
│   └── dashboard.py
│
├── src/
│   └── train_classifier.py
│
├── utils/
│   └── explain_ai.py
│
├── dataset/
├── models/
│
├── eeg_model.pkl
├── test_data.npz
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/EEG_BCI_PROJECT.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app/dashboard.py
```

---

## 📊 Model Performance

- Accuracy achieved: ~66%
- Binary Motor Imagery Classification
- CSP + LDA Pipeline

---

## 🔬 Future Improvements

- Deep Learning Models (CNN/LSTM)
- SHAP Explainability
- Real EEG Device Integration
- Multi-class Motor Imagery
- Real-time EEG Streaming

---

## 👨‍💻 Author

Munzareen Khasab  
Artificial Intelligence & Data Science Engineering
