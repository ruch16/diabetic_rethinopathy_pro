# 🩺 Diabetic Retinopathy Detection App

This is an AI-powered web app that detects the stage of **Diabetic Retinopathy** from retinal (fundus) images.

## 🚀 Features
- Upload retinal images (JPG/PNG)
- CNN model trained on APTOS 2019 dataset
- Predicts 5 DR stages: No DR, Mild, Moderate, Severe, Proliferative DR
- Simple and interactive Streamlit interface

## 🧠 Model
`dr_model.h5` — a TensorFlow model trained on preprocessed fundus images.

## 🛠️ Setup
```bash
pip install -r requirements.txt
streamlit run app.py
