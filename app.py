import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Load the model ---
@st.cache_resource
def load_dr_model():
    model = load_model("dr_model.h5")
    return model

model = load_dr_model()

# --- App title and intro ---
st.title("🩸 Diabetic Retinopathy Detection App")
st.markdown("""
Upload a **retina image** to predict the stage of diabetic retinopathy.  
Model predicts based on deep learning (CNN trained model).
""")

# --- Upload section ---
uploaded_file = st.file_uploader("📤 Upload a retina image (JPG or PNG)", type=["jpg", "jpeg", "png"])

# --- Prediction labels ---
classes = [
    "No Diabet


