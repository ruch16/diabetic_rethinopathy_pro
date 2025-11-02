import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# ✅ Load trained model
# -----------------------------
@st.cache_resource
def load_dr_model():
    model = load_model("dr_model.h5", compile=False)
    return model

model = load_dr_model()

# -----------------------------
# ✅ Image Preprocessing
# -----------------------------
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# ✅ Prediction Function
# -----------------------------
def predict(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    preds = preds[0]
    class_names = [
        "No Diabetic Retinopathy (Healthy)",
        "Mild DR",
        "Moderate DR",
        "Severe DR",
        "Proliferative DR"
    ]

    # Show confidence scores
    st.subheader("Confidence Scores")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {preds[i]*100:.2f}%")

    predicted_class = np.argmax(preds)
    st.success(f"✅ Predicted Stage: {class_names[predicted_class]}")

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------
st.set_page_config(page_title="Diabetic Retinopathy Detector", page_icon="🩺", layout="centered")

st.title("👁️ Diabetic Retinopathy Detection")
st.markdown("Upload a **retinal fundus image** to detect the stage of Diabetic Retinopathy using a deep learning model.")

uploaded_image = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image... please wait"):
            predict(uploaded_image)
