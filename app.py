import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("dr_model.h5")
    return model

model = load_model()

# Class labels for DR stages
classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# -----------------------------
# Image Prediction Function
# -----------------------------
def predict(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return preds[0]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="🩺")

st.title("🩺 Diabetic Retinopathy Detection")
st.write("Upload a **retinal (fundus)** image to predict the stage of diabetic retinopathy.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fundus Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing the image..."):
            preds = predict(image)
            predicted_class = classes[np.argmax(preds)]
            confidence = np.max(preds) * 100

        st.success(f"### 🧠 Pre
