import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------------------------------------
# Load trained DR model
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("dr_model.h5")
    return model

model = load_model()

# Class labels
CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ----------------------------------------------------------
# Prediction Function
# ----------------------------------------------------------
def predict(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return preds[0]

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.set_page_config(page_title="🩺 Diabetic Retinopathy Detection", page_icon="🧠")

st.markdown("<h1 style='text-align: center; color: #d63384;'>🩺 Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
st.write("Upload a **retinal (fundus)** image to detect the stage of Diabetic Retinopathy using AI.")

uploaded_file = st.file_uploader("📤 Upload Fundus Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🩸 Uploaded Fundus Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing the image... please wait ⏳"):
            preds = predict(image)
            predicted_class = CLASSES[np.argmax(preds)]
            confidence = np.max(preds) * 100

        st.success(f"### ✅ Predicted Stage: **{predicted_class}**")
        st.write("### Confidence Levels:")
        for i, label in enumerate(CLASSES):
            st.write(f"{label}: {preds[i]*100:.2f}%")

        st.progress(int(confidence))
        st.balloons()

st.markdown("<br><hr><center>Made with ❤️ using Streamlit & TensorFlow</center>", unsafe_allow_html=True)

