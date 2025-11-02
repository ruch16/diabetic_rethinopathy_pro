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
This model uses a trained deep learning CNN to analyze the image.
""")

# --- Upload section ---
uploaded_file = st.file_uploader("📤 Upload a retina image (JPG or PNG)", type=["jpg", "jpeg", "png"])

# --- Prediction labels ---
classes = [
    "No Diabetic Retinopathy (Healthy)",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

# --- Prediction function ---
def predict_dr(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))  # Change size if your model uses another input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions, axis=1)[0]
    return classes[pred_class], predictions[0]

# --- When an image is uploaded ---
if uploaded_file is not None:
    st.image(uploaded_file, caption="🖼️ Uploaded Retina Image", use_column_width=True)
    st.write("⏳ Analyzing... please wait...")

    label, prob = predict_dr(uploaded_file)

    st.success(f"✅ **Prediction:** {label}")
    st.write("**Confidence scores:**")
    for i, c in enumerate(classes):
        st.write(f"- {c}: {prob[i]:.4f}")

    st.markdown("---")
    st.markdown("⚕️ *AI-based prediction. Always consult an ophthalmologist for confirmation.*")
