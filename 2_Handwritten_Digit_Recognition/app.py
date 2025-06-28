# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")
st.title("🧠 Handwritten Digit Recognition")

# Load model
if not os.path.exists("mnist_cnn_model.h5"):
    st.error("❌ Model not found. Run train_model.py to create mnist_cnn_model.h5")
    st.stop()

model = load_model("mnist_cnn_model.h5")

st.markdown("Upload a **28x28 pixel grayscale** image of a digit (0–9) or draw it using a canvas.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess
    img = Image.open(uploaded_file).convert("L")         # Grayscale
    img_resized = img.resize((28, 28))                   # Resize
    img_inverted = ImageOps.invert(img_resized)          # Invert
    img_norm = np.array(img_inverted) / 255.0            # Normalize
    input_img = img_norm.reshape(1, 28, 28, 1)

    # Show image
    st.image(img_resized, caption="Uploaded Image (Resized to 28x28)", width=150)

    if st.button("🔍 Predict"):
        prob = model.predict(input_img)[0]
        prediction = np.argmax(prob)
        confidence = np.max(prob) * 100
        st.success(f"🧾 Predicted Digit: **{prediction}**")
        st.info(f"Confidence: {confidence:.2f}%")
