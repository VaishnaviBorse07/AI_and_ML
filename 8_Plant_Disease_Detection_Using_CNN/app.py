import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("plant_disease_model.h5")
labels = list(os.listdir("PlantVillage"))

st.title("🌿 Plant Disease Classifier")
file = st.file_uploader("Upload a leaf image", type=["jpg", "png"])

if file:
    img = Image.open(file).resize((128, 128))
    st.image(img, caption="Uploaded Leaf", width=150)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)
    st.success(f"Prediction: {labels[np.argmax(pred)]}")
