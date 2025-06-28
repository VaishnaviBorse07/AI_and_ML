import streamlit as st
import joblib

model = joblib.load("emotion_model.pkl")
vec = joblib.load("emotion_vectorizer.pkl")
le = joblib.load("emotion_encoder.pkl")

st.title("😊 Emotion Detector from Text")
text = st.text_area("Enter a sentence:")

if st.button("Analyze Emotion"):
    vec_input = vec.transform([text])
    prediction = model.predict(vec_input)[0]
    label = le.inverse_transform([prediction])[0]
    st.success(f"Detected Emotion: **{label}**")
