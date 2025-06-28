import streamlit as st
import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("fake_news_vectorizer.pkl")

st.title("📰 Fake News Detector")
input_text = st.text_area("Paste News Article:")

if st.button("Predict"):
    input_vector = vectorizer.transform([input_text])
    result = model.predict(input_vector)[0]
    st.success("Prediction: FAKE" if result == 1 else "Prediction: REAL")
