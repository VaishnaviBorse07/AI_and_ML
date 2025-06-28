# app.py

import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# UI
st.set_page_config(page_title="Email Spam Detector", page_icon="📧")
st.title("📧 Email Spam Detector")
st.write("Enter your email message below to check whether it is spam or not.")

user_input = st.text_area("✉️ Message:", height=150)

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        clean_input = clean_text(user_input)
        input_vector = vectorizer.transform([clean_input])
        prediction = model.predict(input_vector)[0]
        label = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
        st.success(f"Prediction: **{label}**")
