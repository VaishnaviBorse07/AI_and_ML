import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

st.title("📈 Stock Price Predictor - AAPL")
model = load_model("stock_model.h5")
scaler = joblib.load("stock_scaler.pkl")
data = np.load("stock_data.npy")

X_test = np.array([data[-60:]])
X_test = X_test.reshape(1, 60, 1)

prediction = model.predict(X_test)
price = scaler.inverse_transform(prediction)[0][0]
st.success(f"📊 Predicted next close price: ${price:.2f}")
