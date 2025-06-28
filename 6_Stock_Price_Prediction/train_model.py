import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

data = yf.download('AAPL', start='2015-01-01', end='2022-01-01')['Close']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data.values.reshape(-1, 1))

X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5)

model.save("stock_model.h5")
joblib.dump(scaler, "stock_scaler.pkl")
np.save("stock_data.npy", scaled)
