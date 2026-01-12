import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("data/TSLA.csv")
prices = df['Close'].values.reshape(-1,1)

# Scale data
scaler = MinMaxScaler((0,1))
scaled = scaler.fit_transform(prices)

# Prepare sequences
lookback = 60
X, y = [], []
for i in range(lookback, len(scaled)):
    X.append(scaled[i-lookback:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback,1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X, y, epochs=5, batch_size=32)

# Save new model
model.save("models/LSTM_best_fixed.keras", save_format="keras")
print("✅ Model retrained and saved")
