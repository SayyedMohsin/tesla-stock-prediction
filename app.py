import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Tesla Stock Prediction (Dark)", layout="wide")

# ---------------------------
# Custom CSS for dark theme
# ---------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #1c1f26;
    }
    h1, h2, h3, h4 {
        color: #00ffcc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("data/TSLA.csv")
prices = df['Close'].values.reshape(-1,1)

# Scale data
scaler = MinMaxScaler((0,1))
scaled = scaler.fit_transform(prices)

# ---------------------------
# Title & description
# ---------------------------
st.title("🌌 Tesla Stock Prediction Dashboard (Dark Mode)")
st.write("Select your **Lookback Window** and **Prediction Horizon** from the sidebar to generate forecasts.")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Settings")
lookback = st.sidebar.selectbox("Lookback Window (days)", [30, 60, 90, 120])
horizon = st.sidebar.selectbox("Prediction Horizon (days)", [1, 5, 10])

# ---------------------------
# Prepare sequences dynamically
# ---------------------------
X, y = [], []
for i in range(lookback, len(scaled)):
    X.append(scaled[i-lookback:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ---------------------------
# Build model dynamically
# ---------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback,1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# ---------------------------
# Train quickly (demo mode)
# ---------------------------
with st.spinner("Training model..."):
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

# ---------------------------
# Predict future
# ---------------------------
last_sequence = scaled[-lookback:]
future_preds = []

current_seq = last_sequence.reshape(1, lookback, 1)
for _ in range(horizon):
    pred = model.predict(current_seq, verbose=0)[0][0]
    future_preds.append(pred)
    current_seq = np.append(current_seq[:,1:,:], [[[pred]]], axis=1)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

# ---------------------------
# Layout: chart + table
# ---------------------------
col1, col2 = st.columns([2,1])

with col1:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1, horizon+1), future_preds, marker='o', color="#00ffcc", linewidth=2)
    ax.set_facecolor("#0e1117")
    ax.set_title(f"Future {horizon}-Day Prediction (Lookback={lookback})", fontsize=14, color="#00ffcc")
    ax.set_xlabel("Days Ahead", color="#fafafa")
    ax.set_ylabel("Predicted Price ($)", color="#fafafa")
    ax.tick_params(colors="#fafafa")
    ax.grid(True, linestyle="--", alpha=0.4, color="#00ffcc")
    st.pyplot(fig)

with col2:
    st.subheader("📊 Predicted Prices")
    pred_df = pd.DataFrame({
        "Day Ahead": list(range(1, horizon+1)),
        "Predicted Price ($)": [f"{p[0]:.2f}" for p in future_preds]
    })
    st.table(pred_df)

st.success("✅ Forecast generated successfully (Dark Mode)")
