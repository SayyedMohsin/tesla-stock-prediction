import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from scripts.features import assemble_features, make_train_test_multivar
from scripts.models import build_simplernn, build_lstm, build_gru, build_transformer

# --- Config ---
DATA_PATH = os.path.join("data", "TSLA.csv")
NEWS_PATH = os.path.join("data", "tesla_news.csv")  # optional
MACRO_PATH = os.path.join("data", "macro.csv")      # optional

OUTPUT_DIR = os.path.join("outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

HORIZON = 1           # 1, 5, 10
LOOKBACK = 60         # 30-120
EPOCHS = 50
BATCH_SIZE = 32

def plot_actual_pred(y_true, y_pred, title, save_name):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Actual', color='black')
    plt.plot(y_pred, label='Prediction', color='tab:blue')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {save_path}")

def train_eval(model, X_train, y_train, X_test, y_test, price_scaler, name="Model", epochs=50, batch_size=32):
    hist = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1)
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_pred = price_scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = price_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mse = mean_squared_error(y_true, y_pred)
    print(f"{name} Test MSE: {mse:.4f}")
    return y_true, y_pred, mse, hist

def main():
    # 1) Load prices
    df_prices = pd.read_csv(DATA_PATH)
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])

    # 2) Assemble features (put None if you don’t have those CSVs)
    news_path = NEWS_PATH if os.path.exists(NEWS_PATH) else None
    macro_path = MACRO_PATH if os.path.exists(MACRO_PATH) else None

    df_feat, feat_cols, tgt_col = assemble_features(
        df_prices,
        news_sent_path=news_path,
        macro_path=macro_path,
        target_col='Adj Close'
    )
    print("Using features:", feat_cols)

    # 3) Train/Test split
    X_train, X_test, y_train, y_test, scaler_all, price_scaler = make_train_test_multivar(
        df_final=df_feat,
        feature_cols=feat_cols,
        target_col=tgt_col,
        horizon=HORIZON,
        lookback=LOOKBACK,
        test_ratio=0.2
    )
    n_features = X_train.shape[-1]

    # 4) Build models
    rnn_model = build_simplernn(n_features=n_features, lookback=LOOKBACK)
    lstm_model = build_lstm(n_features=n_features, lookback=LOOKBACK)
    gru_model = build_gru(n_features=n_features, lookback=LOOKBACK)
    trf_model = build_transformer(n_features=n_features, lookback=LOOKBACK)

    # 5) Train & evaluate each
    y_true_rnn, y_pred_rnn, mse_rnn, _ = train_eval(rnn_model, X_train, y_train, X_test, y_test, price_scaler, name="SimpleRNN", epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_actual_pred(y_true_rnn, y_pred_rnn, f"Actual vs Predicted (SimpleRNN, horizon={HORIZON})", "simplernn.png")

    y_true_lstm, y_pred_lstm, mse_lstm, _ = train_eval(lstm_model, X_train, y_train, X_test, y_test, price_scaler, name="LSTM", epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_actual_pred(y_true_lstm, y_pred_lstm, f"Actual vs Predicted (LSTM, horizon={HORIZON})", "lstm.png")

    y_true_gru, y_pred_gru, mse_gru, _ = train_eval(gru_model, X_train, y_train, X_test, y_test, price_scaler, name="GRU", epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_actual_pred(y_true_gru, y_pred_gru, f"Actual vs Predicted (GRU, horizon={HORIZON})", "gru.png")

    y_true_trf, y_pred_trf, mse_trf, _ = train_eval(trf_model, X_train, y_train, X_test, y_test, price_scaler, name="Transformer", epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_actual_pred(y_true_trf, y_pred_trf, f"Actual vs Predicted (Transformer, horizon={HORIZON})", "transformer.png")

    # 6) DL ensemble (average)
    # Use the same y_true reference (they match indices); pick one (e.g., LSTM)
    y_pred_ensemble = (y_pred_rnn + y_pred_lstm + y_pred_gru + y_pred_trf) / 4.0
    mse_ens = mean_squared_error(y_true_lstm, y_pred_ensemble)
    print(f"Ensemble (RNN+LSTM+GRU+Transformer) Test MSE: {mse_ens:.4f}")
    plot_actual_pred(y_true_lstm, y_pred_ensemble, f"Actual vs Predicted (Ensemble, horizon={HORIZON})", "ensemble.png")

    # 7) Save summary
    results_path = os.path.join(OUTPUT_DIR, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Horizon={HORIZON}, Lookback={LOOKBACK}\n")
        f.write(f"SimpleRNN MSE: {mse_rnn:.4f}\n")
        f.write(f"LSTM MSE: {mse_lstm:.4f}\n")
        f.write(f"GRU MSE: {mse_gru:.4f}\n")
        f.write(f"Transformer MSE: {mse_trf:.4f}\n")
        f.write(f"Ensemble MSE: {mse_ens:.4f}\n")
    print(f"Saved metrics: {results_path}")

if __name__ == "__main__":
    main()
