import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Sentiment (optional)
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    VADER_READY = True
except Exception:
    VADER_READY = False

def load_news_sentiment(news_path):
    news = pd.read_csv(news_path)
    news['Date'] = pd.to_datetime(news['Date']).dt.date
    if VADER_READY:
        news['compound'] = news['Headline'].astype(str).apply(lambda x: sid.polarity_scores(x)['compound'])
    else:
        # fallback: neutral sentiment if VADER not available
        news['compound'] = 0.0
    daily_sent = news.groupby('Date', as_index=False)['compound'].mean()
    daily_sent.rename(columns={'compound': 'news_sentiment'}, inplace=True)
    return daily_sent

def add_trading_indicators(df, price_col='Adj Close', vol_col='Volume'):
    dfe = df.copy()
    # Moving averages
    dfe['ma_10'] = dfe[price_col].rolling(10).mean()
    dfe['ma_30'] = dfe[price_col].rolling(30).mean()
    dfe['ma_60'] = dfe[price_col].rolling(60).mean()
    # Exponential MA
    dfe['ema_20'] = dfe[price_col].ewm(span=20, adjust=False).mean()
    # Returns
    dfe['return_1d'] = dfe[price_col].pct_change()
    dfe['return_5d'] = dfe[price_col].pct_change(5)
    # RSI (14)
    delta = dfe[price_col].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(14).mean()
    roll_down = pd.Series(down).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    dfe['rsi_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = dfe[price_col].ewm(span=12, adjust=False).mean()
    ema26 = dfe[price_col].ewm(span=26, adjust=False).mean()
    dfe['macd'] = ema12 - ema26
    dfe['macd_signal'] = dfe['macd'].ewm(span=9, adjust=False).mean()
    # Volume features
    if vol_col and vol_col in dfe.columns:
        dfe['vol_ma_10'] = dfe[vol_col].rolling(10).mean()
        dfe['vol_change'] = dfe[vol_col].pct_change()
    else:
        dfe['vol_ma_10'] = np.nan
        dfe['vol_change'] = np.nan
    return dfe

def merge_macro(df_prices, macro_path):
    macro = pd.read_csv(macro_path)
    macro['Date'] = pd.to_datetime(macro['Date']).dt.date
    df_prices_m = df_prices.copy()
    df_prices_m['Date'] = pd.to_datetime(df_prices_m['Date']).dt.date
    macro = macro.sort_values('Date')
    macro = macro.set_index('Date').ffill().reset_index()
    merged = pd.merge(df_prices_m, macro, on='Date', how='left')
    merged = merged.sort_values('Date')
    merged = merged.ffill().bfill()
    return merged

def assemble_features(df_raw, news_sent_path=None, macro_path=None, target_col='Adj Close'):
    df = df_raw.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    if target_col not in df.columns:
        target_col = 'Close'

    df = add_trading_indicators(df, price_col=target_col, vol_col='Volume' if 'Volume' in df.columns else None)

    # Sentiment
    if news_sent_path:
        daily_sent = load_news_sentiment(news_sent_path)
        df['Date_key'] = pd.to_datetime(df['Date']).dt.date
        df = pd.merge(df, daily_sent, left_on='Date_key', right_on='Date', how='left')
        df.drop(columns=['Date'], inplace=True)
        df.rename(columns={'Date_key': 'Date'}, inplace=True)
        df['news_sentiment'] = df['news_sentiment'].fillna(0.0)
    else:
        df['news_sentiment'] = 0.0

    # Macro
    if macro_path:
        df = merge_macro(df, macro_path)

    df = df.ffill().bfill()

    # Base + indicators + sentiment (+ any macro columns present)
    feature_cols = [
        target_col, 'ma_10', 'ma_30', 'ma_60', 'ema_20',
        'return_1d', 'return_5d', 'rsi_14', 'macd', 'macd_signal',
        'vol_ma_10', 'vol_change', 'news_sentiment'
    ]
    macro_cols = [c for c in df.columns if c not in feature_cols + ['Date']]
    use_cols = [c for c in feature_cols if c in df.columns] + macro_cols

    df_final = df[['Date'] + use_cols].dropna().reset_index(drop=True)
    return df_final, use_cols, target_col

def make_sequences_multivar(array_scaled, lookback=60, horizon=1):
    X, y = [], []
    for i in range(lookback, len(array_scaled) - horizon + 1):
        X.append(array_scaled[i - lookback:i, :])
        y.append(array_scaled[i + horizon - 1, 0])  # target = first column
    X = np.array(X)
    y = np.array(y)
    return X, y

def make_train_test_multivar(df_final, feature_cols, target_col='Adj Close', horizon=1, lookback=60, test_ratio=0.2):
    values = df_final[feature_cols].values.astype('float32')
    scaler_all = MinMaxScaler((0, 1))
    scaled = scaler_all.fit_transform(values)
    split_idx = int(len(scaled) * (1 - test_ratio))
    X_all, y_all = make_sequences_multivar(scaled, lookback=lookback, horizon=horizon)
    boundary = split_idx - lookback
    X_train, y_train = X_all[:boundary], y_all[:boundary]
    X_test, y_test = X_all[boundary:], y_all[boundary:]
    price_scaler = MinMaxScaler((0, 1))
    price_scaler.fit(values[:split_idx, [0]])
    return X_train, X_test, y_train, y_test, scaler_all, price_scaler
