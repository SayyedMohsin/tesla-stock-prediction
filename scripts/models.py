from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention

def build_simplernn(n_features, units=64, dropout=0.2, lr=0.002, lookback=60):
    model = Sequential([
        Input(shape=(lookback, n_features)),
        SimpleRNN(units, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def build_lstm(n_features, units=64, dropout=0.2, lr=0.002, lookback=60):
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def build_gru(n_features, units=64, dropout=0.2, lr=0.002, lookback=60):
    model = Sequential([
        Input(shape=(lookback, n_features)),
        GRU(units, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def build_transformer(n_features, d_model=64, n_heads=4, ff_dim=128, dropout=0.2, lr=0.002, lookback=60):
    inp = Input(shape=(lookback, n_features))
    x = Dense(d_model)(inp)
    attn_out = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff = Sequential([Dense(ff_dim, activation='relu'), Dropout(dropout), Dense(d_model)])(x)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    x_last = x[:, -1, :]
    out = Dense(1)(x_last)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model
