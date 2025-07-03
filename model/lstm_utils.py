import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import random
from utils.preprocessing import preprocess_for_lstm
from utils.fetch import detect_volume_anomalies, detect_moving_average_anomalies

def tune_model_manual(X, y, window_size):
    best_loss = float('inf')
    best_model = None
    param_grid = {
        "units": [16, 32, 64],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [1e-2, 1e-3],
        "batch_size": [8, 16, 32]
    }

    for _ in range(5):
        units = random.choice(param_grid["units"])
        dropout = random.choice(param_grid["dropout"])
        learning_rate = random.choice(param_grid["learning_rate"])
        batch_size = random.choice(param_grid["batch_size"])

        model = Sequential()
        model.add(LSTM(units=units, input_shape=(window_size, 1)))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        history = model.fit(X, y, epochs=50, batch_size=batch_size, verbose=0)

        loss = history.history['loss'][-1]
        if loss < best_loss:
            best_loss = loss
            best_model = model
    return best_model

def detect_anomalies(stock_data, stock_symbols, window_size=5):
    for symbol in stock_symbols:
        df = stock_data.get(symbol)
        if df is None or len(df) < window_size + 1:
            continue

        X, y, scaler = preprocess_for_lstm(df, window_size)
        model = tune_model_manual(X, y, window_size)

        y_pred = model.predict(X)
        y_true = y.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        y_true_inv = scaler.inverse_transform(y_true)
        y_pred_inv = scaler.inverse_transform(y_pred)

        error = np.abs((y_true_inv - y_pred_inv) / y_true_inv)
        error_threshold = np.mean(error) + 2 * np.std(error)

        anomalies = error > error_threshold
        anomaly_times = df.index[window_size:][anomalies.flatten()]
        anomaly_log = pd.DataFrame({
            'Timestamp': anomaly_times,
            'Actual Price': y_true_inv[anomalies.flatten()].flatten(),
            'Predicted Price': y_pred_inv[anomalies.flatten()].flatten(),
            'Symbol': symbol
        })

        anomaly_log.to_csv(f"data/{symbol}_LSTM_anomalies.csv", index=False)

        # MA + Volume
        volume_anomalies = detect_volume_anomalies(df)
        bullish_idx, bearish_idx, df_ma = detect_moving_average_anomalies(df)

        if not isinstance(bullish_idx, pd.Index): bullish_idx = pd.Index(bullish_idx)
        if not isinstance(bearish_idx, pd.Index): bearish_idx = pd.Index(bearish_idx)

        combined_idx = bullish_idx.union(bearish_idx)
        combined_anomalies = set(anomaly_times).intersection(set(combined_idx))

        confirmed = anomaly_log[anomaly_log['Timestamp'].isin(combined_anomalies)]
        confirmed.to_csv(f"data/{symbol}_confirmed_anomalies.csv", index=False)
