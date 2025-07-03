# stock-anomaly-detector/utils/visualization.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_lstm_prediction(df, y_true_inv, y_pred_inv, window_size, anomalies=None, symbol=""):
    plt.figure(figsize=(12, 4))
    plt.plot(df.index[window_size:], y_true_inv, label='True')
    plt.plot(df.index[window_size:], y_pred_inv, label='Predicted')
    if anomalies is not None:
        plt.scatter(anomalies, y_true_inv[[df.index[window_size:].get_loc(ts) for ts in anomalies]], 
                    color='red', label='LSTM Anomaly', marker='x')
    plt.title(f"LSTM Prediction vs Actual - {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_volume_anomalies(df, anomaly_idx, symbol=""):
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['Volume'], label='Volume')
    plt.scatter(anomaly_idx, df.loc[anomaly_idx]['Volume'], color='orange', label='Vol Anomaly', marker='o')
    plt.title(f"Volume Anomalies - {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_moving_averages(df_ma, bullish_idx, bearish_idx, symbol=""):
    plt.figure(figsize=(12, 4))
    plt.plot(df_ma.index, df_ma['Close'], label='Price')
    plt.plot(df_ma.index, df_ma['SMA_short'], label='SMA5')
    plt.plot(df_ma.index, df_ma['SMA_long'], label='SMA20')
    plt.scatter(bullish_idx, df_ma.loc[bullish_idx]['Close'], marker='^', color='green', label='Bullish Cross')
    plt.scatter(bearish_idx, df_ma.loc[bearish_idx]['Close'], marker='v', color='red', label='Bearish Cross')
    plt.title(f"MA Crossovers - {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(error, symbol=""):
    plt.figure(figsize=(8, 4))
    plt.hist(error.flatten(), bins=50, color='purple', alpha=0.7)
    plt.title(f"Prediction Error Distribution - {symbol}")
    plt.xlabel("Relative Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
