import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_dataset(series, window_size=5):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

def preprocess_for_lstm(df, window_size=5):
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    X, y = create_dataset(scaled, window_size)
    return X.reshape((X.shape[0], X.shape[1], 1)), y, scaler
