import yfinance as yf
import pandas as pd

def fetch_stock_data(symbols, interval='1m', period='5d'):
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
            if not df.empty:
                df.dropna(inplace=True)
                data[symbol] = df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    return data

def detect_volume_anomalies(df, threshold=0.3):
    vol = df['Volume']
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    vol_change = vol.pct_change().fillna(0)
    return df.index[(abs(vol_change) > threshold)]
