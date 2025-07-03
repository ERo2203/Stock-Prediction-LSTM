
---

### `main.py`

```python
from utils.fetch import fetch_stock_data
from utils.preprocessing import preprocess_for_lstm
from model.lstm_utils import tune_model_manual, detect_anomalies

stock_symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'COALINDIA.NS', 'INFY.NS']
stock_data = fetch_stock_data(stock_symbols, interval='1m', period='5d')

detect_anomalies(stock_data, stock_symbols, window_size=5)
