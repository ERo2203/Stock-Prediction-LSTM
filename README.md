# Stock-Prediction-LSTM

## Stock Anomaly Detector

This project implements a real-time stock anomaly detection system using:

- LSTM-based prediction error
- Volume change analysis
- Moving average crossover detection
- Confirmed anomaly filtering using combined methods

## Features

- Monitors stocks using 1-minute interval data
- Detects anomalies using machine learning and rule-based methods
- Saves results to CSV files for further analysis
- Optional visualization of predictions and anomalies
- Configurable real-time loop for continuous monitoring

## Requirements

- Python 3.8 or higher
- Required Python packages are listed in `requirements.txt`

To install the dependencies, run:


## How to Run

1. Clone this repository or download the source code.

2. Run the main script using Python:


This script will:

- Download 1-minute interval stock price data (up to the past 5 days)
- Train a lightweight LSTM model on each stock
- Detect anomalies based on prediction error, volume, and moving average logic
- Export anomaly results into the `data/` directory as CSV files

## Folder Structure

stock-anomaly-detector/
├── main.py # Main entry point for running detection
├── requirements.txt # Required Python packages
├── README.md # This file
├── model/
│ └── lstm_utils.py # LSTM model training and anomaly detection
├── utils/
│ ├── fetch.py # Stock data downloader and volume anomaly detection
│ ├── preprocessing.py # Scaling and windowing for LSTM
│ └── visualization.py # Plotting helper functions (optional)
├── data/
│ └── *.csv # Output anomaly detection files


## Output

After execution, the script will generate CSV files in the `data/` folder. Each stock symbol will have:

- `<SYMBOL>_LSTM_anomalies.csv` – anomalies detected by LSTM only
- `<SYMBOL>_confirmed_anomalies.csv` – anomalies confirmed by both LSTM and moving average crossover

## Customization

You can customize the following in `main.py`:

- Stock symbols to monitor
- Loop timing and iteration limits
- Model parameters
- Detection thresholds






