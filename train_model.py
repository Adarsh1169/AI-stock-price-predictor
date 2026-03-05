import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

def train_model(symbol):

    # Download stock data
    data = yf.download(symbol, start="2010-01-01")

    if data.empty:
        raise ValueError("No stock data found")

    close_prices = data["Close"].values.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    X = []
    y = []

    # Create sequences
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    # Build model
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(60,1)))
    model.add(LSTM(32))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    # Train model
    model.fit(X, y, epochs=5, batch_size=32)

    # Save model + scaler
    model.save(f"model_{symbol}.h5")
    joblib.dump(scaler, f"scaler_{symbol}.save")

    import os
    print("Saved model at:", os.getcwd())

    print("Model trained and saved")

    return model, scaler


if __name__ == "__main__":

    stocks = [
        "AAPL",
        "TCS.NS",
        "NTPC.NS",
        "ADANIPOWER.NS"
    ]

    for stock in stocks:
        print("Training for", stock)
        train_model(stock)

