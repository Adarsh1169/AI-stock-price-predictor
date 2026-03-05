import yfinance as yf
import numpy as np
import joblib
from tensorflow.keras.models import load_model


def predict_stock(symbol):

    # Load correct model for selected stock
    model = load_model(f"model_{symbol}.h5", compile=False)
    scaler = joblib.load(f"scaler_{symbol}.save")

    data = yf.download(symbol, period="3mo")

    if data.empty:
        return {"error": "No stock data found"}

    close_prices = data["Close"].values.reshape(-1,1)

    scaled_data = scaler.transform(close_prices)

    last_60 = scaled_data[-60:]

    X_test = np.array([last_60])

    prediction = model.predict(X_test)

    predicted_price = scaler.inverse_transform(prediction)

    return float(predicted_price[0][0])