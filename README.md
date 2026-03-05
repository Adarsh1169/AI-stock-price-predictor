# AI Stock Price Predictor

An AI-powered web application that predicts stock prices using LSTM deep learning models.

## Overview
This project uses historical stock market data to predict future stock prices.  
It combines a machine learning model with a web interface and an API backend.

The application allows users to:
- Select a stock
- View recent historical data
- Get a predicted stock price

## Features
- Stock price prediction using LSTM neural networks
- FastAPI backend for prediction API
- Streamlit frontend for interactive UI
- Historical stock data fetched using yFinance
- Supports multiple stocks:
  - Apple (AAPL)
  - TCS
  - NTPC
  - Adani Power

## Tech Stack
- Python
- TensorFlow / Keras
- FastAPI
- Streamlit
- yFinance
- NumPy
- Scikit-learn
- Joblib

## Project Structure
api.py -> FastAPI backend server
app.py -> Streamlit frontend UI
predict.py -> Prediction logic
train_model.py -> LSTM model training script
.gitignore -> Ignore model and cache files

## How It Works

1. Historical stock data is downloaded using **yFinance**.
2. Data is scaled using **MinMaxScaler**.
3. An **LSTM neural network** is trained on past stock prices.
4. The trained model predicts future prices.
5. The prediction is served through a **FastAPI API**.
6. The **Streamlit app** sends requests to the API and displays the results.

7. ## Installation
Clone the repository:
git clone https://github.com/Adarsh1169/AI-stock-price-predictor.git


Navigate into the project folder:
cd AI-stock-price-predictor

Install required dependencies:
pip install -r requirements.txt


## Running the Project

### 1. Train the Models
python train_model.py


### 2. Start the FastAPI Server
Uvicorn api:app --reload


### 3. Run the Streamlit App
streamlit run app.py


The web interface will open in your browser.

## Future Improvements

- Add more stocks
- Improve prediction accuracy
- Add interactive charts
- Deploy the application online
- Integrate real-time stock data
