import streamlit as st
import requests
import yfinance as yf

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📈 AI Stock Price Predictor")

stocks = {
    "Apple": "AAPL",
    "TCS": "TCS.NS",
    "NTPC": "NTPC.NS",
    "Adani Power": "ADANIPOWER.NS"
}

stock_name = st.selectbox(
    "Select Stock",
    ["Select Stock"] + list(stocks.keys())
)

if stock_name == "Select Stock":
    st.stop()

stock_symbol = stocks[stock_name]

if st.button("Predict"):

    # Get stock data
    data = yf.download(stock_symbol, period="3mo")
    data = data.sort_index(ascending=False)

    if data.empty:
        st.error("Stock data unavailable.")
        st.stop()

    current_price = data["Close"].iloc[-1]

    st.subheader("Stock Price (Last 3 Months)")
    st.dataframe(data.tail(60))

    st.subheader("Stock Price Graph")
    chart_data = data.copy()
    chart_data = chart_data.sort_index()
    st.line_chart(chart_data["Close"].reset_index(drop=True))
    

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"symbol": stock_symbol}
        )

        result = response.json()
        predicted_price = result["predicted_price"]
        difference = predicted_price - current_price
        

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current Price", round(current_price, 2))

        with col2:
            st.metric("Predicted Price", round(predicted_price, 2))

            if difference > 0:
                st.success("📈 Model predicts the stock may go UP")
            else:
                st.warning("📉 Model predicts the stock may go DOWN")

    except Exception as e:
        st.warning("Prediction service temporarily slow. Please wait and try again.")

