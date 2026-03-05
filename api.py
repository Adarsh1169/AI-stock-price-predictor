from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_stock

app = FastAPI()

class StockInput(BaseModel):
    symbol: str

@app.get("/")
def home():
    return {"message":"Stock Prediction API"}

@app.post("/predict")

def predict(data: StockInput):

    price = predict_stock(data.symbol)

    return {
        "stock": data.symbol,
        "predicted_price": price
    }