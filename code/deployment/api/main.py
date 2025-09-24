from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Art Auction Price Prediction API")

try:
    model = joblib.load("../../../models/art_auction_price_model_linearrgression.pkl")
except:
    model = None

class ArtworkFeatures(BaseModel):
    period: str
    movement: str
    size_category: str
    creation_year: int
    signed_binary: int

@app.get("/")
def read_root():
    return {"message": "Art Auction Price Prediction API"}

@app.post("/predict")
def predict_price(features: ArtworkFeatures):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Prepare features for prediction
    feature_dict = features.dict()
    
    # Here you would need to preprocess the features similarly to training
    # This is a simplified version
    try:
        # Convert to DataFrame with correct feature names
        # This needs to match the training feature names
        prediction = 100.0  # Placeholder
        return {"predicted_price_usd": round(prediction, 2)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}