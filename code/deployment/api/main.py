from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Optional

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'art_auction_price_model.pkl')
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

app = FastAPI(title="Art Auction Price Prediction API", 
              description="API for predicting art auction prices", 
              version="1.0.0")

class ArtworkFeatures(BaseModel):
    creation_year: Optional[int] = None
    signed_binary: Optional[int] = 0
    avg_dimension_cm: Optional[float] = None
    lifespan: Optional[float] = None
    years_since_death: Optional[float] = None
    paintings: Optional[int] = None
    moma_artwork_count: Optional[int] = None
    is_living: Optional[bool] = True
    period: Optional[str] = "Contemporary"
    movement: Optional[str] = "Modern"
    size_category: Optional[str] = "Medium"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    features_used: dict

@app.get("/")
async def root():
    return {"message": "Art Auction Price Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: ArtworkFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        feature_dict = features.model_dump()
        
        # Create DataFrame with expected columns
        expected_columns = [
            'creation_year', 'signed_binary', 'avg_dimension_cm', 'lifespan',
            'years_since_death', 'paintings', 'moma_artwork_count', 'is_living'
        ]
        
        # Handle categorical variables (simplified)
        input_data = {col: [feature_dict.get(col, 0)] for col in expected_columns}
        
        df = pd.DataFrame(input_data)
        
        # Make prediction
        prediction_log = model.predict(df)[0]
        prediction = np.expm1(prediction_log)  # Convert back from log scale
        
        # Simple confidence estimation (you can improve this)
        confidence = 0.8 if all(df.notna().all()) else 0.5
        
        return PredictionResponse(
            prediction=round(prediction, 2),
            confidence=round(confidence, 2),
            features_used=feature_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": str(type(model).__name__),
        "features_expected": getattr(model, 'feature_names_in_', []).tolist() if hasattr(model, 'feature_names_in_') else []
    }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)