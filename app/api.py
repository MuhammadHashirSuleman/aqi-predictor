from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add project root to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.predictor import AQIPredictor

app = FastAPI(title="Pearls AQI Predictor API", version="1.0")

# Global predictor instance (loaded on startup)
predictor = None

class PredictionResponse(BaseModel):
    aqi_24h: float
    aqi_48h: float
    aqi_72h: float

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = AQIPredictor()
        predictor.load_model()
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor.model is not None}

@app.get("/predict", response_model=PredictionResponse)
def predict_aqi():
    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch latest features automatically
        preds = predictor.predict()
        
        if preds is None:
             raise HTTPException(status_code=500, detail="Prediction failed or no data")
             
        return {
            "aqi_24h": float(preds[0]),
            "aqi_48h": float(preds[1]),
            "aqi_72h": float(preds[2])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
