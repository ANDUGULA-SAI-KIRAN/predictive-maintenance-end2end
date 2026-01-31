# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.predict import get_best_model_metadata, predict_with_confidence, cache
from src.models.shap_analysis import get_shap_plots

app = FastAPI(title="Maintenance AI Backend")

class PredictRequest(BaseModel):
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: int

@app.post("/select_model")
def select_model(model_name: str):
    metadata = get_best_model_metadata(model_name)
    if not metadata:
        raise HTTPException(status_code=404, detail="Model not found in MLflow")
    return metadata

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # 1. Get Prediction
        results = predict_with_confidence(request.dict())
        
        # 2. Get SHAP Plots
        shap_plots = get_shap_plots(cache.model, results['processed_df'])
        
        return {
            "prediction_details": {
                "label": results['label'],
                "confidence": results['confidence'],
                "probability": results['probability']
            },
            "plots": shap_plots
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))