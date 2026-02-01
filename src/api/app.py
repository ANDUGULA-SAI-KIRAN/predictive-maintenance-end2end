# # src/api/app.py
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from src.models.predict import get_best_model_metadata, predict_with_confidence, cache
# from src.models.shap_analysis import get_shap_plots

# app = FastAPI(title="Maintenance AI Backend")

# class PredictRequest(BaseModel):
#     type: str
#     air_temperature: float
#     process_temperature: float
#     rotational_speed: float
#     torque: float
#     tool_wear: int

# @app.post("/select_model")
# def select_model(model_name: str):
#     metadata = get_best_model_metadata(model_name)
#     if not metadata:
#         raise HTTPException(status_code=404, detail="Model not found in MLflow")
#     return metadata

# @app.post("/predict")
# def predict(request: PredictRequest):
#     try:
#         # 1. Get Prediction
#         results = predict_with_confidence(request.dict())
        
#         # 2. Get SHAP Plots
#         shap_plots = get_shap_plots(cache.model, results['processed_df'])
        
#         return {
#             "prediction_details": {
#                 "label": results['label'],
#                 "confidence": results['confidence'],
#                 "probability": results['probability']
#             },
#             "plots": shap_plots
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.models.predict import model_server
from src.models.shap_analysis import get_shap_plots

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Load the best model from DagsHub Registry
    print("Connecting to DagsHub Model Registry...")
    try:
        model_server.load_production_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Startup Error: {e}")
    yield
    # SHUTDOWN: Clean up if necessary
    print("Shutting down API...")

app = FastAPI(title="Maintenance AI Pro Backend", lifespan=lifespan)

class PredictRequest(BaseModel):
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: int

@app.get("/health")
def health():
    return {"status": "ready", "model_metadata": model_server.metadata}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # 1. Prediction using the loaded singleton
        results = model_server.predict(request.dict())
        
        # 2. SHAP Analysis (using the raw model inside the server)
        # Note: access the underlying flavor (sklearn/lgbm) if needed for SHAP
        shap_plots = get_shap_plots(model_server.model, results['processed_df'])
        
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