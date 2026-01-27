# # src/api/app.py

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Dict, Optional
# from src.models.predict import list_available_models, load_model, load_threshold, predict

# app = FastAPI(title="Predictive Maintenance API")

# # Cache to store selected model
# selected_model_cache = None

# # Pydantic model for request body validation
# class PredictRequest(BaseModel):
#     type: str
#     air_temperature: float
#     process_temperature: float
#     rotational_speed: float
#     torque: float
#     tool_wear: int

# @app.get("/models")
# def get_models():
#     """
#     Get available models and select one to be used for predictions.
#     Fetches models from the MLflow tracking database.
#     """
#     models = list_available_models()  # Get available models from MLflow
#     return {"available_models": models}

# @app.post("/select_model")
# def select_model_endpoint(model_name: str):
#     """
#     Endpoint to select a model for future predictions.
#     The selected model will be cached for use in predictions.
#     :param model_name: The model to select (e.g., "LGBM_class_weighted" or "RF_baseline")
#     """
#     global selected_model_cache
#     try:
#         models = list_available_models()
#         model = next(m for m in models if m["model_name"] == model_name)
        
#         # Load the selected model and threshold from MLflow
#         selected_model = load_model(model["run_id"], model["model_name"])
#         threshold = load_threshold(model["run_id"])
        
#         # Cache the model and threshold
#         selected_model_cache = {"model": selected_model, "threshold": threshold}
        
#         return {"message": f"Model {model_name} selected successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/predict")
# def predict_endpoint(request: PredictRequest):
#     """
#     API endpoint for making predictions.
#     Accepts input features from the user and uses the cached model for prediction.
#     :param request: Input features from the user
#     :return: Prediction result
#     """
#     if selected_model_cache is None:
#         raise HTTPException(status_code=400, detail="No model selected. Please select a model first.")

#     input_features = request.model_dump()
#     try:
#         # Predict using the cached model
#         result = predict(input_features, model=selected_model_cache["model"], threshold=selected_model_cache["threshold"])
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.get("/health")
# def health_check():
#     """
#     Health check endpoint to verify the API is up and running.
#     :return: Status message
#     """
#     return {"status": "ok"}



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