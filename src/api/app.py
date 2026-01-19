from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
from src.models.predict import predict, list_available_models, load_model, model_cache, load_threshold
from src.models.select_best import select_latest_model

app = FastAPI(title="Predictive Maintenance API")

class PredictRequest(BaseModel):
    torque: float
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    tool_wear: int
    model_type: Optional[str] = "LGBM_class_weighted"  # default model

@app.get("/models")
def list_models() -> Dict[str, Any]:
    """
    API endpoint to list all available models.
    :return: List of available models in MLflow
    """
    models = list_available_models()  # Fetch models from MLflow or database
    return {"models": models}

@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> Dict[str, Any]:
    """
    API endpoint for making predictions.
    :param request: Input features and model type
    :return: Prediction result in the form of a dictionary
    """
    input_features = request.model_dump()
    model_type = input_features.pop("model_type")  # remove model_type from features
    
    # Check if model is in the cache, if not, load it
    if model_type not in model_cache:
        try:
            # Fetch and cache the selected model
            model_meta, model, threshold = select_latest_model(model_type)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        model = model_cache[model_type]
        threshold = load_threshold(model_type)  # Load the threshold if needed

    # Predict using the selected model
    try:
        result = predict(model, input_features, threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify the API is up and running.
    :return: Status message
    """
    return {"status": "ok"}
