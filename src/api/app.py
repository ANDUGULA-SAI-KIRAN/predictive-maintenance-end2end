# src/api/app.py
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from src.models.predict import manager, predict_with_explanation
from src.models.shap_analysis import get_shap_plots
from typing import Literal

app = FastAPI(
    title="Predictive Maintenance AI API",
    description="API for high-precision failure detection using @champion models from MLflow.",
    version="2.0.0"
)

# 1. Define Request Schema
class MaintenanceRequest(BaseModel):
    type: Literal["Low", "Medium", "High"]
    air_temperature: float = Field(..., gt=200, lt=400)
    process_temperature: float = Field(..., gt=200, lt=400)
    rotational_speed: int = Field(..., gt=0)
    torque: float = Field(..., gt=0)
    tool_wear: int = Field(..., ge=0)

# 2. Endpoint: Health Check
@app.get("/health")
def health_check():
    return {"status": "online", "registry": "connected"}

# 3. Endpoint: Get Model Info (Used by UI Sidebar)
@app.get("/model_info/{model_label}")
def get_model_info(model_label: str):
    """Fetches metadata for the current @champion without running a prediction."""
    try:
        model_data = manager.get_model(model_label)
        return {
            "model": model_label,
            "metadata": model_data["meta"],
            "last_sync": model_data["last_updated"]
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch {model_label}: {str(e)}")

# 4. Endpoint: Predict + Explain
@app.post("/predict/{model_label}")
def predict_and_explain(model_label: str, request: MaintenanceRequest):
    """
    Runs prediction and generates SHAP explanations in one atomic call.
    """
    try:
        # A. Run Prediction Logic
        results = predict_with_explanation(model_label, request.model_dump())
        
        # B. Generate SHAP Plots
        # We fetch the model directly from the manager's cache
        pyfunc_model = manager.get_model(model_label)["model"]
        # 1. Access the underlying flavor implementation
        # For Native LightGBM flavor, it is stored in ._model_impl.lgbm_model
        # if hasattr(pyfunc_model, "_model_impl"):
        #     impl = pyfunc_model._model_impl
            
        #     # Check for LightGBM
        #     if hasattr(impl, "lgbm_model"):
        #         raw_model = impl.lgbm_model
        #     # Check for Scikit-Learn (Random Forest)
        #     elif hasattr(impl, "sklearn_model"):
        #         raw_model = impl.sklearn_model
        #     else:
        #         raw_model = impl
        # else:
        #     raw_model = pyfunc_model

        shap_plots = get_shap_plots(pyfunc_model, results['processed_df'])
        
        return {
            "prediction_details": {
                "label": results['label'],
                "probability": results['probability'],
                "prediction_code": results['prediction']
            },
            "model_context": results['meta'],
            "explanations": shap_plots
        }
        
    except Exception as e:
        # In production, we log the full traceback but return a clean error
        print(f"Prediction Error: {e}") 
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")