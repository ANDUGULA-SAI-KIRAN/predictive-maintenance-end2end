# src/models/predict.py

from typing import Dict, Any
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import mlflow.lightgbm
from src.features.feature_engineering import add_engineered_features

# MLflow config (match train.py)
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "predictive_maintenance"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

TYPE_MAPPING = {'low': 0, 'medium': 1, 'high': 2}

def preprocess_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply type feature encoding (convert to lowercase and map to numeric)."""
    df['type'] = df['type'].str.lower()  # Convert to lowercase
    df['type'] = df['type'].map(TYPE_MAPPING)  # Map categories to numeric values
    return df


# Model loading function
def load_model(run_id: str, model_name: str):
    """
    Load model from MLflow.
    """
    if model_name.startswith("LGBM"):
        artifact_name = "LGBM_model"
        model_uri = f"runs:/{run_id}/{artifact_name}"
        return mlflow.lightgbm.load_model(model_uri)
    else:
        artifact_name = "RF_model"
        model_uri = f"runs:/{run_id}/{artifact_name}"
        return mlflow.sklearn.load_model(model_uri)


# Model selection logic (to get available models)
def list_available_models() -> list:
    """Fetch available models from MLflow."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"]
    )

    models = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        if run_name in {"RF_baseline", "LGBM_class_weighted"}:
            models.append({
                "model_name": run_name,
                "run_id": run.info.run_id,
                "start_time": str(run.info.start_time)
            })
    return models


# Threshold loading function
def load_threshold(run_id: str) -> float:
    """Load optimal threshold from MLflow run metrics."""
    client = MlflowClient()
    run = client.get_run(run_id)
    threshold = run.data.metrics.get("optimal_threshold", 0.5)
    return float(threshold)


# Prediction function (with feature engineering)
def predict(input_features: Dict[str, Any], model, threshold: float) -> Dict[str, Any]:
    """
    Perform prediction after applying feature engineering.
    :param input_features: The input features from the API request.
    :param model: The loaded ML model.
    :param threshold: The optimal threshold for classification.
    :return: A dictionary containing the prediction and probability.
    """
    # Convert API input dict to training column names
    field_map = {
        "torque": "torque",
        "type": "type",
        "air_temperature": "air_temp",
        "process_temperature": "process_temp",
        "rotational_speed": "rpm",
        "tool_wear": "tool_wear"
    }
    df_input = pd.DataFrame([{field_map[k]: v for k, v in input_features.items()}])
    
    # Preprocess type column (encoding)
    df_input = preprocess_type_column(df_input)

    # Apply feature engineering (match training features)
    df_fe = add_engineered_features(df_input)

    # Predict probability (original flavor)
    prob = float(model.predict_proba(df_fe)[0][1])  # Assumes binary classification (0/1)
    pred = int(prob >= threshold)

    return {
        "probability": prob,
        "threshold": threshold,
        "prediction": pred,
        "label": "Machine Failed" if pred == 1 else "No Failure"
    }
