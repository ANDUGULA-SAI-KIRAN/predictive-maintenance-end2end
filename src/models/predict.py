# src/models/predict.py

from typing import Dict, Any, List
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import mlflow.lightgbm

from src.features.feature_engineering import add_engineered_features

# -----------------------------
# MLflow config (match train.py)
# -----------------------------
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "predictive_maintenance"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Cache to avoid reloading the models every time
model_cache = {}

TYPE_MAPPING = {'low': 0, 'medium': 1, 'high': 2}

def preprocess_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply type feature encoding (convert to lowercase and map to numeric)."""
    df['type'] = df['type'].str.lower()  # Convert to lowercase
    df['type'] = df['type'].map(TYPE_MAPPING)  # Map categories to numeric values
    return df

# -----------------------------
# MODEL DISCOVERY
# -----------------------------
def list_available_models() -> List[Dict[str, str]]:
    """List available models from MLflow experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"]
    )

    models: List[Dict[str, str]] = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        if run_name in {"RF_baseline", "LGBM_class_weighted"}:
            models.append({
                "model_name": run_name,
                "run_id": run.info.run_id,
                "start_time": str(run.info.start_time)
            })
    return models

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(run_id: str, model_name: str):
    """
    Load original model flavor so predict_proba() works
    """
    if model_name.startswith("LGBM"):
        artifact_name = "LGBM_model"
        model_uri = f"runs:/{run_id}/{artifact_name}"
        return mlflow.lightgbm.load_model(model_uri)
    else:
        artifact_name = "RF_model"
        model_uri = f"runs:/{run_id}/{artifact_name}"
        return mlflow.sklearn.load_model(model_uri)

# -----------------------------
# LOAD THRESHOLD
# -----------------------------
def load_threshold(run_id: str) -> float:
    """Load optimal threshold from MLflow run metrics."""
    client = MlflowClient()
    run = client.get_run(run_id)
    threshold = run.data.metrics.get("optimal_threshold", 0.5)
    return float(threshold)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(model, input_features: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """
    Perform prediction:
    - Convert API input dict → training column names
    - Apply feature engineering
    - Return probability, threshold, label
    """
    # Map API fields → training columns
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

    # Apply engineered features (must match training)
    df_fe = add_engineered_features(df_input)

    # Predict probability (original flavor)
    prob = float(model.predict_proba(df_fe)[0][1])
    pred = int(prob >= threshold)

    return {
        "probability": prob,
        "threshold": threshold,
        "prediction": pred,
        "label": "Machine Failed" if pred == 1 else "No Failure"
    }

# -----------------------------
# SELECT THE LATEST MODEL
# -----------------------------
def select_latest_model(model_name: str):
    """Select latest model run for given model_name (RF/LGBM)."""
    models = list_available_models()
    filtered = [m for m in models if m["model_name"] == model_name]

    if not filtered:
        raise ValueError(f"No {model_name} models found")

    # Pick latest by start_time
    filtered.sort(key=lambda x: x["start_time"], reverse=True)
    best_meta = filtered[0]

    # Cache the model to avoid loading it multiple times
    if model_name not in model_cache:
        model = load_model(best_meta["run_id"], best_meta["model_name"])
        model_cache[model_name] = model
    else:
        model = model_cache[model_name]

    threshold = load_threshold(best_meta["run_id"])

    return best_meta, model, threshold

# Test if this script runs alone
if __name__ == "__main__":
    # Example: get latest LGBM and RF
    lgb_meta, lgb_model, lgb_threshold = select_latest_model("LGBM_class_weighted")
    rf_meta, rf_model, rf_threshold = select_latest_model("RF_baseline")

    print("Latest LGBM:", lgb_meta, "Threshold:", lgb_threshold)
    print("Latest RF:", rf_meta, "Threshold:", rf_threshold)
