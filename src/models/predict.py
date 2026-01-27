# # src/models/predict.py

# from typing import Dict, Any
# import pandas as pd
# import mlflow
# from mlflow.tracking import MlflowClient
# import mlflow.sklearn
# import mlflow.lightgbm
# from src.features.feature_engineering import add_engineered_features

# # MLflow config (match train.py)
# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# MLFLOW_EXPERIMENT_NAME = "predictive_maintenance"

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# TYPE_MAPPING = {'low': 0, 'medium': 1, 'high': 2}

# def preprocess_type_column(df: pd.DataFrame) -> pd.DataFrame:
#     """Apply type feature encoding (convert to lowercase and map to numeric)."""
#     df['type'] = df['type'].str.lower()  # Convert to lowercase
#     df['type'] = df['type'].map(TYPE_MAPPING)  # Map categories to numeric values
#     return df


# # Model loading function
# def load_model(run_id: str, model_name: str):
#     """
#     Load model from MLflow.
#     """
#     if model_name.startswith("LGBM"):
#         artifact_name = "LGBM_model"
#         model_uri = f"runs:/{run_id}/{artifact_name}"
#         return mlflow.lightgbm.load_model(model_uri)
#     else:
#         artifact_name = "RF_model"
#         model_uri = f"runs:/{run_id}/{artifact_name}"
#         return mlflow.sklearn.load_model(model_uri)


# # Model selection logic (to get available models)
# def list_available_models() -> list:
#     """Fetch available models from MLflow."""
#     client = MlflowClient()
#     experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
#     if experiment is None:
#         return []

#     runs = client.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         order_by=["attributes.start_time DESC"]
#     )

#     models = []
#     for run in runs:
#         run_name = run.data.tags.get("mlflow.runName", "")
#         if run_name in {"RF_baseline", "LGBM_class_weighted"}:
#             models.append({
#                 "model_name": run_name,
#                 "run_id": run.info.run_id,
#                 "start_time": str(run.info.start_time)
#             })
#     return models


# # Threshold loading function
# def load_threshold(run_id: str) -> float:
#     """Load optimal threshold from MLflow run metrics."""
#     client = MlflowClient()
#     run = client.get_run(run_id)
#     threshold = run.data.metrics.get("optimal_threshold", 0.5)
#     return float(threshold)


# # Prediction function (with feature engineering)
# def predict(input_features: Dict[str, Any], model, threshold: float) -> Dict[str, Any]:
#     """
#     Perform prediction after applying feature engineering.
#     :param input_features: The input features from the API request.
#     :param model: The loaded ML model.
#     :param threshold: The optimal threshold for classification.
#     :return: A dictionary containing the prediction and probability.
#     """
#     # Convert API input dict to training column names
#     field_map = {
#         "torque": "torque",
#         "type": "type",
#         "air_temperature": "air_temp",
#         "process_temperature": "process_temp",
#         "rotational_speed": "rpm",
#         "tool_wear": "tool_wear"
#     }
#     df_input = pd.DataFrame([{field_map[k]: v for k, v in input_features.items()}])
    
#     # Preprocess type column (encoding)
#     df_input = preprocess_type_column(df_input)

#     # Apply feature engineering (match training features)
#     df_fe = add_engineered_features(df_input)

#     # Predict probability (original flavor)
#     prob = float(model.predict_proba(df_fe)[0][1])  # Assumes binary classification (0/1)
#     pred = int(prob >= threshold)

#     return {
#         "probability": prob,
#         "threshold": threshold,
#         "prediction": pred,
#         "label": "Machine Failed" if pred == 1 else "No Failure"
#     }



import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from src.features.feature_engineering import add_engineered_features

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Maps for consistency
EXP_MAP = {
    "Random Forest": "predictive_maintenance_rf",
    "LightGBM": "predictive_maintenance_lgbm"
}

TYPE_MAPPING = {'low': 0, 'medium': 1, 'high': 2}

class ModelCache:
    """Singleton-style cache to store the selected model and its metadata."""
    def __init__(self):
        self.model = None
        self.metadata = {}
        self.threshold = 0.5

cache = ModelCache()

def get_best_model_metadata(model_type: str):
    """
    Finds the run with the highest pr_auc in the relevant experiment.
    Returns metadata to display in Streamlit.
    """
    client = MlflowClient()
    exp_name = EXP_MAP[model_type]
    experiment = client.get_experiment_by_name(exp_name)
    
    if not experiment:
        return None

    # Search runs, order by pr_auc descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["metrics.pr_auc DESC"]
    )
    
    if not runs:
        return None

    best_run = runs[0]
    
    # Store in global cache for the API to use
    artifact_name = "RF_model" if model_type == "Random Forest" else "LGBM_model"
    model_uri = f"runs:/{best_run.info.run_id}/{artifact_name}"
    
    if model_type == "Random Forest":
        cache.model = mlflow.sklearn.load_model(model_uri)
    else:
        cache.model = mlflow.lightgbm.load_model(model_uri)
        
    cache.threshold = best_run.data.metrics.get("optimal_threshold", 0.5)
    
    # Metadata for UI
    cache.metadata = {
        "Run Name": best_run.data.tags.get("mlflow.runName", "N/A"),
        "PR-AUC": round(best_run.data.metrics.get("pr_auc", 0), 4),
        "Recall": round(best_run.data.metrics.get("recall_at_opt_thresh", 0), 4),
        "Optimal Threshold": round(cache.threshold, 4),
        "Best Params": best_run.data.params
    }
    
    return cache.metadata

def predict_with_confidence(input_dict: dict):
    """Calculates prediction and confidence score."""
    if cache.model is None:
        raise ValueError("No model selected or cached.")

    # Mapping UI fields to training field names
    field_map = {
        "type": "type",
        "air_temperature": "air_temp",
        "process_temperature": "process_temp",
        "rotational_speed": "rpm",
        "torque": "torque",
        "tool_wear": "tool_wear"
    }
    
    df_input = pd.DataFrame([{field_map[k]: v for k, v in input_dict.items()}])
    df_input['type'] = df_input['type'].str.lower().map(TYPE_MAPPING)
    
    # Apply same feature engineering as train.py
    df_fe = add_engineered_features(df_input)
    
    # Probability
    prob = float(cache.model.predict_proba(df_fe)[0][1])
    prediction = 1 if prob >= cache.threshold else 0
    
    # Confidence Score: how 'sure' the model is
    confidence = max(prob, 1 - prob) * 100
    
    return {
        "prediction": prediction,
        "probability": round(prob, 4),
        "confidence": round(confidence, 2),
        "label": "FAILURE DETECTED" if prediction == 1 else "Normal Operation",
        "processed_df": df_fe # Passing this for SHAP
    }