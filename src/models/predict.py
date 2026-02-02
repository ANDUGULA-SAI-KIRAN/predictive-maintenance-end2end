# src/models/predict.py
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import time
import os
import numpy as np
import dagshub
from src.features.feature_engineering import add_engineered_features
from dotenv import load_dotenv
load_dotenv()

# owner, repo = os.getenv('REPO_OWNER'), os.getenv('REPO_NAME')
# DAGSHUB_URI = f"https://dagshub.com/{owner}/{repo}.mlflow" 
# mlflow.set_tracking_uri(DAGSHUB_URI)

TYPE_MAPPING = {'low': 0, 'medium': 1, 'high': 2}

class ModelManager:
    """Handles professional @champion loading and caching."""
    def __init__(self):
        self.cache = {
            "Failure-Avoidance Mode": {"model": None, "threshold": 0.5, "meta": None, "last_updated": 0},
            "False-Alarm Reduction Mode": {"model": None, "threshold": 0.5, "meta": None, "last_updated": 0}
        }
        self.ttl = 6000  # Cache expires every 100 minutes

    def get_model(self, model_label: str):
        now = time.time()
        # 1. Check if we need to refresh cache
        if not self.cache[model_label]["model"] or (now - self.cache[model_label]["last_updated"] > self.ttl):
            self._refresh_cache(model_label)
        
        return self.cache[model_label]

    def _refresh_cache(self, model_label: str):
        """Fetches the @production model and its threshold artifact."""
        client = MlflowClient()
        # 1. Map UI Label to Registry Name
        reg_name = "LGBM_Model" if model_label == "Failure-Avoidance Mode" else "RF_Model"
        target_alias = "production"

        try:
            # 1. Connect to DagsHub
            owner, repo = os.getenv('REPO_OWNER'), os.getenv('REPO_NAME')
            dagshub.init(repo_name=repo, repo_owner=owner, mlflow=True) # type: ignore
            
            # 2. Load the Model Object
            model_uri = f"models:/{reg_name}@{target_alias}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # 3. Get the Run ID and version 
            model_version_details = client.get_model_version_by_alias(reg_name, target_alias)
            run_id = model_version_details.run_id
            version = model_version_details.version

            # 4. Download Threshold JSON
            local_path = client.download_artifacts(str(run_id), "model_config/best_threshold.json")
            import json
            with open(local_path, 'r') as f:
                threshold_data = json.load(f)
                threshold = threshold_data.get("best_threshold", 0.5)

            # 5. Update Cache
            self.cache[model_label] = {
                "model": model,
                "threshold": threshold,
                "meta": {
                    "Model Name": reg_name,
                    "Alias": target_alias,
                    "Version": version,
                    "Run ID": run_id,
                    "Threshold": round(threshold, 2)
                },
                "last_updated": time.time()
            }
            print(f"Successfully cached {reg_name}@{target_alias} (v{version})")
        except Exception as e:
            print(f"Critical Error loading @{target_alias} for {model_label}: {e}")

manager = ModelManager()

def predict_with_explanation(model_label: str, input_dict: dict):
    # 1. Get model and threshold from manager
    data = manager.get_model(model_label)
    model = data["model"]
    threshold = data["threshold"]

    # 2. Prep Data
    field_map = {"type": "type", "air_temperature": "air_temp", "process_temperature": "process_temp", 
                 "rotational_speed": "rpm", "torque": "torque", "tool_wear": "tool_wear"}
    
    # Create DataFrame with proper column names
    df_input = pd.DataFrame([{field_map[k]: v for k, v in input_dict.items() if k in field_map}])
    
    # Process 'type' and run feature engineering
    df_input['type'] = df_input['type'].str.lower().map(TYPE_MAPPING)
    df_fe = add_engineered_features(df_input)

    print(f"-------{df_fe}")

    # 3. Predict Probability
    # pyfunc.predict usually returns the probability if the flavor is LightGBM/XGBoost
    raw_res = model.predict(df_fe)
    print(f"_______{raw_res}")
    
    # Handle different return types (Single value vs Array)
    if isinstance(raw_res, (np.ndarray, list)):
        prob = float(raw_res[0])
    else:
        prob = float(raw_res)

    # 4. Apply Threshold Logic
    prediction = 1 if prob >= threshold else 0

    print(f"--{prediction}---")
    
    return {
        "prediction": prediction,
        "probability": round(prob, 2),
        "label": "FAILURE DETECTED" if prediction == 1 else "Normal Operation",
        "processed_df": df_fe,
        "meta": data["meta"]
    }