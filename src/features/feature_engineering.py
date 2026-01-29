import pandas as pd
import numpy as np
import os
import yaml
from src.utils.logger import logger

def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven engineered features.
    """
    X_fe = X.copy()

    required_cols = {'rpm', 'torque', 'air_temp', 'process_temp'}
    missing = required_cols - set(X_fe.columns)

    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    # 1. Power: Mechanical work metric
    X_fe["power"] = X_fe["rpm"] * X_fe["torque"]
    
    # 2. Temp Diff: Thermal stress metric
    X_fe["temp_diff"] = X_fe["process_temp"] - X_fe["air_temp"]
    
    # 3. Torque per RPM: Efficiency/load metric
    X_fe["torque_per_rpm"] = X_fe["torque"] / (X_fe["rpm"] + 1e-6)

    return X_fe

if __name__ == "__main__":
    try:
        # Load parameters
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        # Use the paths from your params.yaml
        input_dir = params['preprocess']['output_dir']
        output_dir = params['feature_eng']['output_dir']
        target_col = params['base']['target_col']

        # Ensure the new directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        for split_file in ['train.csv', 'test.csv']:
            path = os.path.join(input_dir, split_file)
            
            if os.path.exists(path):
                logger.info(f"Applying feature engineering to {split_file}")
                df = pd.read_csv(path)
                
                # Separate target
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Transform
                X_engineered = add_engineered_features(X)
                
                # Recombine
                df_final = pd.concat([X_engineered, y], axis=1)
                
                # Save to the feature_engineered directory with a clear name
                save_name = split_file.replace(".csv", "_enriched.csv")
                save_path = os.path.join(output_dir, save_name)
                df_final.to_csv(save_path, index=False)
                
        logger.info(f"Feature engineering complete. Files saved in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise e