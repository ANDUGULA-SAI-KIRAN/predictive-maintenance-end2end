# src/features/feature_engineering.py
import pandas as pd
import numpy as np


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven engineered features.
    This function must be used identically during training and inference.
    """

    X_fe = X.copy()

    required_cols = {'rpm', 'torque', 'air_temp', 'process_temp'}
    missing = required_cols - set(X_fe.columns)

    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    X_fe["power"] = X_fe["rpm"] * X_fe["torque"]
    X_fe["temp_diff"] = X_fe["process_temp"] - X_fe["air_temp"]
    X_fe["torque_per_rpm"] = X_fe["torque"] / (X_fe["rpm"] + 1e-6)

    return X_fe
