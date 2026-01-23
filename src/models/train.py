import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, recall_score
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import mlflow.lightgbm

from src.data.preprocess import load_data, preprocess_base_features, train_test_split_data
from src.features.feature_engineering import add_engineered_features

import matplotlib
matplotlib.use("Agg")


# MLflow and experiment config
MLFLOW_EXPERIMENT_NAME = "predictive_maintenance"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Persistent SQLite backend

RANDOM_STATE = 42
# DATA_PATH = "C:/sai files/projects/predictive-maintenance-end2end/test.csv"
DATA_PATH = "test.csv"

# Utility functions
def optimize_threshold(y_true, y_prob):
    """
    Compute the threshold that maximizes F1 score (or any other metric you want).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    metrics_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precisions[:-1],
        "recall": recalls[:-1],
        "f1": f1_scores[:-1]
    })
    return optimal_threshold, metrics_df


# Training functions
def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    with mlflow.start_run(run_name="RF_baseline"):
        mlflow.sklearn.autolog()  # Enable sklearn autologging

        rf_model.fit(X_train, y_train)

        y_prob = rf_model.predict_proba(X_test)[:, 1]
        opt_thresh, metrics_df = optimize_threshold(y_test, y_prob)

        recall_at_opt_thresh = recall_score(y_test, (y_prob > opt_thresh).astype(int))

        # Log custom metrics
        mlflow.log_metric("optimal_threshold", opt_thresh)
        mlflow.log_metric("recall_at_opt_thresh", recall_at_opt_thresh)
        mlflow.log_metric("pr_auc", auc(metrics_df["recall"], metrics_df["precision"]))

        # Log model explicitly
        mlflow.sklearn.log_model(rf_model, "RF_model")

    print(f"Random Forest training complete. Optimal threshold={opt_thresh:.4f}")
    return rf_model, opt_thresh

def train_lightgbm(X_train, y_train, X_test, y_test):
    print("Training LightGBM...")
    lgb_model = LGBMClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)

    with mlflow.start_run(run_name="LGBM_class_weighted"):
        mlflow.lightgbm.autolog()  # Enable LightGBM autologging

        lgb_model.fit(X_train, y_train)

        y_prob = lgb_model.predict_proba(X_test)[:, 1]
        opt_thresh, metrics_df = optimize_threshold(y_test, y_prob)

        recall_at_opt_thresh = recall_score(y_test, (y_prob > opt_thresh).astype(int))

        # Log custom metrics
        mlflow.log_metric("optimal_threshold", opt_thresh)
        mlflow.log_metric("recall_at_opt_thresh", recall_at_opt_thresh)
        mlflow.log_metric("pr_auc", auc(metrics_df["recall"], metrics_df["precision"]))

        # Log model explicitly
        mlflow.lightgbm.log_model(lgb_model, "LGBM_model")

    print(f"LightGBM training complete. Optimal threshold={opt_thresh:.4f}")
    return lgb_model, opt_thresh


# Main function
def main():
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Connect to SQLite tracking URI
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)  # Set experiment name

    # Load & preprocess data
    df = load_data(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    df = preprocess_base_features(df)

    X_train, X_test, y_train, y_test = train_test_split_data(df)
    X_train_fe = add_engineered_features(X_train)
    X_test_fe = add_engineered_features(X_test)

    # Train both models
    rf_model, rf_threshold = train_random_forest(X_train_fe, y_train, X_test_fe, y_test)
    lgb_model, lgb_threshold = train_lightgbm(X_train_fe, y_train, X_test_fe, y_test)

    print("\nTraining complete. Models saved in MLflow SQLite DB.")
    print(f"Random Forest threshold: {rf_threshold:.4f}")
    print(f"LightGBM threshold: {lgb_threshold:.4f}")

if __name__ == "__main__":
    main()
