# src/models/train.py

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import optuna

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

from src.data.preprocess import load_data, preprocess_base_features, train_test_split_data
from src.features.feature_engineering import add_engineered_features

# --------------------------
# Configuration
# --------------------------
DATA_PATH = "C:/sai files/projects/predictive-maintenance-end2end/test.csv"
MLFLOW_EXPERIMENT_NAME = "predictive_maintenance"
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 20

# --------------------------
# Threshold Optimization
# --------------------------
def optimize_threshold(y_true, y_prob, thresholds=np.arange(0.0, 1.01, 0.01)):
    """
    Compute F1 score across thresholds and return the optimal threshold.
    """
    metrics = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        metrics.append([t, precision, recall, f1])
    
    metrics_df = pd.DataFrame(metrics, columns=['threshold', 'precision', 'recall', 'f1'])
    optimal_idx = metrics_df['f1'].idxmax()
    optimal_threshold = metrics_df.loc[optimal_idx, 'threshold']
    
    return optimal_threshold, metrics_df

# --------------------------
# Model Training Functions with Optuna
# --------------------------
def train_rf_optuna(X_train, y_train, X_test, y_test):
    """Train RF with hyperparameter tuning using Optuna."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": RANDOM_STATE,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        recall = recall_score(y_test, (y_prob >= 0.5).astype(int))
        return recall

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)
    best_params = study.best_params
    rf_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    opt_thresh, threshold_df = optimize_threshold(y_test, y_prob)
    return rf_model, best_params, opt_thresh, threshold_df

def train_lgb_optuna(X_train, y_train, X_test, y_test):
    """Train LGBM with hyperparameter tuning using Optuna."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        recall = recall_score(y_test, (y_prob >= 0.5).astype(int))
        return recall

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)
    best_params = study.best_params
    lgb_model = LGBMClassifier(**best_params, class_weight="balanced", random_state=RANDOM_STATE)
    lgb_model.fit(X_train, y_train)
    y_prob = lgb_model.predict_proba(X_test)[:, 1]
    opt_thresh, threshold_df = optimize_threshold(y_test, y_prob)
    return lgb_model, best_params, opt_thresh, threshold_df

# --------------------------
# Main Training Pipeline
# --------------------------
def main():
    # 1️⃣ Load and preprocess data
    df = load_data(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    df_processed = preprocess_base_features(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df_processed, random_state=RANDOM_STATE)

    # 2️⃣ Feature engineering
    X_train_fe = add_engineered_features(X_train)
    X_test_fe = add_engineered_features(X_test)

    # 3️⃣ MLflow setup
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 4️⃣ Train and log RF
    with mlflow.start_run(run_name="RF_baseline") as run:
        print("\nTraining Random Forest with Optuna...")
        mlflow.autolog()
        rf_model, rf_params, rf_thresh, rf_thresh_df = train_rf_optuna(X_train_fe, y_train, X_test_fe, y_test)
        mlflow.sklearn.log_model(rf_model, artifact_path="RF_baseline")
        mlflow.log_params(rf_params)
        mlflow.log_metrics({
            "optimal_threshold": rf_thresh,
            "precision": precision_score(y_test, (rf_model.predict_proba(X_test_fe)[:, 1] >= rf_thresh).astype(int)),
            "recall": recall_score(y_test, (rf_model.predict_proba(X_test_fe)[:, 1] >= rf_thresh).astype(int)),
            "f1": f1_score(y_test, (rf_model.predict_proba(X_test_fe)[:, 1] >= rf_thresh).astype(int)),
            "pr_auc": auc(*precision_recall_curve(y_test, rf_model.predict_proba(X_test_fe)[:, 1])[:2])
        })
        rf_thresh_df.to_csv("RF_threshold_metrics.csv", index=False)
        mlflow.log_artifact("RF_threshold_metrics.csv")
        mlflow.set_tag("author", "your_name")
        mlflow.set_tag("model_type", "RandomForest")
        print(f"RF done, optimal_threshold={rf_thresh:.2f}")

    # 5️⃣ Train and log LGBM
    with mlflow.start_run(run_name="LGBM_class_weighted") as run:
        print("\nTraining LGBM with Optuna...")
        mlflow.autolog()
        lgb_model, lgb_params, lgb_thresh, lgb_thresh_df = train_lgb_optuna(X_train_fe, y_train, X_test_fe, y_test)
        mlflow.lightgbm.log_model(lgb_model, artifact_path="LGBM_class_weighted")
        mlflow.log_params(lgb_params)
        mlflow.log_metrics({
            "optimal_threshold": lgb_thresh,
            "precision": precision_score(y_test, (lgb_model.predict_proba(X_test_fe)[:, 1] >= lgb_thresh).astype(int)),
            "recall": recall_score(y_test, (lgb_model.predict_proba(X_test_fe)[:, 1] >= lgb_thresh).astype(int)),
            "f1": f1_score(y_test, (lgb_model.predict_proba(X_test_fe)[:, 1] >= lgb_thresh).astype(int)),
            "pr_auc": auc(*precision_recall_curve(y_test, lgb_model.predict_proba(X_test_fe)[:, 1])[:2])
        })
        lgb_thresh_df.to_csv("LGBM_threshold_metrics.csv", index=False)
        mlflow.log_artifact("LGBM_threshold_metrics.csv")
        mlflow.set_tag("author", "your_name")
        mlflow.set_tag("model_type", "LGBM")
        print(f"LGBM done, optimal_threshold={lgb_thresh:.2f}")

    print("\n✅ Training pipeline completed. Models are stored under mlruns/")

# --------------------------
# Entry Point
# --------------------------
if __name__ == "__main__":
    main()
