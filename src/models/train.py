# src/models/train.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import optuna
from utils.logger import logger

from src.data.preprocess import load_data, preprocess_base_features, train_test_split_data
from src.features.feature_engineering import add_engineered_features

import matplotlib
matplotlib.use("Agg")


# MLflow and experiment config
MLFLOW_EXPERIMENT_NAME_RF = "predictive_maintenance_rf"
MLFLOW_EXPERIMENT_NAME_LGBM = "predictive_maintenance_lgbm"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

RANDOM_STATE = 42
DATA_PATH = "test.csv"
N_TRIALS = 50
CV_FOLDS = 5


# Utility functions
def optimize_threshold(y_true, y_prob):
    """
    Compute the threshold that maximizes F1 score (or any other metric you want).
    """
    try:
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
    except Exception as e:
        logger.error(f"Error in optimize_threshold: {e}")
        raise


# Optuna objective functions
def objective_rf(trial, X_train, y_train):
    """Optuna objective for Random Forest with StratifiedKFold CV"""
    try:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
        
        model = RandomForestClassifier(**params)
        
        # StratifiedKFold for imbalanced data
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall', n_jobs=-1)
        
        return cv_scores.mean()
    except Exception as e:
        logger.error(f"Error in RF objective: {e}")
        raise


def objective_lgbm(trial, X_train, y_train):
    """Optuna objective for LightGBM with StratifiedKFold CV"""
    try:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1
        }
        
        model = LGBMClassifier(**params)
        
        # StratifiedKFold for imbalanced data
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall', n_jobs=-1) # pyright: ignore[reportArgumentType]
        
        return cv_scores.mean()
    except Exception as e:
        logger.error(f"Error in LGBM objective: {e}")
        raise


# Training functions
def train_random_forest(X_train, y_train, X_test, y_test):
    logger.info("Starting Random Forest hyperparameter optimization...")
    
    try:
        # Optuna study
        study = optuna.create_study(direction='maximize', study_name='RF_optuna')
        study.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)
        
        best_params = study.best_params
        logger.info(f"Best RF params: {best_params}")
        logger.info(f"Best CV recall: {study.best_value:.4f}")
        
        # Train final model with best params
        best_params['random_state'] = RANDOM_STATE
        best_params['n_jobs'] = -1
        rf_model = RandomForestClassifier(**best_params)
        
        with mlflow.start_run(run_name="RF_optuna_best"):
            mlflow.sklearn.autolog()
            
            rf_model.fit(X_train, y_train)
            
            y_prob = rf_model.predict_proba(X_test)[:, 1]
            opt_thresh, metrics_df = optimize_threshold(y_test, y_prob)
            
            recall_at_opt_thresh = recall_score(y_test, (y_prob > opt_thresh).astype(int))
            
            # Log custom metrics
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_recall_mean", study.best_value)
            mlflow.log_metric("optimal_threshold", opt_thresh)
            mlflow.log_metric("recall_at_opt_thresh", recall_at_opt_thresh)
            mlflow.log_metric("pr_auc", auc(metrics_df["recall"], metrics_df["precision"]))
            
            # Log model
            mlflow.sklearn.log_model(rf_model, "RF_model")
        
        logger.info(f"Random Forest training complete. Optimal threshold={opt_thresh:.4f}")
        return rf_model, opt_thresh
        
    except Exception as e:
        logger.error(f"Error in train_random_forest: {e}")
        raise


def train_lightgbm(X_train, y_train, X_test, y_test):
    logger.info("Starting LightGBM hyperparameter optimization...")
    
    try:
        # Optuna study
        study = optuna.create_study(direction='maximize', study_name='LGBM_optuna')
        study.optimize(lambda trial: objective_lgbm(trial, X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)
        
        best_params = study.best_params
        logger.info(f"Best LGBM params: {best_params}")
        logger.info(f"Best CV recall: {study.best_value:.4f}")
        
        # Train final model with best params
        best_params['random_state'] = RANDOM_STATE
        best_params['n_jobs'] = -1
        best_params['verbose'] = -1
        lgb_model = LGBMClassifier(**best_params)
        
        with mlflow.start_run(run_name="LGBM_optuna_best"):
            mlflow.lightgbm.autolog()
            
            lgb_model.fit(X_train, y_train)
            
            y_prob = lgb_model.predict_proba(X_test)[:, 1]
            opt_thresh, metrics_df = optimize_threshold(y_test, y_prob)
            
            recall_at_opt_thresh = recall_score(y_test, (y_prob > opt_thresh).astype(int))
            
            # Log custom metrics
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_recall_mean", study.best_value)
            mlflow.log_metric("optimal_threshold", opt_thresh)
            mlflow.log_metric("recall_at_opt_thresh", recall_at_opt_thresh)
            mlflow.log_metric("pr_auc", auc(metrics_df["recall"], metrics_df["precision"]))
            
            # Log model
            mlflow.lightgbm.log_model(lgb_model, "LGBM_model")
        
        logger.info(f"LightGBM training complete. Optimal threshold={opt_thresh:.4f}")
        return lgb_model, opt_thresh
        
    except Exception as e:
        logger.error(f"Error in train_lightgbm: {e}")
        raise


# Main function
def main():
    try:
        # Configure MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load & preprocess data
        logger.info(f"Loading data from {DATA_PATH}")
        df = load_data(DATA_PATH)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        
        df = preprocess_base_features(df)
        
        X_train, X_test, y_train, y_test = train_test_split_data(df)
        X_train_fe = add_engineered_features(X_train)
        X_test_fe = add_engineered_features(X_test)
        
        # Train Random Forest with separate experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_RF)
        rf_model, rf_threshold = train_random_forest(X_train_fe, y_train, X_test_fe, y_test)
        
        # Train LightGBM with separate experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_LGBM)
        lgb_model, lgb_threshold = train_lightgbm(X_train_fe, y_train, X_test_fe, y_test)
        
        logger.info("\nTraining complete. Models saved in MLflow SQLite DB.")
        logger.info(f"Random Forest threshold: {rf_threshold:.4f}")
        logger.info(f"LightGBM threshold: {lgb_threshold:.4f}")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()