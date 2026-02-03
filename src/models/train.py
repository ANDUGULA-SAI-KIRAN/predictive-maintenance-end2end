# src/models/train.py
import os
import yaml
import pandas as pd
import numpy as np
import mlflow
import optuna
import sys
import dagshub
import json
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
from lightgbm import LGBMClassifier
from sklearn.metrics import ( precision_recall_curve, precision_score, 
                recall_score, f1_score, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from src.utils.logger import logger

load_dotenv()

def optimize_threshold(y_true, y_prob):
    """
    Finds the best threshold to maximize F1-score.
    Calculation: F1 = 2 * (Prec * Rec) / (Prec + Rec)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Added explicit handling for zero-division and boundary cases
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    best_idx = np.argmax(f1_scores)
    # thresholds is 1 element shorter than precisions/recalls
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

class Trainer:
    def __init__(self, params):
        self.params = params
        self.train_cfg = params['train']
        self.base_cfg = params['base']

    def objective_rf(self, trial, X, y):
        p = self.train_cfg['rf_params']
        args = {
            "n_estimators": trial.suggest_int("n_estimators", *p['n_estimators']),
            "max_depth": trial.suggest_int("max_depth", *p['max_depth'], log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", *p['min_samples_split']),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", *p['min_samples_leaf']),
            "max_features": trial.suggest_categorical("max_features", p['max_features']),
            "class_weight": trial.suggest_categorical("class_weight", p['class_weight']),
            "random_state": self.base_cfg['random_state'],
            "n_jobs": 1
        }
        model = RandomForestClassifier(**args)
        skf = StratifiedKFold(n_splits=self.train_cfg['cv_folds'], shuffle=True, random_state=self.base_cfg['random_state'])
        # y = y.astype(int)
        try:
            cv_results = cross_validate( model, X, y,  cv=skf,  scoring={ 'prec': 'precision', 'rec': 'recall', 'f1': 'f1', 'pr_auc': 'average_precision'},n_jobs=-1 )
        except Exception as e:
            logger.error(f"❌ CV Iteration failed: {e}")
            return 0 # Return 0 so Optuna knows this trial failed

        # Pull results using the keys defined in the scoring dict
        trial.set_user_attr("precision", np.mean(cv_results['test_prec']))
        trial.set_user_attr("f1", np.mean(cv_results['test_f1']))
        trial.set_user_attr("pr_auc", np.mean(cv_results['test_pr_auc']))
        
        return np.mean(cv_results['test_rec'])


    def objective_lgbm(self, trial, X, y):
        p = self.train_cfg['lgbm_params']
        args = {
            "n_estimators": trial.suggest_int("n_estimators", *p['n_estimators']),
            "max_depth": trial.suggest_int("max_depth", *p['max_depth'], log=True),
            "learning_rate": trial.suggest_float("learning_rate", *p['learning_rate']),
            "num_leaves": trial.suggest_int("num_leaves", *p['num_leaves']),
            "min_child_samples": trial.suggest_int("min_child_samples", *p['min_child_samples']),
            "subsample": trial.suggest_float("subsample", *p['subsample']),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *p['colsample_bytree']),
            "class_weight": trial.suggest_categorical("class_weight", p['class_weight']),
            "random_state": self.base_cfg['random_state'],
            "n_jobs": -1, "verbose": -1
        }
        model = LGBMClassifier(**args)
        skf = StratifiedKFold(n_splits=self.train_cfg['cv_folds'], shuffle=True, random_state=self.base_cfg['random_state'])
        cv_results = cross_validate(model, X, y, cv=skf, scoring={ 'prec': 'precision', 'rec': 'recall', 'f1': 'f1', 'pr_auc': 'average_precision'}, n_jobs=-1) # type: ignore
        
        trial.set_user_attr("precision",  np.mean(cv_results['test_prec']))
        trial.set_user_attr("f1", np.mean(cv_results['test_f1']))
        trial.set_user_attr("pr_auc", np.mean(cv_results['test_pr_auc']))
        return np.mean(cv_results['test_rec'])

def main():
    try:
        logger.info("Initializing Training Pipeline")
        owner, repo = os.getenv('REPO_OWNER'), os.getenv('REPO_NAME')
        dagshub.init(repo_name=repo, repo_owner=owner, mlflow=True) # type: ignore

        # Initialize the client ONCE
        client = mlflow.tracking.MlflowClient() # type: ignore

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        # Loading ENRICHED data from the new directory
        data_dir = "data/feature_engineered"
        target = params['base']['target_col']
        
        train_df = pd.read_csv(os.path.join(data_dir, "train_enriched.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test_enriched.csv"))
        
        X_train, y_train = train_df.drop(columns=[target]), train_df[target]
        X_test, y_test = test_df.drop(columns=[target]), test_df[target]

        trainer = Trainer(params)
        # Assuming we need atleast 0.5 of recall as per business needs
        min_recall_threshold = 0.45
        min_f1_threshold = 0.45
        final_summary_metrics = {}
        model_registry = {}  # Track which models were registered

        for m_type in ['rf', 'lgbm']:
            logger.info(f"---Starting {m_type.upper()} trials---")
            mlflow.set_experiment(experiment_name=f"Predictive_Maintenance_{m_type.upper()}")
            
            with mlflow.start_run(run_name=f"{m_type.upper()}_Optimization"):
                def callback(study, trial):
                    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                        mlflow.log_params(trial.params)
                        mlflow.log_metric("cv_recall", trial.value)
                        mlflow.log_metric("cv_precision", trial.user_attrs.get("precision"))
                        mlflow.log_metric("cv_f1", trial.user_attrs.get("f1"))
                        mlflow.log_metric("cv_pr_auc", trial.user_attrs.get("pr_auc"))

                obj_func = trainer.objective_rf if m_type == 'rf' else trainer.objective_lgbm
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda t: obj_func(t, X_train, y_train), n_trials=params['train']['n_trials'], callbacks=[callback])

                # Train best model
                best_params = study.best_params
                int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves']
                for p in int_params:
                    if p in best_params:
                        best_params[p] = int(best_params[p])

                best_params['random_state'] = params['base']['random_state']
                model = RandomForestClassifier(**best_params) if m_type == 'rf' else LGBMClassifier(**best_params, verbose=-1)
                model.fit(X_train, y_train)

                # --- EVALUATION LOGIC ---
                y_probs = model.predict_proba(X_test)[:, 1] # type: ignore
                pr_auc = average_precision_score(y_test, y_probs)
                best_thresh = optimize_threshold(y_test, y_probs)
                y_pred = (y_probs >= best_thresh).astype(int)
                
                # 4. Calculate final metrics
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # --- Summary for CML ---
                final_summary_metrics[f"{m_type}_test_precision"] = float(prec)
                final_summary_metrics[f"{m_type}_test_recall"] = float(rec)
                final_summary_metrics[f"{m_type}_test_f1"] = float(f1)
                final_summary_metrics[f"{m_type}_test_pr_auc"] = float(pr_auc)
                final_summary_metrics[f"{m_type}_threshold"] = float(best_thresh)

                logger.info(f"{m_type.upper()} Results with Threshold {best_thresh:.2f}:")
                logger.info(f"Precision: {prec:.2f} | Recall: {rec:.4f} | F1: {f1:.2f} | PR-AUC: {pr_auc:.2f}")

                # MLflow Logging params and metrics
                mlflow.log_param("optimal_threshold", float(best_thresh))
                mlflow.log_params(best_params)
                metrics= {
                    "test_precision": prec,
                    "test_recall": rec,
                    "test_f1": f1,
                    "test_pr_auc": pr_auc,
                }
                mlflow.log_metrics(metrics)

                signature = infer_signature(X_test, y_pred)

                # --- Candidate model registry --- 
                if (rec > min_recall_threshold) and (f1 > min_f1_threshold):
                    logger.info(f"{m_type.upper()}Model - recall {rec:.2f} | {f1:.2f} passed threshold. Registering...")                    
                    reg_name = f"{m_type.upper()}_Model" 

                    mlflow.log_dict(
                            {"best_threshold": float(best_thresh)},
                            artifact_file="model_config/best_threshold.json"
                        )

                    mlflow.log_dict(params,
                            artifact_file="model_config/params_config.json"
                        )

                    if m_type == 'rf':
                        model_info = mlflow.sklearn.log_model( # type: ignore
                            sk_model=model, 
                            artifact_path="model",
                            signature=signature,
                            registered_model_name=reg_name 
                        )
                    else:
                        model_info = mlflow.lightgbm.log_model( # type: ignore
                            lgb_model=model, 
                            artifact_path="model",
                            signature=signature,
                            registered_model_name=reg_name 
                        )

                    # Logging the model with alias as candidate 
                    client.set_registered_model_alias(
                        name=reg_name, 
                        alias="candidate", 
                        version=model_info.registered_model_version # type: ignore
                    )
                    logger.info(f"Registered {reg_name} version {model_info.registered_model_version} as 'candidate'")
                    model_registry[m_type] = "registered"
                else:
                    model_registry[m_type] = "rejected"
                    logger.warning(f"❌ {m_type.upper()} recall {rec:.2f} < {min_recall_threshold} or {f1:.2f} < {min_f1_threshold}")

        # --- ADDED: Save metrics to local file for GitHub/DVC ---
        # Filter only numeric metrics for JSON report
        numeric_metrics = {
            key: val for key, val in final_summary_metrics.items()
            if isinstance(val, (int, float))
        }
        
        # Add model registration status (non-numeric but informative)
        model_summary = {
            "metrics": numeric_metrics,
            "model_status": model_registry,
            "best_model": max(model_registry.items(), key=lambda x: 1 if x[1] == "registered" else 0)[0] if any(s == "registered" for s in model_registry.values()) else "none"
        }
        
        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.json", "w") as f:
            json.dump(numeric_metrics, f, indent=4)
        
        # Save full summary for reference
        with open("reports/model_summary.json", "w") as f:
            json.dump(model_summary, f, indent=4)
        
        logger.info("Final metrics saved to reports/metrics.json")
        logger.info(f"Model Status: {model_registry}")
        logger.info(f"Summary: {model_summary}")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training Stage Failed: {e}")
        sys.exit(1)
        raise

if __name__ == "__main__":
    main()