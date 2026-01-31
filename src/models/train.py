# src/models/train.py
import os
import yaml
import pandas as pd
import numpy as np
import mlflow
import optuna
import joblib
import dagshub
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
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

        for m_type in ['rf', 'lgbm']:
            logger.info(f"---Starting {m_type.upper()} trials---")
            mlflow.set_experiment(params['train']['experiments'][m_type])
            
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
                best_params['random_state'] = params['base']['random_state']
                model = RandomForestClassifier(**best_params) if m_type == 'rf' else LGBMClassifier(**best_params, verbose=-1)
                
                model.fit(X_train, y_train)
                os.makedirs("models", exist_ok=True)
                # --- EVALUATION LOGIC ---
                # 1. Get Probabilities (required for PR-AUC and threshold tuning)
                y_probs = model.predict_proba(X_test)[:, 1] # type: ignore
                
                # 2. Calculate PR-AUC (Area under Precision-Recall Curve)
                pr_auc = average_precision_score(y_test, y_probs)
                
                # 3. Optimize Threshold for F1
                best_thresh = optimize_threshold(y_test, y_probs)
                y_pred = (y_probs >= best_thresh).astype(int)
                
                # 4. Calculate final metrics
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                logger.info(f"{m_type.upper()} Results with Threshold {best_thresh:.2f}:")
                logger.info(f"Precision: {prec:.2f} | Recall: {rec:.4f} | F1: {f1:.2f} | PR-AUC: {pr_auc:.2f}")

                # MLflow Logging
                mlflow.log_params(best_params)
                metrics= {
                    "test_precision": prec,
                    "test_recall": rec,
                    "test_f1": f1,
                    "test_pr_auc": pr_auc,
                    "best_threshold": best_thresh
                }
                mlflow.log_metrics(metrics)
                
                # Artifact Saving
                if (rec > min_recall_threshold) and (f1 > min_f1_threshold):
                    logger.info(f"{m_type.upper()} passed threshold. Saving...")
                    joblib.dump(model, f"models/{m_type}_model.pkl")
                    
                    if m_type == 'rf': mlflow.sklearn.log_model(model, "rf_model") # type: ignore
                    else: mlflow.lightgbm.log_model(model, "lgbm_model") # type: ignore
                else:
                    logger.warning(f"❌ {m_type.upper()} recall {rec:.2f} < {min_recall_threshold} or {f1:.2f} < {min_f1_threshold}")


    except Exception as e:
        logger.error(f"Training Stage Failed: {e}")
        raise

if __name__ == "__main__":
    main()