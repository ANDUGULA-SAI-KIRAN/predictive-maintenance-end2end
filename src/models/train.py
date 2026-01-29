# # # src/models/train.py
# import os
# import yaml
# import pandas as pd
# import numpy as np
# import mlflow
# import optuna
# from dotenv import load_dotenv
# from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import precision_recall_curve, auc, recall_score
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# import dagshub
# from src.utils.logger import logger
# from src.features.feature_engineering import add_engineered_features

# load_dotenv()


# def optimize_threshold(y_true, y_prob):
#     precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
#     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
#     best_idx = np.argmax(f1_scores)
#     optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
#     metrics_df = pd.DataFrame({
#         "threshold": thresholds, "precision": precisions[:-1],
#         "recall": recalls[:-1], "f1": f1_scores[:-1]
#     })
#     return optimal_threshold, metrics_df


# class Trainer:
#     def __init__(self, params):
#         self.params = params
#         self.train_cfg = params['train']
#         self.base_cfg = params['base']

#     def objective_rf(self, trial, X, y):
#         p = self.train_cfg['rf_params']
#         args = {
#             "n_estimators": trial.suggest_int("n_estimators", *p['n_estimators']),
#             "max_depth": trial.suggest_int("max_depth", *p['max_depth'], log=True),
#             "min_samples_split": trial.suggest_int("min_samples_split", *p['min_samples_split']),
#             "min_samples_leaf": trial.suggest_int("min_samples_leaf", *p['min_samples_leaf']),
#             "max_features": trial.suggest_categorical("max_features", p['max_features']),
#             "class_weight": trial.suggest_categorical("class_weight", p['class_weight']),
#             "random_state": self.base_cfg['random_state'],
#             "n_jobs": -1
#         }
#         model = RandomForestClassifier(**args)
#         skf = StratifiedKFold(n_splits=self.train_cfg['cv_folds'], shuffle=True, random_state=self.base_cfg['random_state'])
#         return cross_val_score(model, X, y, cv=skf, scoring='recall', n_jobs=-1).mean()

#     def objective_lgbm(self, trial, X, y):
#         p = self.train_cfg['lgbm_params']
#         args = {
#             "n_estimators": trial.suggest_int("n_estimators", *p['n_estimators']),
#             "max_depth": trial.suggest_int("max_depth", *p['max_depth'], log=True),
#             "learning_rate": trial.suggest_float("learning_rate", *p['learning_rate']),
#             "num_leaves": trial.suggest_int("num_leaves", *p['num_leaves']),
#             "min_child_samples": trial.suggest_int("min_child_samples", *p['min_child_samples']),
#             "subsample": trial.suggest_float("subsample", *p['subsample']),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", *p['colsample_bytree']),
#             "class_weight": trial.suggest_categorical("class_weight", p['class_weight']),
#             "random_state": self.base_cfg['random_state'],
#             "n_jobs": -1, "verbose": -1
#         }
#         model = LGBMClassifier(**args)
#         skf = StratifiedKFold(n_splits=self.train_cfg['cv_folds'], shuffle=True, random_state=self.base_cfg['random_state'])
#         return cross_val_score(model, X, y, cv=skf, scoring='recall', n_jobs=-1).mean()

# def main():

#     # 2. Extract Repo info from .env
#     owner = os.getenv('REPO_OWNER')
#     repo = os.getenv('REPO_NAME')
#     token = os.getenv('DAGSHUB_USER_TOKEN')

#     if not all([owner, repo, token]):
#         raise ValueError("Missing REPO_OWNER, REPO_NAME, or DAGSHUB_USER_TOKEN in .env")

#     # 3. THE STANDARD APPROACH: Initialize DagsHub
#     # This automatically sets mlflow.set_tracking_uri and auth
#     dagshub.init(repo_name=repo, repo_owner=owner, mlflow=True)
    
#     # Optional: For double-checking in logs
#     print(f"Tracking to: {mlflow.get_tracking_uri()}")

#     with open("params.yaml", "r") as f:
#         params = yaml.safe_load(f)

#     # DagsHub MLflow Setup
#     remote_url = f"https://dagshub.com/{os.getenv('REPO_OWNER')}/{os.getenv('REPO_NAME')}.mlflow"
#     mlflow.set_tracking_uri(remote_url)
    
#     trainer = Trainer(params)
    
#     # Load Data
#     train_df = pd.read_csv(os.path.join(params['preprocess']['output_dir'], "train.csv"))
#     test_df = pd.read_csv(os.path.join(params['preprocess']['output_dir'], "test.csv"))
    
#     X_train = add_engineered_features(train_df.drop(columns=[params['base']['target_col']]))
#     y_train = train_df[params['base']['target_col']]
#     X_test = add_engineered_features(test_df.drop(columns=[params['base']['target_col']]))
#     y_test = test_df[params['base']['target_col']]

#     for m_type in ['rf', 'lgbm']:
#         mlflow.set_experiment(params['train']['experiments'][m_type])
        
#         with mlflow.start_run(run_name=f"{m_type.upper()}_Parent_Optimization"):
#             # Nested logging callback
#             def callback(study, trial):
#                 with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
#                     mlflow.log_params(trial.params)
#                     mlflow.log_metric("recall", trial.value)

#             obj_func = trainer.objective_rf if m_type == 'rf' else trainer.objective_lgbm
#             study = optuna.create_study(direction='maximize')
#             study.optimize(lambda t: obj_func(t, X_train, y_train), 
#                            n_trials=params['train']['n_trials'], 
#                            callbacks=[callback])

#             # Final Model Training with Best Params
#             best_params = study.best_params
#             best_params['random_state'] = params['base']['random_state']
            
#             if m_type == 'rf':
#                 model = RandomForestClassifier(**best_params)
#             else:
#                 best_params['verbose'] = -1
#                 model = LGBMClassifier(**best_params)

#             model.fit(X_train, y_train)
#             y_prob = model.predict_proba(X_test)[:, 1]
#             opt_thresh, _ = optimize_threshold(y_test, y_prob)
            
#             # Log Final Results to Parent
#             mlflow.log_params(best_params)
#             mlflow.log_metric("best_cv_recall", study.best_value)
#             mlflow.log_metric("opt_threshold", opt_thresh)
            
#             if m_type == 'rf': mlflow.sklearn.log_model(model, "rf_model")
#             else: mlflow.lightgbm.log_model(model, "lgbm_model")
            
#             logger.info(f"Finished {m_type} optimization.")

# if __name__ == "__main__":
#     main()


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
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.utils.logger import logger

load_dotenv()

def optimize_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_idx = np.argmax(f1_scores)
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
            "n_jobs": -1
        }
        model = RandomForestClassifier(**args)
        skf = StratifiedKFold(n_splits=self.train_cfg['cv_folds'], shuffle=True, random_state=self.base_cfg['random_state'])
        return cross_val_score(model, X, y, cv=skf, scoring='recall', n_jobs=-1).mean()

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
        return cross_val_score(model, X, y, cv=skf, scoring='recall', n_jobs=-1).mean()

def main():
    try:
        logger.info("Initializing Training Pipeline")
        owner, repo = os.getenv('REPO_OWNER'), os.getenv('REPO_NAME')
        dagshub.init(repo_name=repo, repo_owner=owner, mlflow=True)

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

        for m_type in ['rf', 'lgbm']:
            logger.info(f"Starting {m_type.upper()} trials")
            mlflow.set_experiment(params['train']['experiments'][m_type])
            
            with mlflow.start_run(run_name=f"{m_type.upper()}_Optimization"):
                def callback(study, trial):
                    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                        mlflow.log_params(trial.params)
                        mlflow.log_metric("recall", trial.value)

                obj_func = trainer.objective_rf if m_type == 'rf' else trainer.objective_lgbm
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda t: obj_func(t, X_train, y_train), n_trials=params['train']['n_trials'], callbacks=[callback])

                # Train best model
                best_params = study.best_params
                best_params['random_state'] = params['base']['random_state']
                model = RandomForestClassifier(**best_params) if m_type == 'rf' else LGBMClassifier(**best_params, verbose=-1)
                
                model.fit(X_train, y_train)
                
                # Artifact Saving
                os.makedirs("models", exist_ok=True)
                joblib.dump(model, f"models/{m_type}_model.pkl")
                
                # MLflow Logging
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_recall", study.best_value)
                
                if m_type == 'rf': mlflow.sklearn.log_model(model, "rf_model")
                else: mlflow.lightgbm.log_model(model, "lgbm_model")
                
                logger.info(f"Finished {m_type} optimization and saved local artifact.")

    except Exception as e:
        logger.error(f"Training Stage Failed: {e}")
        raise

if __name__ == "__main__":
    main()