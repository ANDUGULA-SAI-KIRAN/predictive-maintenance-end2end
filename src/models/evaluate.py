# src/models/evaluate.py
import os
import yaml
import json
import pandas as pd
import mlflow
import dagshub
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

# Importing your existing project logger
from src.utils.logger import logger

def evaluate():
    try:
        # 1. Load Parameters
        logger.info("Loading evaluation parameters from params.yaml")
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        
        config = params['evaluate']
        target_col = params['base']['target_col']
        
        # 2. Setup DagsHub/MLflow
        logger.info("Initializing MLflow tracking via DagsHub")
        owner, repo = os.getenv('REPO_OWNER'), os.getenv('REPO_NAME')
        dagshub.init(repo_name=repo, repo_owner=owner, mlflow=True) # type: ignore

        # 3. Load Test Data
        logger.info(f"Reading test data from {config['test_data']}")
        try:
            test_df = pd.read_csv(config['test_data'])
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            # Prepare evaluation dataframe for mlflow.evaluate()
            eval_df = X_test.copy()
            eval_df['label'] = y_test
            
        except FileNotFoundError as e:
            logger.error(f"Test data file missing at {config['test_data']}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            raise 

        # Defining thresholds for Automated Registration
        RECALL_THRESHOLD = 0.50
        F1_THRESHOLD = 0.45
        final_metrics = {}

        # 4. Evaluate Models (RF and LGBM)
        for model_name in ['rf', 'lgbm']:
            # We assume the models were saved as .pkl in the 'models/' directory during train.py
            model_path = os.path.join("models", f"{model_name}_model.pkl")
            logger.info(f"Starting evaluation for {model_name.upper()} using {model_path}")
            
            try:
                if not os.path.exists(model_path):
                    logger.warning(f"Model file not found: {model_path}. Skipping {model_name}.")
                    continue

                # Load the local model artifact
                model = joblib.load(model_path)
                
                with mlflow.start_run(run_name=f"Final_Eval_{model_name.upper()}"):
                    model_info = mlflow.sklearn.log_model(sk_model=model, name=f"{model_name}_model") # type: ignore
                    model_uri = model_info.model_uri

                    result = mlflow.models.evaluate( # type: ignore
                        model=model_uri,
                        data=eval_df,
                        targets=target_col,
                        model_type="classifier",
                        evaluator_config={"log_model_explainability": False }
                    )
                    
                    # Extract metrics AFTER evaluation
                    m_recall = float(result.metrics["recall_score"])
                    m_f1 = float(result.metrics["f1_score"])
                    m_precision = float(result.metrics["precision_score"])

                    # Generate Confusion Matrix Plot for CML
                    y_pred = model.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(cmap=plt.cm.Blues) # type: ignore
                    plt.title(f"Confusion Matrix: {model_name.upper()}")
                    
                    # Save plot locally for GitHub Action to find
                    plot_path = f"reports/{model_name}_cm.png"
                    plt.savefig(plot_path)
                    plt.close()
                    
                    # Log plot to MLflow
                    mlflow.log_artifact(plot_path)

                    # Threshold check: Automated Registration Logic
                    if m_recall >= RECALL_THRESHOLD and m_f1 >= F1_THRESHOLD:
                        logger.info(f"🏆 {model_name} passed thresholds. Registering model...")
                        mlflow.register_model(
                            model_uri=f"runs:/{mlflow.active_run().info.run_id}/{model_name}_model", # type: ignore
                            name=f"AI4I_{model_name.upper()}_Production"
                        )
                        mlflow.set_tag("model_status", "candidate")
                    else:
                        logger.warning(f"⚠️ {model_name} failed thresholds (Recall: {m_recall}). Not registered.")

                    # Store key metrics for the local JSON report
                    final_metrics[model_name] = {
                        "recall": m_recall,
                        "f1_score": m_f1,
                        "precision": m_precision
                        }
                    
                logger.info(f"Successfully evaluated {model_name.upper()}.")

            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
                continue 
        # Save Results
        with open(config['metrics_file'], "w") as f:
            json.dump(final_metrics, f, indent=4)

    except Exception as e:
        logger.critical(f"Critical failure in evaluation pipeline: {e}")
        raise

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    evaluate()