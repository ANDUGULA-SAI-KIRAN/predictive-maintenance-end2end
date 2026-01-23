import shap
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os

MLFLOW_TRACKING_URI = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def compute_shap(model, X_sample: pd.DataFrame):
    """
    Compute SHAP values for a single input sample
    """
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    # Plot local SHAP values
    plt.figure()
    shap.plots.bar(shap_values)
    plt.tight_layout()
    return shap_values
