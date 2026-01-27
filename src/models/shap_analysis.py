# import shap
# import mlflow
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# MLFLOW_TRACKING_URI = os.path.join(os.getcwd(), "mlruns")
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# def compute_shap(model, X_sample: pd.DataFrame):
#     """
#     Compute SHAP values for a single input sample
#     """
#     explainer = shap.Explainer(model, X_sample)
#     shap_values = explainer(X_sample)

#     # Plot local SHAP values
#     plt.figure()
#     shap.plots.bar(shap_values)
#     plt.tight_layout()
#     return shap_values

##############################
# import shap
# import matplotlib.pyplot as plt
# import io
# import base64

# def get_shap_plots(model, input_df):
#     """
#     Generates Waterfall and Bar plots.
#     Returns them as base64 strings for the Streamlit UI.
#     """
#     # Create explainer
#     explainer = shap.Explainer(model)
#     shap_values = explainer(input_df)
    
#     plots = {}

#     # 1. Waterfall Plot (Local)
#     plt.figure(figsize=(10, 6))
#     shap.plots.waterfall(shap_values[0], show=False)
#     plots['waterfall'] = plt_to_base64(plt)
    
#     # 2. Bar Plot (Importance)
#     plt.figure(figsize=(10, 6))
#     shap.plots.bar(shap_values[0], show=False)
#     plots['bar'] = plt_to_base64(plt)
    
#     return plots

# def plt_to_base64(plt_obj):
#     """Converts matplotlib plot to base64 string for API transmission."""
#     buf = io.BytesIO()
#     plt_obj.savefig(buf, format="png", bbox_inches="tight")
#     plt_obj.close()
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


######################

# import shap
# import matplotlib.pyplot as plt
# import io
# import base64
# import numpy as np

# def get_shap_plots(model, input_df):
#     """
#     Generates Waterfall and Bar plots.
#     Using TreeExplainer which is optimized for RF and LGBM.
#     """
#     # Force numpy compatibility for SHAP if needed
#     # (Though downgrading numpy is the true fix)
    
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(input_df)
    
#     # For binary classification, SHAP returns a list [prob_0, prob_1]
#     # We take the values for the 'Failure' class (usually index 1)
#     if isinstance(shap_values, list):
#         current_shap_values = shap_values[1][0]
#         expected_value = explainer.expected_value[1]
#     else:
#         # Some versions/models return a single array
#         current_shap_values = shap_values[0]
#         expected_value = explainer.expected_value

#     plots = {}

#     # 1. Waterfall Plot
#     plt.figure(figsize=(10, 6))
#     # Create an Explanation object which Waterfall expects
#     exp = shap.Explanation(
#         values=current_shap_values, 
#         base_values=expected_value, 
#         data=input_df.iloc[0].values, 
#         feature_names=input_df.columns.tolist()
#     )
#     shap.plots.waterfall(exp, show=False)
#     plots['waterfall'] = plt_to_base64(plt)
    
#     # 2. Bar Plot
#     plt.figure(figsize=(10, 6))
#     shap.plots.bar(exp, show=False)
#     plots['bar'] = plt_to_base64(plt)
    
#     return plots

# def plt_to_base64(plt_obj):
#     buf = io.BytesIO()
#     plt_obj.savefig(buf, format="png", bbox_inches="tight")
#     plt_obj.close()
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


import shap
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def get_shap_plots(model, input_df):
    """
    Generates Waterfall and Bar plots.
    Using TreeExplainer which is optimized for RF and LGBM.
    """
    # Force numpy compatibility for SHAP if needed
    # (Though downgrading numpy is the true fix)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # For binary classification, SHAP returns a list [prob_0, prob_1]
    # We take the values for the 'Failure' class (usually index 1)
    if isinstance(shap_values, list):
        current_shap_values = shap_values[1][0]
        expected_value = explainer.expected_value[1]
    else:
        # Some versions/models return a single array
        current_shap_values = shap_values[0]
        expected_value = explainer.expected_value

    plots = {}

    # 1. Waterfall Plot
    plt.figure(figsize=(10, 6))
    # Create an Explanation object which Waterfall expects
    exp = shap.Explanation(
        values=current_shap_values, 
        base_values=expected_value, 
        data=input_df.iloc[0].values, 
        feature_names=input_df.columns.tolist()
    )
    shap.plots.waterfall(exp, show=False)
    plots['waterfall'] = plt_to_base64(plt)
    
    # 2. Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(exp, show=False)
    plots['bar'] = plt_to_base64(plt)
    
    return plots

def plt_to_base64(plt_obj):
    buf = io.BytesIO()
    plt_obj.savefig(buf, format="png", bbox_inches="tight")
    plt_obj.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")