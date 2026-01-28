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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # For binary classification, SHAP returns a list [prob_0, prob_1]
    # We take the values for the 'Failure' class (usually index 1)
    if isinstance(shap_values, list):
        current_shap_values = shap_values[1][0]
        expected_value = explainer.expected_value[1] # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
    else:
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