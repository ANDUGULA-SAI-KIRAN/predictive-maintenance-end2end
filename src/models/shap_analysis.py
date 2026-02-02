import matplotlib
# MUST be the very first matplotlib call to prevent "Main thread" crashes
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import shap
import io
import base64
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_shap_plots(pyfunc_model, input_df):
    try:
        # 1. UNWRAP THE MODEL
        impl = pyfunc_model._model_impl
        native_model = None
        
        logger.info(f"Received model type: {type(pyfunc_model)}")
        logger.info(f"Model implementation type: {type(impl)}")

        # Logic for Sklearn (Working in your logs)
        if hasattr(impl, "sklearn_model"):
            native_model = impl.sklearn_model
            logger.info(f"Found native model in .sklearn_model")
            logger.info(f"Passing to SHAP: {type(native_model)}")
        
        # Logic for LightGBM (MLflow wraps it in _LGBModelWrapper)
        elif hasattr(impl, "get_raw_model"):
            # MLflow's _LGBModelWrapper has get_raw_model() method
            try:
                native_model = impl.get_raw_model()
                logger.info(f"Extracted LightGBM model via get_raw_model(): {type(native_model)}")
            except:
                # Fallback: try lgb_model attribute
                if hasattr(impl, "lgb_model"):
                    native_model = impl.lgb_model
                    logger.info(f"Extracted LightGBM model via lgb_model: {type(native_model)}")
                else:
                    raise ValueError(f"Cannot extract model from wrapper: {type(impl)}")
        
        elif hasattr(impl, "lgb_model"):
            # Direct attribute access
            native_model = impl.lgb_model
            logger.info(f"Extracted LightGBM model via lgb_model: {type(native_model)}")
        
        else:
            # Log available attributes for debugging
            available_attrs = [attr for attr in dir(impl) if not attr.startswith('_')]
            logger.warning(f"Could not find standard attributes. Available: {available_attrs}")
            raise ValueError(f"Unsupported model wrapper: {type(impl)}")

        # 2. INITIALIZE EXPLAINER
        # We pass the native model. TreeExplainer works with LGBM Booster or Sklearn RF
        explainer = shap.TreeExplainer(native_model)
        
        # 3. CALCULATE AND SLICE FOR FAILURE (CLASS 1)
        # We want to explain why a machine IS failing
        shap_values = explainer.shap_values(input_df)

        # Handling different return shapes from RF vs LGBM
        if isinstance(shap_values, list):
            # RF usually returns [class0, class1]
            sv = shap_values[1]
            ev = explainer.expected_value[1]
        elif len(shap_values.shape) == 3:
            # Shape: (samples, features, classes)
            sv = shap_values[0, :, 1]
            ev = explainer.expected_value[1]
        elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
            # Shape: (samples, classes)
            sv = shap_values[:, 1]
            ev = explainer.expected_value[1]
        else:
            # Single output / Regression / Binary LGBM native
            sv = shap_values
            ev = explainer.expected_value

        # Standardize for a single explanation row
        if len(sv.shape) > 1:
            sv = sv[0]
        if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
            ev = ev[1]

        # 4. CREATE EXPLANATION OBJECT
        exp = shap.Explanation(
            values=sv,
            base_values=ev,
            data=input_df.iloc[0].values,
            feature_names=input_df.columns.tolist()
        )

        # 5. PLOT GENERATION (Thread-safe)
        plots = {}
        
        logger.info(f"Raw SHAP results type: {type(shap_values)}")
        
        try:
            # Create Waterfall Plot
            fig1 = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp, show=False)
            plots['waterfall'] = plt_to_base64(fig1)
        except Exception as e:
            logger.warning(f"Failed to generate waterfall plot: {str(e)}")
        
        try:
            # Create Bar Plot
            fig2 = plt.figure(figsize=(10, 6))
            shap.plots.bar(exp, show=False)
            plots['bar'] = plt_to_base64(fig2)
        except Exception as e:
            logger.warning(f"Failed to generate bar plot: {str(e)}")
        
        logger.info("SHAP plots generated successfully.")
        return plots

    except Exception as e:
        logger.error(f"Failed to generate SHAP: {str(e)}", exc_info=True)
        return None

def plt_to_base64(fig_obj):
    """Converts a specific figure object to base64 to avoid thread conflicts."""
    try:
        buf = io.BytesIO()
        fig_obj.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close(fig_obj)  # Properly close the specific figure
        return base64_str
    except Exception as e:
        logger.error(f"Error converting plot to base64: {str(e)}")
        plt.close('all')  # Force close all figures on error
        return None