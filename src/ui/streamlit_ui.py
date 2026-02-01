# # # src/ui/streamlit_ui.py
# import streamlit as st
# import requests
# import base64
# from PIL import Image
# import io

# # Configuration
# API_URL = "http://127.0.0.1:8000"

# st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

# st.title("🛠️ Predictive Maintenance Dashboard")
# st.markdown("---")

# # Sidebar for Model Selection (Interaction 1)
# st.sidebar.header("Step 1: Model Selection")
# model_choice = st.sidebar.selectbox("Choose Model", ["LightGBM", "Random Forest"])

# if st.sidebar.button("Load & Initialize Model"):
#     with st.spinner(f"Fetching best {model_choice} from MLflow..."):
#         try:
#             response = requests.post(f"{API_URL}/select_model?model_name={model_choice}")
#             if response.status_code == 200:
#                 metadata = response.json()
#                 st.session_state['metadata'] = metadata
#                 st.sidebar.success(f"{model_choice} Loaded!")
#             else:
#                 st.sidebar.error("Failed to load model metadata.")
#         except Exception as e:
#             st.sidebar.error(f"Connection Error: {e}")

# # Display Metadata if model is loaded
# if 'metadata' in st.session_state:
#     with st.expander("📊 Selected Model Metadata", expanded=True):
#         m = st.session_state['metadata']
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Run Name", m['Run Name'])
#         col2.metric("PR-AUC Score", f"{m['PR-AUC']:.4f}")
#         col3.metric("Optimal Threshold", f"{m['Optimal Threshold']:.4f}")
#         st.write("**Best Parameters:**", m['Best Params'])

# st.markdown("---")

# # Main Area for Prediction (Interaction 2)
# st.header("Step 2: Machine Sensor Inputs")
# col_a, col_b = st.columns(2)

# with col_a:
#     m_type = st.selectbox("Machine Type", ["Low", "Medium", "High"])
#     air_temp = st.slider("Air Temperature [K]", 295.0, 305.0, 300.0)
#     process_temp = st.slider("Process Temperature [K]", 305.0, 315.0, 310.0)

# with col_b:
#     rpm = st.slider("Rotational Speed [rpm]", 1200, 2800, 1500)
#     torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0)
#     tool_wear = st.slider("Tool Wear [min]", 0, 250, 100)

# if st.button("Predict Failure Risk"):
#     if 'metadata' not in st.session_state:
#         st.warning("Please select and load a model in the sidebar first!")
#     else:
#         payload = {
#             "type": m_type,
#             "air_temperature": air_temp,
#             "process_temperature": process_temp,
#             "rotational_speed": rpm,
#             "torque": torque,
#             "tool_wear": tool_wear
#         }
        
#         with st.spinner("Analyzing sensors and generating SHAP values..."):
#             res = requests.post(f"{API_URL}/predict", json=payload)
            
#             if res.status_code == 200:
#                 data = res.json()
#                 details = data['prediction_details']
#                 plots = data['plots']
                
#                 # Display Results
#                 st.subheader("Result")
#                 conf = details['confidence']
#                 color = "red" if details['label'] == "FAILURE DETECTED" else "green"
                
#                 st.markdown(f"### Status: :{color}[{details['label']}]")
#                 st.progress(conf / 100)
#                 st.write(f"**Confidence Level:** {conf}%")
                
#                 # Display SHAP Plots
#                 st.markdown("---")
#                 st.subheader("Explainable AI (SHAP) Insights")
                
#                 tab1, tab2 = st.tabs(["Local Impact (Waterfall)", "Feature Importance (Bar)"])
                
#                 with tab1:
#                     st.write("This plot shows how each sensor reading contributed to *this specific* prediction.")
#                     img_waterfall = base64.b64decode(plots['waterfall'])
#                     st.image(img_waterfall, use_container_width=True)
                    
#                 with tab2:
#                     st.write("This plot shows the global importance of features for this specific instance.")
#                     img_bar = base64.b64decode(plots['bar'])
#                     st.image(img_bar, use_container_width=True)
#             else:
#                 st.error(f"Error in prediction: {res.text}")



import streamlit as st
import requests
import base64

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI4I Maintenance Dashboard", layout="wide")

# --- Helper Functions ---
def get_api_health():
    """Checks if the FastAPI backend is up and gets metadata for loaded models."""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None

# --- UI Header ---
st.title("🛠️ Industrial Predictive Maintenance")
st.markdown("Automated failure detection with Explainable AI (SHAP).")

# --- Sidebar: System Status & Strategy ---
st.sidebar.header("📡 System Control")
health_data = get_api_health()

if health_data and health_data.get("status") == "ready":
    st.sidebar.success("System Status: Operational")
    
    st.sidebar.subheader("1. Select Strategy")
    # This radio button controls which model the backend uses
    strategy = st.sidebar.radio(
        "Maintenance Goal:",
        options=["Balanced", "High Recall"],
        help="Balanced: Best F1-Score (RF). High Recall: Minimize missed failures (LGBM)."
    )
    
    # Map UI name to API parameter
    mode_map = {"Balanced": "balanced", "High Recall": "recall"}
    selected_mode = mode_map[strategy]

    # Show metadata for the active strategy
    metadata = health_data.get("model_metadata", {}).get(selected_mode, {})
    with st.sidebar.expander("📄 Strategy Metadata", expanded=True):
        st.write(f"**Model:** {metadata.get('type')}")
        st.write(f"**Focus:** {metadata.get('goal')}")
else:
    st.sidebar.error("System Status: Offline")
    st.stop() # Halt UI until backend is ready

st.markdown("---")

# --- Main Area: Sensor Inputs ---
st.header("⚙️ Machine Sensor Data")
col_a, col_b = st.columns(2)

with col_a:
    m_type = st.selectbox("Machine Type", ["Low", "Medium", "High"])
    air_temp = st.slider("Air Temperature [K]", 295.0, 305.0, 300.0)
    process_temp = st.slider("Process Temperature [K]", 305.0, 315.0, 310.0)

with col_b:
    rpm = st.slider("Rotational Speed [rpm]", 1200, 2800, 1500)
    torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0)
    tool_wear = st.slider("Tool Wear [min]", 0, 250, 100)

# --- Prediction & SHAP ---
if st.button("🚀 Run Diagnostic"):
    payload = {
        "type": m_type, "air_temperature": air_temp,
        "process_temperature": process_temp, "rotational_speed": rpm,
        "torque": torque, "tool_wear": tool_wear
    }
    
    # We pass the selected_mode as a query parameter
    with st.spinner(f"Analyzing using {strategy} strategy..."):
        res = requests.post(f"{API_URL}/predict?mode={selected_mode}", json=payload)
        
        if res.status_code == 200:
            data = res.json()
            details = data['prediction_details']
            plots = data['plots']
            
            # Show Result
            color = "red" if details['label'] == "FAILURE DETECTED" else "green"
            st.subheader("Diagnostic Result")
            st.markdown(f"### Status: :{color}[{details['label']}]")
            
            conf = details['confidence']
            st.progress(conf / 100)
            st.write(f"**Confidence Level:** {conf}%")
            
            # Show SHAP Plots
            st.markdown("---")
            st.subheader("🔍 Failure Root-Cause Analysis (SHAP)")
            tab1, tab2 = st.tabs(["Waterfall (Local)", "Bar (Global)"])
            
            with tab1:
                st.image(base64.b64decode(plots['waterfall']), use_container_width=True)
            with tab2:
                st.image(base64.b64decode(plots['bar']), use_container_width=True)
        else:
            st.error(f"Prediction Error: {res.text}")