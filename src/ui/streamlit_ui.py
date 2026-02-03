# src/ui/streamlit_ui.py
import streamlit as st
import requests
import base64
from PIL import Image
import io

# 1. Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Maintenance AI | @champion Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar: Model Strategy
st.sidebar.title("🚀 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Model Strategy", 
    ["Failure-Avoidance Mode", "False-Alarm Reduction Mode"],
    help="Choose LightGBM when when cost of machine failure is high" \
    "Choose False-Alarm Reduction Mode when cost of false alarms is high"
)

# Automatic Metadata Sync
try:
    info_res = requests.get(f"{API_URL}/model_info/{model_choice}")
    if info_res.status_code == 200:
        model_data = info_res.json()
        st.sidebar.success(f"Connected")
        
        # Display Registry Info in Sidebar
        with st.sidebar.expander("🔍 Registry Details"):
            st.json(model_data['metadata'])
    else:
        st.sidebar.error("Could not sync with Registry API.")
except Exception as e:
    st.sidebar.warning("API Offline. Please start the FastAPI server.")

# 3. Main Dashboard UI
st.title("🛠️ Predictive Maintenance Intelligence")
st.markdown(f"Running Inference on: **{model_choice}**")

# Input Section
with st.container():
    st.subheader("📡 Real-time Sensor Telemetry")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        m_type = st.selectbox("Machine Grade", ["Low", "Medium", "High"])
        air_temp = st.slider("Air Temp [K]", 295.0, 305.0, 300.0)
    with col2:
        process_temp = st.slider("Process Temp [K]", 305.0, 315.0, 310.0)
        rpm = st.slider("Rotational Speed [RPM]", 1000, 3000, 1500)
    with col3:
        torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0)
        tool_wear = st.slider("Tool Wear [min]", 0, 250, 50)

st.divider()

# # 4. Prediction Execution
# if st.button("🚀 Analyze Failure Risk", use_container_width=True):
#     payload = {
#         "type": m_type,
#         "air_temperature": air_temp,
#         "process_temperature": process_temp,
#         "rotational_speed": int(rpm),
#         "torque": torque,
#         "tool_wear": tool_wear
#     }
    
#     with st.spinner("Inference in progress..."):
#         res = requests.post(f"{API_URL}/predict/{model_choice}", json=payload)
        
#         if res.status_code == 200:
#             data = res.json()
#             details = data['prediction_details']
#             plots = data['explanations']
            
#             # --- Results Row ---
#             res_col1, res_col2 = st.columns([1, 2])
            
#             with res_col1:
#                 st.subheader("Inference Result")
#                 color = "red" if details['prediction_code'] == 1 else "green"
#                 st.markdown(f"<h1 style='color: {color};'>{details['label']}</h1>", unsafe_allow_html=True)
#                 st.metric("Probability", f"{details['probability']*100:.1f}%")
#                 st.metric("Threshold Used", f"{data['model_context']['Threshold']}")
            
#             with res_col2:
#                 st.subheader("SHAP Explanation")
#                 tab1, tab2 = st.tabs(["Waterfall (Local Impact)", "Bar (Global Magnitude)"])
#                 with tab1:
#                     st.image(base64.b64decode(plots['waterfall']))
#                 with tab2:
#                     st.image(base64.b64decode(plots['bar']))
#         else:
#             st.error(f"Prediction Error: {res.text}")

# 4. Prediction Execution
if st.button("🚀 Analyze Failure Risk", use_container_width=True):
    payload = {
        "type": m_type,
        "air_temperature": air_temp,
        "process_temperature": process_temp,
        "rotational_speed": int(rpm),
        "torque": torque,
        "tool_wear": tool_wear
    }
    
    with st.spinner("Inference in progress..."):
        try:
            res = requests.post(f"{API_URL}/predict/{model_choice}", json=payload)
            
            if res.status_code == 200:
                data = res.json()
                details = data.get('prediction_details', {})
                plots = data.get('explanations', {})
                
                # --- Results Row ---
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.subheader("Inference Result")
                    # Handle None or missing codes safely
                    pred_code = details.get('prediction_code', 0)
                    color = "red" if pred_code == 1 else "green"
                    
                    st.markdown(f"<h1 style='color: {color};'>{details.get('label', 'N/A')}</h1>", unsafe_allow_html=True)
                    st.metric("Probability", f"{details.get('probability', 0)*100:.1f}%")
                    
                    # Safely access threshold from model_context
                    m_context = data.get('model_context', {})
                    st.metric("Threshold Used", f"{m_context.get('Threshold', 'N/A')}")
                
                with res_col2:
                    st.subheader("SHAP Explanation")
                    tab1, tab2 = st.tabs(["Waterfall (Local Impact)", "Bar (Global Magnitude)"])
                    
                    with tab1:
                        img_data = plots.get('waterfall')
                        if img_data: # Only decode if string is not empty
                            st.image(base64.b64decode(img_data))
                        else:
                            st.warning("Waterfall plot generation failed in backend.")
                            
                    with tab2:
                        img_data = plots.get('bar')
                        if img_data:
                            st.image(base64.b64decode(img_data))
                        else:
                            st.warning("Bar plot generation failed in backend.")
            else:
                st.error(f"API Error: {res.text}")
                
        except Exception as e:
            st.error(f"Connection or UI Error: {str(e)}")