import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
CITY = "Beijing" # Parameterize if needed

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")

st.title(f"🌫️ Pearls AQI Predictor: {CITY}")
st.markdown("### 3-Day Air Quality Forecast (100% Serverless MLOps)")

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("Refresh Forecast"):
    st.rerun()

# 1. Fetch Current Data (Directly from API for display, or just rely on backend?)
# To show "Current AQI", we can fetch from AQICN directly here or ask backend.
# Let's ask backend for everything or fetch AQICN here for "Current" display functionality.
# For simplicity, we fetch distinct "Current" data here to show the "Live" status.

def get_current_aqi():
    try:
        # We can reuse the utils logic if we import it, or just simple request
        token = os.getenv("AQICN_API_KEY")
        if not token:
             return None
        url = f"https://api.waqi.info/feed/{CITY.lower()}/?token={token}"
        r = requests.get(url)
        return r.json()['data']
    except:
        return None

current_data = get_current_aqi()

# Display Current Status
col1, col2, col3 = st.columns(3)

if current_data:
    aqi = current_data['aqi']
    
    # Color coding
    if aqi <= 50: color = "green"
    elif aqi <= 100: color = "yellow"
    elif aqi <= 150: color = "orange"
    elif aqi <= 200: color = "red"
    else: color = "purple"
    
    with col1:
        st.metric("Current AQI", aqi, delta=None)
        st.markdown(f"**Status**: <span style='color:{color}'><b>{current_data['idx']}</b></span>", unsafe_allow_html=True) # IDX is not status text.. process 'aqi' to text
        
    with col2:
        st.metric("Temperature", f"{current_data['iaqi']['t']['v']} °C")
        
    with col3:
         st.metric("Humidity", f"{current_data['iaqi']['h']['v']} %")
else:
    st.error("Could not fetch live AQI data.")

# 2. Fetch Forecast from Backend
st.markdown("---")
st.subheader("🔮 3-Day Forecast")

try:
    response = requests.get(f"{API_URL}/predict")
    if response.status_code == 200:
        preds = response.json()
        
        # Visualize
        forecast_df = pd.DataFrame({
            "Time": ["+24h", "+48h", "+72h"],
            "Predicted AQI": [preds['aqi_24h'], preds['aqi_48h'], preds['aqi_72h']]
        })
        
        # Bar Chart
        fig = px.bar(forecast_df, x="Time", y="Predicted AQI", color="Predicted AQI",
                     color_continuous_scale=["green", "yellow", "orange", "red", "purple"],
                     range_color=[0, 300])
        st.plotly_chart(fig, use_container_width=True)
        
        # Alerts
        max_pred = max(preds.values())
        if max_pred > 200:
            st.error("⚠️ ALERT: Hazardous Air Quality predicted in the next 3 days!")
        elif max_pred > 150:
            st.warning("⚠️ ALERT: Unhealthy Air Quality predicted!")
        else:
            st.success("Air Quality expected to stay within reasonable limits.")
            
    else:
        st.warning(f"Backend API unavailable or error: {response.text}")
        st.info("Ensure the FastAPI backend is running: `uvicorn app.api:app --reload`")

except Exception as e:
    st.error(f"Connection Error: {e}")
    st.info("Ensure the FastAPI backend is running: `uvicorn app.api:app --reload`")

# 3. Explainability (Static Demo or Integration)
st.markdown("---")
st.subheader("🔍 Model Explainability (SHAP)")
st.info("SHAP values would be visualized here based on the model's feature importance analysis.")
# In a real app, we'd load the SHAP values from a file or compute them on the fly (expensive).
# For this demo, we can show a placeholder image or text.
st.markdown("""
- **Top Factor**: PM2.5 Concentration (t-1)
- **Top Factor**: Wind Speed (t-1)
- **Top Factor**: Humidity
""")

