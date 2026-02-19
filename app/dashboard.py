import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .aqi-good { color: #4CAF50; font-weight: bold; }
    .aqi-moderate { color: #FFEB3B; font-weight: bold; }
    .aqi-sensitive { color: #FF9800; font-weight: bold; }
    .aqi-unhealthy { color: #F44336; font-weight: bold; }
    .aqi-hazardous { color: #9C27B0; font-weight: bold; }
    .stMetric > div { background-color: #f8f9fa; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/air-quality.png", width=80)
    st.title("🌫️ AQI Predictor")
    st.markdown("---")
    
    # City Selection
    st.subheader("🏙️ Select City")
    CITY = st.selectbox(
        "Choose a city for live data:",
        ["Beijing", "Delhi", "Karachi", "Lahore", "Shanghai", "Mumbai"],
        index=0
    )
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    # Info section
    st.subheader("ℹ️ About")
    st.markdown("""
    **Pearls AQI Predictor** uses machine learning to forecast air quality for the next 3 days.
    
    - 📊 **Data**: AQICN API
    - 🤖 **Model**: Gradient Boosting
    - ☁️ **Infrastructure**: Hopsworks
    - 🔄 **Updates**: Hourly
    """)
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Main Header
st.markdown('<p class="main-header">🌫️ Pearls AQI Predictor</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">3-Day Air Quality Forecast for <b>{CITY}</b> | 100% Serverless MLOps</p>', unsafe_allow_html=True)

# Helper functions
def get_aqi_status(aqi):
    """Get AQI status text and color"""
    if aqi <= 50:
        return "Good", "#4CAF50", "🟢"
    elif aqi <= 100:
        return "Moderate", "#FFEB3B", "🟡"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "#FF9800", "🟠"
    elif aqi <= 200:
        return "Unhealthy", "#F44336", "🔴"
    elif aqi <= 300:
        return "Very Unhealthy", "#9C27B0", "🟣"
    else:
        return "Hazardous", "#8B0000", "⚫"

def get_current_aqi(city):
    """Fetch current AQI from AQICN API"""
    try:
        token = os.getenv("AQICN_API_KEY")
        if not token:
            return None
        url = f"https://api.waqi.info/feed/{city.lower()}/?token={token}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get('status') == 'ok':
            return data['data']
        return None
    except:
        return None

# Fetch current data
current_data = get_current_aqi(CITY)

# Current AQI Section
st.markdown("### 📍 Current Air Quality")

if current_data:
    aqi = current_data.get('aqi', 0)
    status, color, emoji = get_aqi_status(aqi)
    iaqi = current_data.get('iaqi', {})
    
    # Main metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🌡️ AQI",
            value=f"{emoji} {aqi}",
            delta=status
        )
    
    with col2:
        temp = iaqi.get('t', {}).get('v', 'N/A')
        st.metric(label="🌡️ Temperature", value=f"{temp} °C" if temp != 'N/A' else 'N/A')
    
    with col3:
        humidity = iaqi.get('h', {}).get('v', 'N/A')
        st.metric(label="💧 Humidity", value=f"{humidity} %" if humidity != 'N/A' else 'N/A')
    
    with col4:
        pm25 = iaqi.get('pm25', {}).get('v', 'N/A')
        st.metric(label="🪔 PM2.5", value=f"{pm25} µg/m³" if pm25 != 'N/A' else 'N/A')
    
    with col5:
        pm10 = iaqi.get('pm10', {}).get('v', 'N/A')
        st.metric(label="🌬️ PM10", value=f"{pm10} µg/m³" if pm10 != 'N/A' else 'N/A')
    
    # Status box
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {color}22, {color}11); 
                border-left: 4px solid {color}; 
                padding: 1rem; 
                border-radius: 5px; 
                margin: 1rem 0;">
        <b>Status:</b> {emoji} <span style="color: {color}; font-weight: bold;">{status}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Station:</b> {current_data.get('city', {}).get('name', CITY)}
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("❌ Could not fetch live AQI data. Please check your AQICN_API_KEY.")

# 3-Day Forecast Section
st.markdown("---")
st.markdown("### 🔮 3-Day AQI Forecast")

try:
    response = requests.get(f"{API_URL}/predict", timeout=10)
    if response.status_code == 200:
        preds = response.json()
        
        # Forecast cards
        col1, col2, col3 = st.columns(3)
        
        forecast_times = [
            ("+24 Hours", preds['aqi_24h'], datetime.now() + timedelta(hours=24)),
            ("+48 Hours", preds['aqi_48h'], datetime.now() + timedelta(hours=48)),
            ("+72 Hours", preds['aqi_72h'], datetime.now() + timedelta(hours=72))
        ]
        
        for col, (label, aqi_val, date) in zip([col1, col2, col3], forecast_times):
            status, color, emoji = get_aqi_status(aqi_val)
            with col:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}33, {color}11); 
                            border: 2px solid {color}; 
                            border-radius: 15px; 
                            padding: 1.5rem; 
                            text-align: center;">
                    <h4 style="margin: 0; color: #333;">{label}</h4>
                    <p style="color: #666; margin: 0.5rem 0;">{date.strftime('%b %d, %H:%M')}</p>
                    <h1 style="color: {color}; margin: 0.5rem 0;">{emoji} {int(aqi_val)}</h1>
                    <p style="color: {color}; font-weight: bold;">{status}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Forecast chart
        forecast_df = pd.DataFrame({
            "Time": ["+24h", "+48h", "+72h"],
            "Predicted AQI": [preds['aqi_24h'], preds['aqi_48h'], preds['aqi_72h']],
            "Date": [(datetime.now() + timedelta(hours=h)).strftime('%b %d') for h in [24, 48, 72]]
        })
        
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=forecast_df['Time'],
            y=forecast_df['Predicted AQI'],
            marker=dict(
                color=forecast_df['Predicted AQI'],
                colorscale=[[0, '#4CAF50'], [0.33, '#FFEB3B'], [0.5, '#FF9800'], [0.66, '#F44336'], [1, '#9C27B0']],
                cmin=0,
                cmax=300,
            ),
            text=[f"{int(v)}" for v in forecast_df['Predicted AQI']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>'
        ))
        
        # Add threshold lines
        fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
        fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy (Sensitive)")
        fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Unhealthy")
        
        fig.update_layout(
            title="Predicted AQI Over Next 3 Days",
            xaxis_title="Forecast Period",
            yaxis_title="AQI Value",
            yaxis=dict(range=[0, max(350, max(forecast_df['Predicted AQI']) + 50)]),
            showlegend=False,
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Alerts
        max_pred = max(preds.values())
        if max_pred > 200:
            st.error("🚨 **HAZARDOUS ALERT**: Dangerous air quality predicted! Avoid outdoor activities.")
        elif max_pred > 150:
            st.warning("⚠️ **UNHEALTHY ALERT**: Air quality may be unhealthy. Sensitive groups should limit outdoor exposure.")
        elif max_pred > 100:
            st.info("🟡 **MODERATE**: Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.")
        else:
            st.success("✅ **GOOD**: Air quality is satisfactory with little or no health risk.")
            
    else:
        st.warning(f"⚠️ Backend API unavailable: {response.text}")
        st.info("💡 Make sure the FastAPI backend is running: `uvicorn app.api:app --reload`")

except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to prediction API")
    st.info("""
    **To start the backend:**
    ```bash
    uvicorn app.api:app --reload --port 8000
    ```
    """)
except Exception as e:
    st.error(f"❌ Error: {e}")

# SHAP Explainability Section
st.markdown("---")
st.markdown("### 🔍 Model Explainability (SHAP)")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    **Feature Importance** - What drives AQI predictions:
    
    | Rank | Feature | Impact |
    |------|---------|--------|
    | 1 | PM2.5 (t-1) | 🟢🟢🟢🟢🟢 High |
    | 2 | AQI Lag 24h | 🟢🟢🟢🟢 High |
    | 3 | Wind Speed | 🟢🟢🟢 Medium |
    | 4 | Humidity | 🟢🟢🟢 Medium |
    | 5 | Temperature | 🟢🟢 Low |
    | 6 | Hour of Day | 🟢🟢 Low |
    """)

with col2:
    st.markdown("""
    **Key Insights:**
    
    - 🌡️ **PM2.5 concentration** is the strongest predictor of future AQI
    - 📈 **Historical AQI patterns** (24h lag) heavily influence forecasts
    - 🌬️ **Wind speed** helps disperse pollutants, lowering AQI
    - 💧 **Humidity** can trap particles, increasing AQI
    - ⏰ **Time of day** affects traffic patterns and emissions
    """)

# AQI Reference Guide
st.markdown("---")
st.markdown("### 📖 AQI Reference Guide")

aqi_guide = pd.DataFrame({
    "AQI Range": ["0-50", "51-100", "101-150", "151-200", "201-300", "301+"],
    "Level": ["🟢 Good", "🟡 Moderate", "🟠 Unhealthy (Sensitive)", "🔴 Unhealthy", "🟣 Very Unhealthy", "⚫ Hazardous"],
    "Health Advisory": [
        "Air quality is satisfactory",
        "Acceptable; sensitive individuals may experience minor effects",
        "Sensitive groups may experience health effects",
        "Everyone may begin to experience health effects",
        "Health alert: everyone may experience serious effects",
        "Emergency conditions; entire population affected"
    ]
})

st.dataframe(aqi_guide, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌫️ <b>Pearls AQI Predictor</b> | Powered by Hopsworks & AQICN</p>
    <p>Data updates hourly | Model retrains daily</p>
</div>
""", unsafe_allow_html=True)

