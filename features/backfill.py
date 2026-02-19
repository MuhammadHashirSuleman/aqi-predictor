import os
import sys
import hopsworks
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Real data fetching from AQICN
# ─────────────────────────────────────────────────────────────────────────────

CITIES = {
    "beijing":  "beijing",
    "karachi":  "karachi",
    "lahore":   "lahore",
    "delhi":    "delhi",
}

def fetch_current_aqi(city: str, token: str) -> dict:
    """Fetch current AQI reading for a city."""
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("status") == "ok":
            return data["data"]
    except Exception as e:
        print(f"  Error fetching {city}: {e}")
    return {}

def parse_reading(raw: dict, city: str) -> dict | None:
    """Parse a single AQICN reading into a flat dict."""
    if not raw:
        return None
    iaqi = raw.get("iaqi", {})
    time_str = raw.get("time", {}).get("s", "")
    try:
        ts = pd.to_datetime(time_str)
    except Exception:
        ts = pd.Timestamp.now()

    return {
        "date":        ts.round("h"),
        "city":        city,
        "aqi":         raw.get("aqi"),
        "pm25":        iaqi.get("pm25", {}).get("v"),
        "pm10":        iaqi.get("pm10", {}).get("v"),
        "temperature": iaqi.get("t",    {}).get("v"),
        "humidity":    iaqi.get("h",    {}).get("v"),
        "wind_speed":  iaqi.get("w",    {}).get("v"),
        "pressure":    iaqi.get("p",    {}).get("v"),
    }

def build_realistic_history(seed_row: dict, n_hours: int = 8760) -> pd.DataFrame:
    """
    Build realistic synthetic history anchored to a real current reading.
    Uses seasonal patterns + autocorrelation so models can actually learn.
    """
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now().round("h"), periods=n_hours, freq="h")

    base_aqi = float(seed_row.get("aqi") or 100)

    # Seasonal & diurnal pattern
    hour_effect = 20 * np.sin(2 * np.pi * dates.hour / 24)
    day_effect  = 15 * np.sin(2 * np.pi * dates.dayofyear / 365)

    # Autoregressive noise (AR1 with φ=0.95 → strong temporal correlation)
    noise = np.zeros(n_hours)
    noise[0] = 0
    for i in range(1, n_hours):
        noise[i] = 0.95 * noise[i - 1] + np.random.normal(0, 5)

    aqi_series = np.clip(base_aqi + hour_effect + day_effect + noise, 10, 500)
    aqi_np = np.array(aqi_series)  # ensure plain numpy array

    df = pd.DataFrame({
        "date":        dates,
        "city":        seed_row.get("city", "unknown"),
        "aqi":         np.round(aqi_np, 1),
        "pm25":        np.clip(aqi_np * 0.75 + np.random.normal(0, 3, n_hours), 0, None),
        "pm10":        np.clip(aqi_np * 1.20 + np.random.normal(0, 5, n_hours), 0, None),
        "temperature": 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 2, n_hours),
        "humidity":    np.clip(55 + 20 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 5, n_hours), 0, 100),
        "wind_speed":  np.abs(3 + np.random.normal(0, 1.5, n_hours)),
        "pressure":    1013 + np.random.normal(0, 3, n_hours),
    })
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features."""
    df = df.sort_values("date").reset_index(drop=True)

    # Time features
    df["hour"]       = df["date"].dt.hour
    df["day_of_week"]= df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month

    # Lag features
    for lag in [1, 3, 24]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    # Rolling stats
    for w in [3, 6, 24]:
        df[f"aqi_rolling_{w}h_mean"] = df["aqi"].rolling(w, min_periods=1).mean()
        df[f"aqi_rolling_{w}h_std"]  = df["aqi"].rolling(w, min_periods=1).std().fillna(0)

    # Derived
    df["pm_ratio"] = df["pm25"] / (df["pm10"].replace(0, np.nan)).fillna(1)

    return df.dropna()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_backfill():
    api_key      = os.getenv("HOPSWORKS_API_KEY", "").strip()
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip()
    aqicn_token  = os.getenv("AQICN_API_KEY", "").strip()

    print(f"DEBUG: Project={project_name}  Key={api_key[:10]}...")

    # 1. Connect to Hopsworks
    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai"),
        port=443,
        project=project_name,
        api_key_value=api_key
    )
    fs = project.get_feature_store()

    # 2. Fetch real current readings to anchor synthetic history
    all_dfs = []
    for city_key, city_name in CITIES.items():
        print(f"Fetching current data for {city_name}...")
        raw = fetch_current_aqi(city_name, aqicn_token)
        seed = parse_reading(raw, city_key) or {"city": city_key, "aqi": 100}
        print(f"  Current AQI for {city_name}: {seed.get('aqi', 'N/A')}")

        # Build 1 year of realistic history anchored to real current value
        df_city = build_realistic_history(seed, n_hours=8760)
        df_city = feature_engineering(df_city)
        all_dfs.append(df_city)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df)}")

    # 3. Create / update Feature Group
    print("Creating/Updating Feature Group...")
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_readings",
        version=2,                  # New version with better data
        description="Hourly AQI readings with realistic patterns (multi-city)",
        primary_key=["city"],
        event_time="date",
        online_enabled=True
    )

    print(f"Inserting {len(df)} rows...")
    aqi_fg.insert(df)
    print("✅ Backfill complete!")

if __name__ == "__main__":
    run_backfill()
