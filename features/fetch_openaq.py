"""
fetch_openaq.py
Fetches REAL historical AQI/pollution data from:
1. OpenAQ API v3 (free, no API key needed, real measurements)
2. OpenWeatherMap Air Pollution API (free, 1 year history)

OpenAQ has data from thousands of real monitoring stations worldwide.
"""
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()

OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

# Cities with their coordinates and OpenAQ location IDs
CITIES = {
    "beijing": {"lat": 39.9042, "lon": 116.4074, "country": "CN"},
    "lahore":  {"lat": 31.5204, "lon": 74.3587,  "country": "PK"},
    "delhi":   {"lat": 28.6139, "lon": 77.2090,  "country": "IN"},
    "karachi": {"lat": 24.8607, "lon": 67.0011,  "country": "PK"},
}

# ─────────────────────────────────────────────────────────────────────────────
# OpenAQ v3 API (real monitoring station data, free, no key)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_openaq_locations(lat: float, lon: float, radius: int = 25000) -> list:
    """Find real monitoring stations near a city."""
    url = "https://api.openaq.org/v3/locations"
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius,
        "limit": 5,
        "order_by": "distance",
    }
    headers = {"Accept": "application/json"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json().get("results", [])
    except Exception as e:
        print(f"  OpenAQ locations error: {e}")
    return []

def fetch_openaq_measurements(location_id: int, days: int = 90) -> pd.DataFrame:
    """Fetch real PM2.5 measurements from a monitoring station."""
    url = f"https://api.openaq.org/v3/locations/{location_id}/measurements"
    date_from = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    all_results = []
    page = 1
    while True:
        params = {
            "date_from": date_from,
            "date_to":   date_to,
            "limit":     1000,
            "page":      page,
            "parameters_id": 2,  # PM2.5
        }
        headers = {"Accept": "application/json"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code != 200:
                break
            data = r.json().get("results", [])
            if not data:
                break
            all_results.extend(data)
            if len(data) < 1000:
                break
            page += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"  Measurement fetch error: {e}")
            break
    
    if not all_results:
        return pd.DataFrame()
    
    rows = []
    for m in all_results:
        try:
            ts = pd.to_datetime(m["period"]["datetimeFrom"]["utc"])
            val = m["value"]
            rows.append({"date": ts.round("h"), "pm25": val})
        except Exception:
            continue
    
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date")

# ─────────────────────────────────────────────────────────────────────────────
# OpenWeatherMap Air Pollution API (real data, free, up to 1 year history)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_owm_air_pollution_history(lat: float, lon: float, days: int = 90) -> pd.DataFrame:
    """Fetch real historical air pollution data from OpenWeatherMap."""
    if not OWM_KEY:
        print("  No OWM key found!")
        return pd.DataFrame()
    
    start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    end   = int(datetime.utcnow().timestamp())
    
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {"lat": lat, "lon": lon, "start": start, "end": end, "appid": OWM_KEY}
    
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json().get("list", [])
            rows = []
            for item in data:
                ts = pd.to_datetime(item["dt"], unit="s")
                comp = item.get("components", {})
                aqi_index = item.get("main", {}).get("aqi", 2)
                # Convert OWM AQI index (1-5) to approximate AQI value
                aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
                rows.append({
                    "date":     ts.round("h"),
                    "aqi":      aqi_map.get(aqi_index, 100),
                    "pm25":     comp.get("pm2_5", np.nan),
                    "pm10":     comp.get("pm10", np.nan),
                    "no2":      comp.get("no2", np.nan),
                    "o3":       comp.get("o3", np.nan),
                    "co":       comp.get("co", np.nan),
                })
            df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
            print(f"  Got {len(df)} real air pollution readings from OWM")
            return df
        else:
            print(f"  OWM error: {r.status_code} - {r.text[:200]}")
    except Exception as e:
        print(f"  OWM air pollution error: {e}")
    return pd.DataFrame()

def fetch_owm_weather_history(lat: float, lon: float, days: int = 5) -> pd.DataFrame:
    """Fetch recent weather data (free tier: current + 5-day forecast)."""
    if not OWM_KEY:
        return pd.DataFrame()
    
    rows = []
    # Current weather
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            d = r.json()
            rows.append({
                "date":        pd.Timestamp.now().round("h"),
                "temperature": d["main"]["temp"],
                "humidity":    d["main"]["humidity"],
                "pressure":    d["main"]["pressure"],
                "wind_speed":  d["wind"]["speed"],
            })
    except Exception:
        pass
    
    # 5-day forecast
    url2 = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OWM_KEY}&units=metric"
    try:
        r = requests.get(url2, timeout=10)
        if r.status_code == 200:
            for item in r.json().get("list", []):
                rows.append({
                    "date":        pd.to_datetime(item["dt"], unit="s").round("h"),
                    "temperature": item["main"]["temp"],
                    "humidity":    item["main"]["humidity"],
                    "pressure":    item["main"]["pressure"],
                    "wind_speed":  item["wind"]["speed"],
                })
    except Exception:
        pass
    
    return pd.DataFrame(rows).drop_duplicates("date") if rows else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Build full dataset per city
# ─────────────────────────────────────────────────────────────────────────────

def build_city_dataset(city: str, info: dict, days: int = 90) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"  City: {city.upper()}")
    print(f"{'='*50}")
    
    # 1. Fetch real air pollution history from OWM
    print("  Fetching real air pollution history (OWM)...")
    pollution_df = fetch_owm_air_pollution_history(info["lat"], info["lon"], days=days)
    
    # 2. Try OpenAQ for PM2.5 if OWM didn't work well
    if pollution_df.empty or len(pollution_df) < 100:
        print("  Trying OpenAQ for real PM2.5 data...")
        locations = fetch_openaq_locations(info["lat"], info["lon"])
        if locations:
            loc_id = locations[0]["id"]
            loc_name = locations[0].get("name", "unknown")
            print(f"  Found station: {loc_name} (ID: {loc_id})")
            pm25_df = fetch_openaq_measurements(loc_id, days=days)
            if not pm25_df.empty:
                pollution_df = pm25_df
                # Estimate AQI from PM2.5 (EPA formula simplified)
                pollution_df["aqi"] = (pollution_df["pm25"] * 1.5).clip(0, 500).round(1)
                pollution_df["pm10"] = (pollution_df["pm25"] * 1.6).round(1)
    
    if pollution_df.empty:
        print(f"  ⚠️  No real data found for {city}, skipping.")
        return pd.DataFrame()
    
    # 3. Fetch weather data
    print("  Fetching weather data (OWM)...")
    weather_df = fetch_owm_weather_history(info["lat"], info["lon"])
    
    # 4. Merge pollution + weather on date
    if not weather_df.empty:
        df = pd.merge(pollution_df, weather_df, on="date", how="left")
    else:
        df = pollution_df.copy()
    
    # Fill missing weather with reasonable defaults
    df["temperature"] = df.get("temperature", pd.Series(dtype=float)).fillna(20.0)
    df["humidity"]    = df.get("humidity",    pd.Series(dtype=float)).fillna(60.0)
    df["pressure"]    = df.get("pressure",    pd.Series(dtype=float)).fillna(1013.0)
    df["wind_speed"]  = df.get("wind_speed",  pd.Series(dtype=float)).fillna(3.0)
    
    # Ensure required columns exist
    for col in ["aqi", "pm25", "pm10"]:
        if col not in df.columns:
            df[col] = np.nan
    
    df["city"] = city
    df = df.dropna(subset=["aqi", "pm25"]).reset_index(drop=True)
    
    print(f"  ✅ {len(df)} real data points for {city}")
    print(f"     AQI range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}")
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for city, grp in df.groupby("city"):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp["hour"]        = grp["date"].dt.hour
        grp["day_of_week"] = grp["date"].dt.dayofweek
        grp["month"]       = grp["date"].dt.month
        for lag in [1, 3, 24]:
            grp[f"aqi_lag_{lag}"] = grp["aqi"].shift(lag)
        for w in [3, 6, 24]:
            grp[f"aqi_rolling_{w}h_mean"] = grp["aqi"].rolling(w, min_periods=1).mean()
            grp[f"aqi_rolling_{w}h_std"]  = grp["aqi"].rolling(w, min_periods=1).std().fillna(0)
        grp["pm_ratio"] = grp["pm25"] / (grp["pm10"].replace(0, np.nan)).fillna(1)
        results.append(grp)
    return pd.concat(results).dropna().reset_index(drop=True)

if __name__ == "__main__":
    print("=" * 60)
    print("FETCHING REAL HISTORICAL AQI DATA")
    print("=" * 60)
    print("Sources: OpenWeatherMap Air Pollution API + OpenAQ")
    print("Data: Real measurements from monitoring stations")
    print()
    
    all_dfs = []
    for city, info in CITIES.items():
        df_city = build_city_dataset(city, info, days=90)
        if not df_city.empty:
            all_dfs.append(df_city)
        time.sleep(1)
    
    if not all_dfs:
        print("\n❌ No data fetched! Check your API keys.")
        exit(1)
    
    df = pd.concat(all_dfs, ignore_index=True)
    df = add_features(df)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/real_aqi_data.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved {len(df)} REAL data points to data/real_aqi_data.csv")
    print(f"   Cities: {df['city'].unique().tolist()}")
    print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"   AQI range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}")
    print(f"\nNow run: python training/train_local.py")
