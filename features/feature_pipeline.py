import os
import sys
import hopsworks
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Add features/ to path so imports work from repo root OR features/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from aqi_utils import fetch_aqi_data, process_aqi_data, feature_engineering

# Load environment variables
load_dotenv()

def run_feature_pipeline():
    # 1. Connect to Hopsworks
    # Using connection for consistency
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY", "").strip(),
        project=os.getenv("HOPSWORKS_PROJECT_NAME", "").strip(),
        host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
    )
    fs = project.get_feature_store()
    
    # 2. Fetch Live Data
    print("Fetching live data...")
    raw_data = fetch_aqi_data(os.getenv("AQICN_API_KEY"))
    current_df = process_aqi_data(raw_data)
    
    if current_df.empty:
        print("No data fetched. Exiting.")
        return

    # 3. Get or create Feature Group (ensures it exists for insert later)
    print("Getting/creating Feature Group...")
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_readings",
        version=2,
        description="Hourly AQI readings with features",
        primary_key=["city"],
        event_time="date",
        online_enabled=True
    )
    print(f"Feature Group object: {aqi_fg}, type: {type(aqi_fg)}")
    
    if aqi_fg is None:
        raise RuntimeError("Failed to get or create feature group - returned None")

    # 4. Fetch History for Context (Lags/Rolling)
    # We need at least 24 hours of history to compute 24h lag/rolling
    print("Fetching context from Feature Store...")
    history_df = pd.DataFrame()
    try:
        # Calculate cutoff time
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(hours=48)
        
        # This reads into a dataframe
        history_df = aqi_fg.select_all().read() 
        if not history_df.empty:
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df = history_df[history_df['date'] >= cutoff_date]
        
    except Exception as e:
        print(f"Could not fetch history (maybe first run): {e}")
        history_df = pd.DataFrame()

    # 5. Combine & Compute Features
    print("Computing features...")
    if not history_df.empty:
        # Drop the current timestamp if it already exists in history to avoid dups before processing
        # (Though process_aqi_data sorts it out)
        history_df = history_df[history_df['date'] < current_df['date'].iloc[0]]
        full_df = pd.concat([history_df, current_df])
    else:
        full_df = current_df
        
    full_df = full_df.sort_values('date')
    processed_df = feature_engineering(full_df)
    
    # 6. Extract only the new data point(s)
    # We only want to insert the latest row(s) that match our current fetch
    new_data = processed_df[processed_df['date'].isin(current_df['date'])]
    
    if new_data.empty:
        print("No new unique data to insert.")
        return

    # 7. Insert into Feature Store
    print(f"Inserting {len(new_data)} rows...")
    
    # CASTING to match Hopsworks schema (int for AQI, float for others)
    new_data['aqi'] = new_data['aqi'].astype(int)
    new_data['pm25'] = new_data['pm25'].astype(float)
    new_data['pm10'] = new_data['pm10'].astype(float)
    new_data['temperature'] = new_data['temperature'].astype(float)
    new_data['humidity'] = new_data['humidity'].astype(float)
    new_data['pressure'] = new_data['pressure'].astype(float)
    new_data['wind_speed'] = new_data['wind_speed'].astype(float)
    new_data['pm_ratio'] = new_data['pm_ratio'].astype(float)
    
    # Ensure date is timezone-naive or strictly UTC to match Hopsworks
    new_data['date'] = pd.to_datetime(new_data['date']).dt.tz_localize(None)

    aqi_fg.insert(new_data)
    print("Feature Pipeline complete!")

if __name__ == "__main__":
    run_feature_pipeline()
