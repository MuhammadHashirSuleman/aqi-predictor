import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv
from aqi_utils import fetch_historical_aqicn, feature_engineering

# Load environment variables
load_dotenv()

def run_backfill():
    # Debug: Check what key is actually being loaded
    api_key = os.getenv("HOPSWORKS_API_KEY", "").strip()
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip()
    
    print(f"DEBUG: Loaded Project: {project_name}")
    print(f"DEBUG: Loaded Key (first 10): {api_key[:10] if api_key else 'None'}")
    
    # 1. Connect to Hopsworks
    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai"),
        port=443,
        project=project_name,
        api_key_value=api_key
    )
    fs = project.get_feature_store()

    # 2. Fetch Historical Data (365 days)
    print("Fetching historical data...")
    # Using synthetic data for backfill since real historical API access is often paid/limited
    # In a real scenario, this would loop through dates and call valid history endpoints
    df = fetch_historical_aqicn(os.getenv("AQICN_API_KEY")) 
    
    # 3. Feature Engineering
    print("Computing features...")
    # Sort by date to ensure rolling windows work
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Generate lag features and rolling stats on the full history
    df = feature_engineering(df)
    
    # Drop rows with NaN if any (from initial lag windows)
    df = df.dropna()

    # 4. Create Feature Group
    print("Creating/Updating Feature Group...")
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_readings",
        version=1,
        description="Hourly AQI readings and weather data",
        primary_key=["city"],
        event_time="date",
        online_enabled=True
    )
    
    # 5. Insert Data
    print(f"Inserting {len(df)} rows...")
    aqi_fg.insert(df)
    print("Backfill complete!")

if __name__ == "__main__":
    run_backfill()
