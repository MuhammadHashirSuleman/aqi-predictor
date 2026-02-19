import os
import hopsworks
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def inspect_data():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY", "").strip(),
        project=os.getenv("HOPSWORKS_PROJECT_NAME", "").strip(),
        host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
    )
    fs = project.get_feature_store()
    
    print("Fetching Feature Group Version 2...")
    aqi_fg = fs.get_feature_group(name="aqi_readings", version=2)
    df = aqi_fg.select_all().read()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    print(f"Loaded {len(df)} rows.")
    print(f"Cities found: {df['city'].unique()}")
    
    # Check for autocorrelation (Real data has high auto-corr, random has ~0)
    print("\n--- Autocorrelation Check (Lag-1) ---")
    for city in df['city'].unique():
        city_data = df[df['city'] == city].sort_values('date')
        corr = city_data['aqi'].autocorr(lag=1)
        print(f"{city:10s}: {corr:.4f} (Expected > 0.90 for real data)")
        
        # Plot first 200 hours
        plt.figure(figsize=(10, 4))
        plt.plot(city_data['date'].iloc[:100], city_data['aqi'].iloc[:100])
        plt.title(f"AQI Sample for {city} (Lag-1 Corr: {corr:.2f})")
        plt.grid(True)
        plt.savefig(f"aqi_sample_{city}.png")
        print(f"Saved plot: aqi_sample_{city}.png")

if __name__ == "__main__":
    inspect_data()
