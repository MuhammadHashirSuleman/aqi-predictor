import os
import hopsworks
import time
from dotenv import load_dotenv

load_dotenv()

print("Checking Feature Store materialization status...")

project = hopsworks.login(
    host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai"),
    port=443,
    project=os.getenv("HOPSWORKS_PROJECT_NAME", "").strip(),
    api_key_value=os.getenv("HOPSWORKS_API_KEY", "").strip()
)

fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_readings", version=1)

print(f"Feature Group: {fg.name}")
print(f"Version: {fg.version}")

# Try to read a small sample to check if data is available
try:
    print("\nAttempting to read data...")
    df = fg.read(online=False)
    print(f"✅ SUCCESS! Feature Group has {len(df)} rows available.")
    print(f"\nFirst few rows:")
    print(df.head())
except Exception as e:
    print(f"❌ Data not ready yet. Error: {e}")
    print("\nThe materialization job is still running.")
    print("Check job status at: https://eu-west.cloud.hopsworks.ai:443/p/8315/jobs")
    print("\nWait 2-5 minutes and run this script again, or run:")
    print("  python training/train.py")
