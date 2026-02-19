import os
import hopsworks
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class AQIPredictor:
    def __init__(self):
        self.project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY", "").strip(),
            project=os.getenv("HOPSWORKS_PROJECT_NAME", "").strip(),
            host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
        )
        self.fs = self.project.get_feature_store()
        self.mr = self.project.get_model_registry()
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_meta = None
        
        # Caching
        self._features_cache = None
        self._features_cache_time = None
        self._prediction_cache = None
        self._prediction_cache_time = None
        self._cache_ttl = 600  # 10 minutes
        
    def load_model(self):
        """Downloads and loads the latest approved model."""
        try:
            print("Fetching latest model...")
            model_meta = self.mr.get_models("aqi_forecaster")[0] 
            self.model_meta = model_meta
            
            model_dir = model_meta.download()
            
            # Load scaler and feature columns
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            self.feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
            self.model = joblib.load(os.path.join(model_dir, "model.pkl"))
                
            print(f"Model {model_meta.name} v{model_meta.version} loaded.")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def fetch_latest_features(self):
        """Fetches features with caching to avoid slow Hopsworks calls."""
        # Check cache first
        if self._features_cache is not None and self._features_cache_time is not None:
            elapsed = (datetime.now() - self._features_cache_time).total_seconds()
            if elapsed < self._cache_ttl:
                print(f"Using cached features ({int(self._cache_ttl - elapsed)}s remaining)")
                return self._features_cache
        
        try:
            print("Fetching features from Hopsworks (this may take a minute)...")
            aqi_fg = self.fs.get_feature_group(name="aqi_readings", version=2)
            if aqi_fg is None:
                print("Error: Feature group 'aqi_readings' version 2 not found.")
                return self._features_cache  # Return stale cache if available
            
            df = aqi_fg.select_all().read()
            df['date'] = pd.to_datetime(df['date'])
            latest_data = df.sort_values('date').iloc[[-1]]
            
            # Update cache
            self._features_cache = latest_data
            self._features_cache_time = datetime.now()
            print("Features cached successfully!")
            
            return latest_data
            
        except Exception as e:
            print(f"Error fetching features: {e}")
            # Return stale cache if available
            if self._features_cache is not None:
                print("Returning stale cached features")
                return self._features_cache
            return None

    def predict(self, data=None):
        """
        Generates predictions for T+24, T+48, T+72 with caching.
        """
        # Check prediction cache first
        if self._prediction_cache is not None and self._prediction_cache_time is not None:
            elapsed = (datetime.now() - self._prediction_cache_time).total_seconds()
            if elapsed < self._cache_ttl:
                print(f"Using cached prediction ({int(self._cache_ttl - elapsed)}s remaining)")
                return self._prediction_cache
        
        if self.model is None:
            if not self.load_model():
                return None
        
        if data is None:
            data = self.fetch_latest_features()
        
        if data is None or data.empty:
            return self._prediction_cache  # Return stale cache
            
        # Ensure columns match training
        if self.feature_cols:
            X = data[self.feature_cols].values
        else:
            drop_cols = ['date', 'city', 'y_24', 'y_48', 'y_72']
            input_cols = [c for c in data.columns if c not in drop_cols]
            X = data[input_cols].values
        
        # Scale
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Predict
        preds = self.model.predict(X_scaled)
        result = preds[0]
        
        # Cache prediction
        self._prediction_cache = result
        self._prediction_cache_time = datetime.now()
        print("Prediction cached!")
             
        return result

if __name__ == "__main__":
    predictor = AQIPredictor()
    result = predictor.predict()
    print("Predictions (24h, 48h, 72h):", result)
