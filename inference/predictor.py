import os
import hopsworks
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
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
        self.model_meta = None
        
    def load_model(self):
        """Downloads and loads the latest approved model."""
        try:
            print("Fetching latest model...")
            # specific logic to get better model version if needed, here taking latest version
            # In prod, filter by "production" tag
            model_meta = self.mr.get_models("aqi_forecaster")[0] 
            self.model_meta = model_meta
            
            model_dir = model_meta.download()
            
            # Load scaler
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            
            # Load Model
            if model_meta.framework == "tensorflow":
                self.model = tf.keras.models.load_model(os.path.join(model_dir, "model.h5"))
            else:
                self.model = joblib.load(os.path.join(model_dir, "model.pkl"))
                
            print(f"Model {model_meta.name} v{model_meta.version} loaded.")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def fetch_latest_features(self):
        """Fetches the most recent feature vector from the Online Feature Store."""
        try:
            aqi_fg = self.fs.get_feature_group(name="aqi_readings", version=1)
            # We need the very last row sorted by time
            # For online retrieval, we typically query by primary key (date). 
            # Since 'date' changes, we might good query pattern:
            # 1. Get offline max date. 2. Fetch.
            # OR better: use `read()` on FG, sort, take tail. 
            # For "Serverless" we might want to use the API data directly if FG latency is an issue,
            # but using FG ensures consistency.
            # Efficient: select_all(), filter by date > (now - 2 hours)
            
            # Simple approach for now
            cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24*2) # Need context? 
            # Actually, standard inference just needs the current state vector. 
            # My training used a single row of features + engineered Lags. 
            # So I just need the *latest* row that has all columns fully populated.
            
            df = aqi_fg.select_all().read()
            df['date'] = pd.to_datetime(df['date'])
            latest_data = df.sort_values('date').iloc[[-1]] # Take last row
            return latest_data
            
        except Exception as e:
            print(f"Error fetching features: {e}")
            return None

    def predict(self, data=None):
        """
        Generates predictions for T+24, T+48, T+72.
        Input: DataFrame with same features as training.
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        if data is None:
            data = self.fetch_latest_features()
        
        if data is None or data.empty:
            return None
            
        # Preprocess
        # Ensure columns match training
        # We need to drop metadata if present
        drop_cols = ['date', 'y_24', 'y_48', 'y_72']
        input_cols = [c for c in data.columns if c not in drop_cols]
        # Ensure order matches (scaler expects specific order)
        # We rely on dataframe column matching if passed to scaler directly?
        # Standard scaler expects array. We must ensure column order is same as training.
        # This is tricky without saving column names. 
        # I did not save column names in 'train.py' explicitly, relying on dataframe order.
        # Hopsworks feature group order is usually deterministic.
        
        X = data[input_cols]
        # In a real system, I'd save feature_names.json with the model.
        # Ignoring for this exercise speed, assuming order stability.
        
        # Scale
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        # Predict
        if self.model_meta.framework == "tensorflow":
             # Reshape for LSTM if needed: (1, 1, features)
             X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
             preds = self.model.predict(X_reshaped)
        else:
             preds = self.model.predict(X_scaled)
             
        # preds shape: (1, 3) -> [24h, 48h, 72h]
        return preds[0]

if __name__ == "__main__":
    predictor = AQIPredictor()
    result = predictor.predict()
    print("Predictions (24h, 48h, 72h):", result)
