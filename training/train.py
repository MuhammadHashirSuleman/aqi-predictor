import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load environment variables
load_dotenv()

def create_targets(df):
    """Creates target columns for T+24, T+48, T+72 hours."""
    # We want to predict AQI at specific horizons
    df = df.sort_values('date')
    df['y_24'] = df['aqi'].shift(-24)
    df['y_48'] = df['aqi'].shift(-48)
    df['y_72'] = df['aqi'].shift(-72)
    return df

def train_model():
    # 1. Connect to Hopsworks
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY", "").strip(),
        project=os.getenv("HOPSWORKS_PROJECT_NAME", "").strip(),
        host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
    )
    fs = project.get_feature_store()
    
    # 2. Fetch Data
    print("Fetching training data...")
    try:
        aqi_fg = fs.get_feature_group(name="aqi_readings", version=1)
        df = aqi_fg.select_all().read()
    except Exception as e:
        print(f"Failed to fetch feature group: {e}")
        return

    # 3. Data Preparation
    print("Preparing data...")
    df['date'] = pd.to_datetime(df['date'])
    df = create_targets(df)
    df = df.dropna() # Drop rows where targets are NaN (last 72h)
    
    # Define features - drop metadata, categorical identifiers, and targets
    drop_cols = ['date', 'city', 'y_24', 'y_48', 'y_72']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Extra safety: keep only numeric columns (handles any unexpected string columns)
    X_all = df[feature_cols]
    X_all = X_all.select_dtypes(include=[np.number])
    feature_cols = list(X_all.columns)
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    X = df[feature_cols]
    y = df[['y_24', 'y_48', 'y_72']]
    
    # Split Data (Time-based split, no shuffling)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Scaling (Important for Ridge and LSTM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for inference
    joblib.dump(scaler, 'scaler.pkl')

    # --- Model 1: Ridge Regression (MultiOutput) ---
    print("Training Ridge Regression...")
    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    print(f"Ridge RMSE: {rmse_ridge}")

    # --- Model 2: Random Forest ---
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1)
    rf.fit(X_train, y_train) # Tree models don't strictly need scaling
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    print(f"Random Forest RMSE: {rmse_rf}")

    # --- Model 3: LSTM (TensorFlow) ---
    print("Training LSTM...")
    # Reshape for LSTM: [samples, time steps, features]
    # Here we treat the input features as one time step for simplicity in this architecture, 
    # or better, we could use the lag features as a sequence if we reshaped upstream.
    # Given we have engineered lag features columns, we can just treat it as a dense input 
    # or reshape the lag columns into time steps. 
    # For simplicity/robustness in this script, we'll use a simple dense network or 
    # reshape X to (samples, 1, features) which is technically an RNN on the feature vector.
    
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
    model_lstm.add(Dense(3)) # Output 3 horizons
    model_lstm.compile(optimizer='adam', loss='mse')
    
    model_lstm.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
    y_pred_lstm = model_lstm.predict(X_test_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
    print(f"LSTM RMSE: {rmse_lstm}")

    # --- Comparison & Selection ---
    models = {
        "ridge": {"model": ridge, "rmse": rmse_ridge},
        "rf": {"model": rf, "rmse": rmse_rf},
        "lstm": {"model": model_lstm, "rmse": rmse_lstm}
    }
    
    best_model_name = min(models, key=lambda k: models[k]["rmse"])
    best_model = models[best_model_name]["model"]
    print(f"Best Model: {best_model_name} (RMSE: {models[best_model_name]['rmse']})")

    # --- Registration ---
    print(f"Registering {best_model_name} model...")
    mr = project.get_model_registry()
    
    # Create a local directory for artifacts
    model_dir = "model_artifacts"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        
    # Save model and scaler
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    
    if best_model_name == "lstm":
        best_model.save(f"{model_dir}/model.h5")
        framework = "tensorflow"
    else:
        joblib.dump(best_model, f"{model_dir}/model.pkl")
        framework = "sklearn"

    # Evaluate metrics on test set
    metrics = {
        "rmse": float(models[best_model_name]["rmse"]),
        "mae": float(mean_absolute_error(y_test, 
               y_pred_lstm if best_model_name == 'lstm' else (y_pred_ridge if best_model_name == 'ridge' else y_pred_rf)))
    }

    # Register model using Hopsworks Model Registry API
    aqi_model = mr.python.create_model(
        name="aqi_forecaster",
        metrics=metrics,
        description=f"Best model: {best_model_name}. RMSE: {metrics['rmse']:.2f}",
        input_example=X_test.iloc[0].to_dict()
    )
    
    aqi_model.save(model_dir)
    print(f"Model registered successfully! RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")

if __name__ == "__main__":
    train_model()
