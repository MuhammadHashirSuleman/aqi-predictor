import os
import sys
import hopsworks
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

def create_targets_per_city(df_city):
    """Create targets for a single city's data (sorted by date)."""
    df_city = df_city.sort_values('date').reset_index(drop=True)
    df_city['y_24'] = df_city['aqi'].shift(-24)
    df_city['y_48'] = df_city['aqi'].shift(-48)
    df_city['y_72'] = df_city['aqi'].shift(-72)
    return df_city

def prepare_data(df):
    """
    Strict per-city processing to avoid data leakage.
    1. Group by city
    2. Sort by date
    3. Create targets & lags (if not already present)
    4. Split 80/20 by time
    """
    train_pieces = []
    test_pieces = []
    
    # Process each city independently
    for city, group in df.groupby('city'):
        group = group.sort_values('date').reset_index(drop=True)
        
        # Create targets
        group = create_targets_per_city(group)
        
        # Drop rows where targets are NaN (last 72 hours)
        group = group.dropna(subset=['y_24', 'y_48', 'y_72'])
        
        if len(group) < 100:
            continue
            
        # Time-based split for this city
        split_idx = int(len(group) * 0.8)
        train_pieces.append(group.iloc[:split_idx])
        test_pieces.append(group.iloc[split_idx:])

    if not train_pieces:
        raise ValueError("No data available after processing!")

    train_df = pd.concat(train_pieces).reset_index(drop=True)
    test_df = pd.concat(test_pieces).reset_index(drop=True)
    
    return train_df, test_df

def train_model():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY", "").strip(),
        project=os.getenv("HOPSWORKS_PROJECT_NAME", "").strip(),
        host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
    )
    fs = project.get_feature_store()

    print("Fetching training data (Version 2)...")
    try:
        aqi_fg = fs.get_feature_group(name="aqi_readings", version=2)
        df = aqi_fg.select_all().read()
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Prepare data with strict per-city splitting
    print("Preparing data (per-city logic)...")
    df['date'] = pd.to_datetime(df['date'])
    train_df, test_df = prepare_data(df)
    
    # Select features
    drop_cols = {'date', 'city', 'y_24', 'y_48', 'y_72'}
    feature_cols = [c for c in train_df.columns 
                    if c not in drop_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df[['y_24', 'y_48', 'y_72']].values
    X_test = test_df[feature_cols].values
    y_test = test_df[['y_24', 'y_48', 'y_72']].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # ── Comparison of Models ──────────────────────────────────────────────────
    
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    models = {}
    
    # 1. Random Forest (Robust baseline)
    print("\n[1] Random Forest (n_estimators=100)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    models['rf'] = rf

    # 2. Gradient Boosting (Often best accuracy)
    print("\n[2] Gradient Boosting (sklearn)...")
    gb = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
    gb.fit(X_train, y_train)
    models['gb'] = gb
    
    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("RESULTS (R² Score - Higher is better, max 1.0)")
    print("="*50)
    
    best_name = None
    best_score = -999
    best_model = None
    best_metrics = {}

    for name, model in models.items():
        if name == 'rf':  # RF doesn't need scaling
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test) # GB also handles unscaled well usually, but consistency
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"{name.upper():5s} | R²: {r2:.4f}  | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        
        if r2 > best_score:
            best_score = r2
            best_name = name
            best_model = model
            best_metrics = {"r2": r2, "rmse": rmse, "mae": mae}

    print(f"\n✅ Winner: {best_name.upper()} with R²={best_score:.4f}")

    # ── Registration ──────────────────────────────────────────────────────────
    print("\nRegistering best model to Hopsworks...")
    mr = project.get_model_registry()
    
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, f"{model_dir}/model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    joblib.dump(feature_cols, f"{model_dir}/feature_cols.pkl")
    
    input_ex = {col: float(X_test[0][i]) for i, col in enumerate(feature_cols)}
    
    aqi_model = mr.python.create_model(
        name="aqi_forecaster",
        metrics=best_metrics,
        description=f"Best model: {best_name.upper()}. R²={best_score:.4f}",
        input_example=input_ex
    )
    aqi_model.save(model_dir)
    print("Model registered successfully!")

if __name__ == "__main__":
    train_model()
