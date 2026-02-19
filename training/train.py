import os
import sys
import hopsworks
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create T+24, T+48, T+72 target columns."""
    df = df.sort_values('date').reset_index(drop=True)
    df['y_24'] = df['aqi'].shift(-24)
    df['y_48'] = df['aqi'].shift(-48)
    df['y_72'] = df['aqi'].shift(-72)
    return df

def evaluate(y_true, y_pred, name: str) -> dict:
    """Compute RMSE, MAE, R² and print them."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"  {name:20s} | RMSE={rmse:7.3f}  MAE={mae:7.3f}  R²={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    # 1. Connect
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY", "").strip(),
        project=os.getenv("HOPSWORKS_PROJECT_NAME", "").strip(),
        host=os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
    )
    fs = project.get_feature_store()

    # 2. Fetch
    print("Fetching training data...")
    try:
        # Try version 1 first (original backfill), fallback to version 2
        try:
            aqi_fg = fs.get_feature_group(name="aqi_readings", version=1)
            df = aqi_fg.select_all().read()
            if len(df) == 0:
                raise Exception("Empty Feature Group")
        except Exception:
            print("  Trying version 2...")
            aqi_fg = fs.get_feature_group(name="aqi_readings", version=2)
            df = aqi_fg.select_all().read()
    except Exception as e:
        print(f"Failed to fetch feature group: {e}")
        return

    print(f"Loaded {len(df)} rows.")

    # 3. Prepare
    print("Preparing data...")
    df['date'] = pd.to_datetime(df['date'])
    df = create_targets(df)
    df = df.dropna()

    # Keep only numeric features
    drop_cols = {'date', 'city', 'y_24', 'y_48', 'y_72'}
    feature_cols = [c for c in df.columns
                    if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].values
    y = df[['y_24', 'y_48', 'y_72']].values

    # Time-based split (80 / 20)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    results = {}

    # ── Model 1: Ridge Regression ──────────────────────────────────────────
    print("\n[1] Ridge Regression")
    ridge_params = {'estimator__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge_cv = GridSearchCV(
        MultiOutputRegressor(Ridge()),
        ridge_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )
    ridge_cv.fit(X_train_s, y_train)
    ridge = ridge_cv.best_estimator_
    print(f"  Best alpha: {ridge_cv.best_params_}")
    y_pred_ridge = ridge.predict(X_test_s)
    results['ridge'] = {**evaluate(y_test, y_pred_ridge, "Ridge"), 'model': ridge, 'preds': y_pred_ridge}

    # ── Model 2: Random Forest ─────────────────────────────────────────────
    print("\n[2] Random Forest")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=2,
        max_features='sqrt', n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['rf'] = {**evaluate(y_test, y_pred_rf, "Random Forest"), 'model': rf, 'preds': y_pred_rf}

    # ── Model 3: Gradient Boosting ─────────────────────────────────────────
    print("\n[3] Gradient Boosting (XGBoost-style)")
    gb = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ), n_jobs=-1
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results['gb'] = {**evaluate(y_test, y_pred_gb, "Gradient Boosting"), 'model': gb, 'preds': y_pred_gb}

    # ── Model 4: LSTM ──────────────────────────────────────────────────────
    print("\n[4] LSTM (TensorFlow)")
    X_train_lstm = X_train_s.reshape(X_train_s.shape[0], 1, X_train_s.shape[1])
    X_test_lstm  = X_test_s.reshape(X_test_s.shape[0],  1, X_test_s.shape[1])

    lstm_model = Sequential([
        Input(shape=(1, X_train_s.shape[1])),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3)
    ])
    lstm_model.compile(optimizer='adam', loss='huber')
    es = EarlyStopping(patience=5, restore_best_weights=True)
    lstm_model.fit(
        X_train_lstm, y_train,
        epochs=50, batch_size=64,
        validation_split=0.1,
        callbacks=[es], verbose=0
    )
    y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0)
    results['lstm'] = {**evaluate(y_test, y_pred_lstm, "LSTM"), 'model': lstm_model, 'preds': y_pred_lstm}

    # ── Select Best ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name:20s} | R²={r['r2']:.4f}  RMSE={r['rmse']:.3f}")

    best_name = max(results, key=lambda k: results[k]['r2'])
    best = results[best_name]
    print(f"\n✅ Best Model: {best_name}  (R²={best['r2']:.4f})")

    # ── Save & Register ────────────────────────────────────────────────────
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    joblib.dump(feature_cols, f"{model_dir}/feature_cols.pkl")

    if best_name == 'lstm':
        best['model'].save(f"{model_dir}/model.keras")
    else:
        joblib.dump(best['model'], f"{model_dir}/model.pkl")

    # Register in Hopsworks Model Registry
    print("\nRegistering model in Hopsworks...")
    mr = project.get_model_registry()
    metrics = {"rmse": best['rmse'], "mae": best['mae'], "r2": best['r2']}

    aqi_model = mr.python.create_model(
        name="aqi_forecaster",
        metrics=metrics,
        description=f"Best model: {best_name} | R²={best['r2']:.4f} | RMSE={best['rmse']:.2f}",
        input_example={f: float(X_test[0, i]) for i, f in enumerate(feature_cols)}
    )
    aqi_model.save(model_dir)
    print(f"✅ Model registered! R²={best['r2']:.4f}  RMSE={best['rmse']:.2f}  MAE={best['mae']:.2f}")

if __name__ == "__main__":
    train_model()
