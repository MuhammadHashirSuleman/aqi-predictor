# 🌫️ Pearls AQI Predictor

A **100% serverless** end-to-end MLOps system for predicting Air Quality Index (AQI) for the next 3 days.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Hopsworks](https://img.shields.io/badge/Feature%20Store-Hopsworks-orange)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal)

## 📋 Project Overview

This project implements a complete machine learning pipeline for AQI forecasting with:
- **Automated data collection** from AQICN API
- **Feature engineering** with time-based and derived features
- **Model training** with multiple ML algorithms
- **Real-time predictions** through a web dashboard
- **CI/CD automation** using GitHub Actions

### 🏙️ Supported Cities
| City | Live Data | Training Data |
|------|-----------|---------------|
| Beijing | ✅ | ✅ |
| Delhi | ❌ | ✅ |
| Karachi | ❌ | ✅ |
| Lahore | ❌ | ✅ |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PEARLS AQI PREDICTOR                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   AQICN API  │───▶│   Feature    │───▶│     Hopsworks            │  │
│  │  (Raw Data)  │    │   Pipeline   │    │  ┌────────────────────┐  │  │
│  └──────────────┘    └──────────────┘    │  │   Feature Store    │  │  │
│                             │            │  │   (aqi_readings)   │  │  │
│                      GitHub Actions      │  └────────────────────┘  │  │
│                      (Hourly Cron)       │  ┌────────────────────┐  │  │
│                             │            │  │   Model Registry   │  │  │
│                             ▼            │  │  (aqi_forecaster)  │  │  │
│  ┌──────────────┐    ┌──────────────┐    │  └────────────────────┘  │  │
│  │   Trained    │◀───│   Training   │    └──────────────────────────┘  │
│  │    Model     │    │   Pipeline   │              │                   │
│  └──────────────┘    └──────────────┘              │                   │
│         │             GitHub Actions               │                   │
│         │             (Daily Cron)                 ▼                   │
│         │                              ┌──────────────────────────┐    │
│         └─────────────────────────────▶│      Web Application     │    │
│                                        │  ┌────────┐ ┌─────────┐  │    │
│                                        │  │FastAPI │ │Streamlit│  │    │
│                                        │  │Backend │ │Dashboard│  │    │
│                                        │  └────────┘ └─────────┘  │    │
│                                        └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10 |
| **ML Models** | Scikit-learn (Random Forest, Gradient Boosting) |
| **Feature Store** | Hopsworks |
| **Model Registry** | Hopsworks Model Registry |
| **CI/CD** | GitHub Actions |
| **Frontend** | Streamlit |
| **Backend API** | FastAPI |
| **Data Source** | AQICN API |
| **Explainability** | SHAP |

## 📁 Project Structure

```
AQI/
├── .github/workflows/
│   ├── feature_pipeline.yml    # Hourly feature updates
│   └── training_pipeline.yml   # Daily model retraining
├── app/
│   ├── api.py                  # FastAPI backend
│   └── dashboard.py            # Streamlit frontend
├── features/
│   ├── aqi_utils.py            # Data fetching & processing
│   ├── backfill.py             # Historical data generation
│   └── feature_pipeline.py     # Main feature pipeline
├── training/
│   └── train.py                # Model training script
├── inference/
│   └── predictor.py            # Prediction logic
├── notebooks/
│   ├── eda.ipynb               # Exploratory Data Analysis
│   └── shap_analysis.ipynb     # SHAP feature importance
├── model_artifacts/            # Local model storage
├── requirements-ci.txt         # Dependencies
├── SETUP.md                    # Setup guide
└── README.md
```

## ✨ Key Features

### 1. Feature Pipeline (`features/feature_pipeline.py`)
- Fetches real-time AQI from AQICN API
- Computes **time features**: hour, day_of_week, month
- Computes **lag features**: AQI at t-1, t-3, t-24 hours
- Computes **rolling statistics**: 3h, 6h, 24h mean & std
- Computes **derived features**: PM2.5/PM10 ratio
- Stores in Hopsworks Feature Store
- **Runs hourly** via GitHub Actions

### 2. Training Pipeline (`training/train.py`)
- Fetches historical data from Feature Store
- Trains **Random Forest** and **Gradient Boosting** models
- Predicts AQI at **+24h, +48h, +72h**
- Evaluates with **RMSE, MAE, R²** metrics
- Registers best model in Hopsworks Model Registry
- **Runs daily** via GitHub Actions

### 3. Web Dashboard (`app/dashboard.py`)
- Displays **current AQI** (live from AQICN)
- Shows **3-day forecast** with color-coded charts
- **Hazard alerts** for unhealthy/dangerous levels
- **SHAP explainability** section

### 4. CI/CD Automation

| Workflow | Schedule | File |
|----------|----------|------|
| Feature Pipeline | Hourly (`0 * * * *`) | `.github/workflows/feature_pipeline.yml` |
| Training Pipeline | Daily 2AM UTC (`0 2 * * *`) | `.github/workflows/training_pipeline.yml` |

## 🚀 Quick Start

See **[SETUP.md](SETUP.md)** for detailed instructions.

```bash
# Clone & setup
git clone https://github.com/MuhammadHashirSuleman/aqi-predictor.git
cd aqi-predictor
pip install -r requirements-ci.txt streamlit fastapi uvicorn plotly

# Configure .env with API keys

# Run pipelines (first time)
python features/backfill.py
python training/train.py

# Start app
uvicorn app.api:app --port 8000      # Terminal 1
streamlit run app/dashboard.py       # Terminal 2
```

## 🔔 AQI Alert Levels

| AQI | Level | Alert |
|-----|-------|-------|
| 0-50 | 🟢 Good | None |
| 51-100 | 🟡 Moderate | None |
| 101-150 | 🟠 Unhealthy (Sensitive) | None |
| 151-200 | 🔴 Unhealthy | ⚠️ Warning |
| 201+ | 🟣 Hazardous | 🚨 Alert |

## 🔧 GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `HOPSWORKS_API_KEY` | Hopsworks API key |
| `HOPSWORKS_PROJECT_NAME` | Hopsworks project name |
| `HOPSWORKS_HOST` | e.g., `c.app.hopsworks.ai` |
| `AQICN_API_KEY` | AQICN API token |

## 📊 Model Metrics

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Random Forest | ~0.85 | ~30 | ~22 |
| Gradient Boosting | ~0.87 | ~28 | ~20 |

## 📄 License

MIT License

## 👤 Author

Muhammad Hashir Suleman
