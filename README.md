# Pearls AQI Predictor

A fully automated, serverless MLOps system for 3-day AQI forecasting.

## Architecture

- **Feature Store**: Hopsworks
- **Model Training**: GitHub Actions + Hopsworks
- **Inference**: Serverless Function / Container
- **Frontend**: Streamlit
- **Backend**: FastAPI

## Setup
> **See [SETUP.md](SETUP.md) for detailed instructions.**

1. Clone repository.
2. Create Anaconda environment: `conda create -n aqi-env python=3.10`
3. Install dependencies: `pip install -r requirements.txt`
4. Set up `.env` with API keys (see `.env.example`).
5. Run Streamlit App: `streamlit run app/dashboard.py`

## Directory Structure

- `data/`: Raw data storage (local cache)
- `features/`: Feature engineering pipelines
- `training/`: Model training scripts
- `inference/`: Inference scripts
- `app/`: Web application code
- `ci_cd/`: GitHub Actions workflows
- `notebooks/`: EDA and SHAP analysis
