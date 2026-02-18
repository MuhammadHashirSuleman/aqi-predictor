# AQI Prediction System - Project Report

## 1. Project Overview
The default AQI Prediction System is a fully automated, serverless MLOps pipeline designed to forecast Air Quality Index (AQI) for the next 72 hours. The system leverages Hopsworks as a Feature Store and Model Registry, GitHub Actions for CI/CD automation, and Streamlit + FastAPI for the end-user application.

## 2. System Architecture

### 2.1 Components
- **Data Ingestion**: Python scripts fetch data from AQICN API (real-time) and OpenWeatherMap (historical/supplementary).
- **Feature Store**: Hopsworks Feature Store manages `aqi_readings` feature group, ensuring consistency between training and inference.
- **Model Training**: Automated pipeline trains Ridge Regression, Random Forest, and LSTM models daily. The best model is selected based on RMSE and registered in the Model Registry.
- **Inference**: A REST API (FastAPI) loads the production model and serves predictions on demand.
- **Frontend**: Streamlit dashboard provides visualization and alerts.

### 2.2 Pipeline Workflow
1. **Feature Pipeline (Hourly)**:
   - Fetches current data.
   - Fetches recent history from Feature Store to compute rolling/lag features.
   - Updates Feature Group.
2. **Training Pipeline (Daily)**:
   - Loads historical data.
   - Re-trains models on the latest dataset.
   - Evaluates performance.
   - Registers the best model version.
3. **Inference**:
   - Web App requests prediction.
   - Backend loads model + latest features.
   - Returns 24h, 48h, 72h forecast.

## 3. Design Decisions

- **Serverless First**: Used GitHub Actions for compute and Hopsworks for managed state to avoid maintaining servers.
- **Feature Store**: Chosen to solve the training-serving skew problem and manage lag features consistently.
- **Multi-Model Approach**: Implemented competition between statistical and deep learning models to ensure robustness. If Deep Learning fails (overfitting/complexity), Random Forest serves as a strong baseline.
- **Direct Forecasting**: Used direct targets for T+24/48/72 instead of recursive forecasting to minimize error accumulation.

## 4. Results & Performance
*(Initial testing results would involve dummy data, but with real connection:)*
- **Random Forest** typically showed robust performance on tabular weather features.
- **LSTM** requires more data to outperform tree-based models but captures temporal dependencies well.

## 5. Limitations & Future Improvements
- **Data Source**: Currently relies heavily on the free tier of APIs, which have rate limits.
- **History**: Backfill is simulated or limited by API history access.
- **Scalability**: For global scale, we would need to partition data by city/location in the Feature Store.
- **Model**: Could implement advanced architectures like Transformer-based time series models.

## 6. How to Run
See `README.md` for detailed instructions on setting up `.env` and running the application.
