# 🛠️ Setup Guide - Pearls AQI Predictor

Complete step-by-step instructions for setting up the AQI Predictor locally and configuring GitHub Actions.

## 📋 Prerequisites

- Python 3.10+
- Git
- GitHub account (for CI/CD)
- Hopsworks account (free tier: https://app.hopsworks.ai)
- AQICN API key (free: https://aqicn.org/data-platform/token/)

---

## 🖥️ Local Development Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/MuhammadHashirSuleman/aqi-predictor.git
cd aqi-predictor
```

### Step 2: Create Python Environment

**Using Conda (Recommended):**
```bash
conda create -n aqi-env python=3.10 -y
conda activate aqi-env
```

**Using venv:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements-ci.txt
pip install streamlit fastapi uvicorn plotly tensorflow shap
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Hopsworks Configuration
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=your_project_name
HOPSWORKS_HOST=c.app.hopsworks.ai

# AQICN API
AQICN_API_KEY=your_aqicn_token

# Optional
OPENWEATHER_API_KEY=your_openweather_key
API_URL=http://localhost:8000
```

#### Getting API Keys:

1. **Hopsworks**:
   - Sign up at https://app.hopsworks.ai
   - Create a new project
   - Go to Account Settings → API Keys → Create new key
   - Copy the API key, project name, and host

2. **AQICN**:
   - Go to https://aqicn.org/data-platform/token/
   - Request a free API token
   - Token will be emailed to you

---

## 🚀 Running the Application

### First-Time Setup

#### 1. Backfill Historical Data
This populates the Feature Store with training data (run once):

```bash
python features/backfill.py
```

Expected output:
```
Fetching current data for beijing...
Fetching current data for karachi...
...
Total rows: 35040
Inserting rows...
✅ Backfill complete!
```

#### 2. Train the Model
This trains and registers the model (run once, then daily via CI/CD):

```bash
python training/train.py
```

Expected output:
```
Connecting to Hopsworks...
Fetching training data...
TRAINING MODELS
[1] Random Forest...
[2] Gradient Boosting...
RESULTS
RF    | R²: 0.85  | RMSE: 28.5 | MAE: 21.2
GB    | R²: 0.87  | RMSE: 26.3 | MAE: 19.8
✅ Winner: GB with R²=0.87
Model registered successfully!
```

### Running the App

#### Terminal 1 - Start Backend API:
```bash
uvicorn app.api:app --reload --port 8000
```

#### Terminal 2 - Start Dashboard:
```bash
streamlit run app/dashboard.py
```

#### Access the Application:
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## ⚙️ GitHub Actions Setup

### Step 1: Add Repository Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `HOPSWORKS_API_KEY` | Your Hopsworks API key |
| `HOPSWORKS_PROJECT_NAME` | Your Hopsworks project name |
| `HOPSWORKS_HOST` | `c.app.hopsworks.ai` |
| `AQICN_API_KEY` | Your AQICN API token |
| `OPENWEATHER_API_KEY` | (Optional) OpenWeather API key |

### Step 2: Enable GitHub Actions

Actions are enabled by default. The workflows are in `.github/workflows/`:

- **`feature_pipeline.yml`** - Runs every hour
- **`training_pipeline.yml`** - Runs daily at 2 AM UTC

### Step 3: Manual Trigger (Testing)

1. Go to Actions tab in your repository
2. Select "Feature Pipeline (Hourly)" or "Training Pipeline (Daily)"
3. Click "Run workflow" → "Run workflow"

### Workflow Schedules

| Workflow | Cron | Description |
|----------|------|-------------|
| Feature Pipeline | `0 * * * *` | Every hour at minute 0 |
| Training Pipeline | `0 2 * * *` | Daily at 2:00 AM UTC |

---

## 🧪 Testing

### Test Feature Pipeline Locally:
```bash
python features/feature_pipeline.py
```

### Test Training Pipeline Locally:
```bash
python training/train.py
```

### Test API Endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Get predictions
curl http://localhost:8000/predict
```

---

## 🔧 Troubleshooting

### "Model not loaded" Error
- Ensure you've run `python training/train.py` at least once
- Check if model exists in Hopsworks Model Registry
- Verify `HOPSWORKS_API_KEY` is correct

### "Could not fetch live AQI data"
- Verify `AQICN_API_KEY` in `.env`
- Check if AQICN API is accessible

### GitHub Actions Failing
- Check if all secrets are configured
- View workflow logs for specific errors
- Ensure `requirements-ci.txt` has all dependencies

### Hopsworks Connection Issues
- Verify API key hasn't expired
- Check project name is correct
- Ensure host is `c.app.hopsworks.ai`

### Disk Space Issues (Local)
```bash
# Clear pip cache
pip cache purge

# Clear conda cache
conda clean --all
```

---

## 📁 File Descriptions

| File | Purpose |
|------|--------|
| `features/feature_pipeline.py` | Fetches live AQI data, computes features, stores in Hopsworks |
| `features/backfill.py` | Generates historical data for training |
| `features/aqi_utils.py` | Utility functions for data processing |
| `training/train.py` | Trains ML models, registers best model |
| `inference/predictor.py` | Loads model, makes predictions |
| `app/api.py` | FastAPI backend for predictions |
| `app/dashboard.py` | Streamlit frontend dashboard |
| `notebooks/eda.ipynb` | Exploratory Data Analysis |
| `notebooks/shap_analysis.ipynb` | SHAP feature importance analysis |

---

## 🌐 Deployment Options

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your repository
4. Add secrets in Streamlit Cloud settings
5. Deploy!

### Railway / Render
1. Create new project
2. Connect GitHub repository
3. Set environment variables
4. Deploy both API and dashboard

---

## 📞 Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Review GitHub Actions logs
3. Open an issue on the repository

---

**Happy Forecasting! 🌫️**

# AQI Predictor - Setup & Execution Guide

This guide will walk you through setting up the project using **Anaconda** and configuring the necessary **API keys**.

## 1. Environment Setup (Anaconda)

Open your **Anaconda Prompt** or **Terminal** and follow these steps:

### Create a New Environment
Create a clean environment named `aqi-env` with Python 3.10:
```bash
conda create -n aqi-env python=3.10 -y
```

### Activate the Environment
```bash
conda activate aqi-env
```

### Install Dependencies
Navigate to the project directory (if not already there) and install the required libraries:
```bash
cd /d h:\AQI
pip install -r requirements.txt
```

---

## 2. API Key Configuration

You need 3 API keys to run this system.

### A. AQICN (Air Quality Data)
1. Go to [https://aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/)
2. Enter your email and name.
3. Copy the token sent to your email.

### B. OpenWeatherMap (Weather Data)
1. Go to [https://home.openweathermap.org/users/sign_up](https://home.openweathermap.org/users/sign_up)
2. Sign up and go to the **API Keys** tab.
3. Copy your "Default" key.

### C. Hopsworks (Feature Store & Model Registry)
1. Go to [https://www.hopsworks.ai/](https://www.hopsworks.ai/) and sign up (Free).
2. Create a new **Project** (e.g., `aqi_project`).
3. Click on your user profile (top right) -> **Settings** -> **API Keys**.
4. Create a new key and check the following scopes:
   - [x] **PROJECT**
   - [x] **FEATURESTORE**
   - [x] **MODELREGISTRY**
   - [x] **JOB**
   - [x] **DATASET_CREATE**
   - [x] **DATASET_VIEW**
5. Copy the API Key.

### D. Configure `.env` File
1. In the project folder `h:\AQI`, you will see a file named `.env.example`.
2. Rename it to `.env`:
   ```bash
   ren .env.example .env
   ```
3. Open `.env` in a text editor (Notepad, VS Code) and paste your keys:
   ```ini
   HOPSWORKS_API_KEY=your_pasted_hopsworks_key_here
   HOPSWORKS_PROJECT_NAME=aqi_project
   AQICN_API_KEY=your_pasted_aqicn_key_here
   OPENWEATHER_API_KEY=your_pasted_openweather_key_here
   API_URL=http://localhost:8000
   ```

---

## 3. Running the System

Execute these scripts in order:

### Step 1: Initialize Feature Store (Backfill)
Downloads 1 year of historical data and populates Hopsworks.
```bash
python features/backfill.py
```
*Wait for this to complete (can take a few minutes).*

### Step 2: Train Models
Trains Ridge, Random Forest, and LSTM models and registers the best one.
```bash
python training/train.py
```

### Step 3: Start the Web Application
You need **two** terminals for this (or run one in background).

**Terminal 1 (Backend API):**
```bash
uvicorn app.api:app --reload
```

**Terminal 2 (Frontend Dashboard):**
```bash
streamlit run app/dashboard.py
```
Open the URL shown (usually `http://localhost:8501`) to view your predictions!
