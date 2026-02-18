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
