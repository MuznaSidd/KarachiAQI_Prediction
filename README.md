# Pearls AQI Predictor

Default City: Karachi

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  ![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)

**Objective**: To predict and monitor Air Quality Index (AQI) of Karachi for three days using end-to-end MLOps integration.

---

## 📖 Overview

Pearls AQI Predictor is a fully automated Air Quality Index (AQI) forecasting solution built for **Karachi**.

The system:

* Fetches historical and live aqi pollutants + weather data using **Openmeteo api**
* Computes AQI using official EPA standards
* Performs data validation and cleaning
* Engineers time-series features and uploads the features into Hopsworks feature store
* Trains multiple ML models (Ridge,Xgboost,Randomforest) for 1–3 day AQI forecasting
* Register all models in Hopsworks Model Registry
* Deploy predictions using Streamlit
* Automates feature and training pipeline using GitHub Actions

The entire workflow follows production-level MLOps practices.

---

## System Architecture

### Feature Pipeline (Hourly / Incremental)

* Fetches pollutant and weather data from Open meteo
* Performs one-year historical backfill (initial setup)
* Computes AQI using EPA breakpoint tables
* Applies data cleaning and validation
* Uploads the features into feature store
* Engineers time-based features
* Inserts only new records afterwards into Feature Store to avoid duplication of data

---

### Training Pipeline (Daily)

* Pulls latest features from Feature Store
* Applies time-series aware split
* Trains multiple ml models
* Evaluates using RMSE,MAE and R²
* Selects best model per forecast day
* Registers models in Hopsworks Model Registry with proper versioning

Three Models are trained for:

* Day 1 Forecast
* Day 2 Forecast
* Day 3 Forecast

---

### Streamlit Dashboard

* Loads latest BEST model for each day automatically
* Displays 1, 2, 3 day AQI forecasts
* Includes EDA visualizations
* Fetches latest data dynamically

---

### Automation (CI/CD)

* Hourly Feature Pipeline via GitHub Actions
* Daily Training Pipeline automation
* Automatic model versioning

---

## 🧠 Models Used

| Model                 | Purpose                 | Framework    | Role in Pipeline           |
| --------------------- | ----------------------- | ------------ | -------------------------- |
| Ridge                 | AQI Forecast (1–3 days) | Scikit-learn | Baseline Linear Model      |
| RandomForestRegressor | AQI Forecast (1–3 days) | Scikit-learn | BEST Model (Day 1 & Day 3) |
| XGBoostRegressor      | AQI Forecast (1–3 days) | XGBoost      | BEST Model (Day 2)         |


---

## 📊 Performance Metrics (Hopsworks integrated)

| Forecast Day | Best Model    | RMSE  | MAE   | R²    |
| ------------ | ------------- | ----- | ----- | ----- |
| Day 1        | Random Forest | 28.69 | 22.81 | 0.743 |
| Day 2        | XGBoost       | 30.33 | 24.45 | 0.716 |
| Day 3        | Random Forest | 30.02 | 24.48 | 0.723 |

All metrics are automatically logged and synced from the Hopsworks Model Registry.
The best-performing model for each forecast horizon is selected based on the lowest RMSE value.


## 💡 Explainability

* **SHAP** used for global feature importance
* Identifies most influential pollutants and weather parameters

---

## ⚠️ AQI Hazard Levels

| AQI Range | Category                       |
| --------- | ------------------------------ |
| 0–50      | Good                           |
| 51–100    | Moderate                       |
| 101–150   | Unhealthy for Sensitive Groups |
| 151–200   | Unhealthy                      |
| 201–300   | Very Unhealthy                 |
| 301–500   | Hazardous                      |

AQI calculated using official EPA interpolation rules.

---

## 🧩 Tech Stack

* **Language:** Python 3.10.11
* **ML:** Scikit-learn, XGBoost
* **Visualization:** Streamlit, Matplotlib
* **Explainability:** SHAP
* **MLOps:** Hopsworks Feature Store & Model Registry
* **Automation:** GitHub Actions (CI/CD)

---

## 🧱 Folder Structure

```
AQI/
│
├── .github/workflows/
│   ├── feature_pipeline.yml
│   └── training_pipeline.yml
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── processed/
│   │   └── aq_weather_clean.csv
│   ├── raw/
│   │   └── aq_weather_raw.csv
│
├── notebooks/
│   ├── eda_outputs/
│   ├── eda/
│   ├── feature_store eda.ipynb
│   ├── shap_analysis/
│   ├── shap_plots/
│   └── shap.ipynb
│
├── pipelines/
│   ├── backfill_data.py
│   ├── feature_pipeline.py
│
├── src/
│   ├── features/
│   │   ├── data_cleaning.py
│   │   ├── data_fetcher.py
│   ├── training/
│   │   ├── data_quality_check.py
│   │   ├── training_pipeline.py
│   ├── utils/
│   │   ├── aqi_calculator.py
│   │   └── test_hopsworks.py
│
├── .env
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 🌐 Live Deployment

🚀 **Streamlit App:**

```
https://muzna-karachi-aqi-predictor.streamlit.app/
```

---

## 🧰 Installation Guide

**Step 1: Create Virtual Environment**

```
python -m venv .env
.env\Scripts\Activate.ps1
```

**Step 2: Install Dependencies**

```
pip install -r requirements.txt
```

## 🚀 Usage Instructions

**Run Feature Pipeline**

```
python -m pipelines.feature_pipeline
```

**Run Training Pipeline**

```
python src/training/training_pipeline.py
```

**Launch Streamlit Dashboard**

```
streamlit run app/streamlit_app.py
```

---

## ☁️ Hopsworks Integration

Set in `.env`:

```
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT_NAME=your_project_name

CITY_NAME=Karachi
LATITUDE=24.8607
LONGITUDE=67.0011
```
---

## 🌐 Visual Preview Of The Dashboard

![aproject final final output](https://github.com/user-attachments/assets/46f9eb6c-147a-476f-8ff7-0f887226cae2)
![aproject ff output 2](https://github.com/user-attachments/assets/b2075e7c-1ec8-4af5-8315-d4455eb0b7b6)


## Results & Achievements

* One-year historical backfill of AQI pollutants+weather data
* EPA-based AQI computation
* Features stored into Hopsworks feature store
* Models uploaded into Model Registry with proper versioning
* Multi-model AQI forecasting (1–3 days)
* AQI Forecast and EDA dashboard
* CI/CD automation
* Explainable AI integration
* End-to-end MLOps implementation

---

# Author

**Muzna Siddiqui**

Karachi, Pakistan

AQI Forecasting System For Karachi– End-to-End MLOps Project

---
