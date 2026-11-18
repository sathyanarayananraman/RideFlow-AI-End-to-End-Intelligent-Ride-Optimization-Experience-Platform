# RideFlow AI — Intelligent Ride Optimization & Experience Platform

RideFlow AI is a machine-learning and deep-learning based ride-management prototype. It brings together ride-demand forecasting, driver-supply prediction, demand-based surge and Price estimation, cancellation-risk prediction, ETA estimation, demand-hotspot classification, and image-based driver-behaviour classification in a Streamlit dashboard.

The project uses **synthetic ride data** . The current user interface is a **Streamlit application**.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Workflow](#system-workflow)
- [Models and Evaluation](#models-and-evaluation)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [How to Use the Dashboard](#how-to-use-the-dashboard)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

## Project Overview

Ride-hailing services need to coordinate ride demand, available drivers, pricing, arrival-time estimates, cancellations, and driver safety. RideFlow AI demonstrates how predictive models can support several of these operational decisions through one interactive dashboard.

The project is organized into two implemented modelling areas:

- **Machine Learning:** demand prediction, driver-supply prediction, surge-multiplier prediction, and ride-cancellation classification.
- **Deep Learning:** ETA prediction using an LSTM, demand-hotspot classification using a neural network, and driver-behaviour classification from uploaded images using a CNN.

## Features

### 1. Price and ETA

- Select pickup and drop-off zones.
- Select weather and traffic conditions, date, and hour.
- Estimate trip duration and distance using Prediction.
- Predict ride demand and driver supply.
- Estimate a surge multiplier from the demand/supply context.
- Calculate the displayed fare using the application's base-fare and surge calculation.

### 2. Demand and Operations

- Forecast hourly ride demand.
- Predict driver availability.
- Classify demand hotspots into Low, Medium, and High levels.
- Estimate ride-cancellation probability .

### 3. Driver Behaviour

- Upload a JPG, JPEG, or PNG driver image.
- Display the predicted behaviour and model confidence.
- The dashboard presents the behaviour categories as:
  - Safe Driving
  - Phone Usage
  - Rash Driving

## System Workflow

```text
Synthetic Ride Data
        |
        v
Data Preparation and Feature Engineering
        |
        +-------------------------------+
        |                               |
        v                               v
Machine Learning Models             Deep Learning Models
        |                               |
        |-- Ride Demand                 |-- ETA (LSTM)
        |-- Driver Supply               |-- Demand Hotspot Classifier
        |-- Surge Multiplier            |-- Driver Behaviour (CNN)
        |-- Cancellation Risk           |
        +---------------+---------------+
                        |
                        v
              Saved Model Artifacts
                        |
                        v
                Streamlit Dashboard
                        |
                        v
       Interactive Predictions and Results
```

## Models and Evaluation

The following values are taken from the evaluation outputs in the supplied notebooks. They are model evaluation results on the notebook's data split.

### Machine Learning

| Task | Model | Evaluation metric | Result |
|---|---|---:|---:|
| Ride demand prediction | XGBoost Regressor | R² | 0.9453 |
| Ride demand prediction | XGBoost Regressor | RMSE | 14.7097 |
| Ride demand prediction | XGBoost Regressor | MAE | 10.7166 |
| Driver supply prediction | XGBoost Regressor | R² | 0.9378 |
| Driver supply prediction | XGBoost Regressor | RMSE | 12.4805 |
| Driver supply prediction | XGBoost Regressor | MAE | 9.1132 |
| Surge multiplier prediction | XGBoost Regressor | R² | 0.8449 |
| Surge multiplier prediction | XGBoost Regressor | RMSE | 0.0520 |
| Surge multiplier prediction | XGBoost Regressor | MAE | 0.0347 |
| Ride cancellation prediction | XGBoost Classifier | Accuracy | 93.62% |
| Ride cancellation prediction | XGBoost Classifier | F1-score | 0.8422 |
| Ride cancellation prediction | XGBoost Classifier | ROC-AUC | 0.9780 |

### Deep Learning

| Task | Model | Evaluation metric | Result |
|---|---|---:|---:|
| ETA prediction | LSTM | Test RMSE | 2.97 minutes |
| Demand hotspot classification | Dense neural network | Test accuracy | 82.81% |
| Driver behaviour classification | CNN | Dashboard output | Safe Driving, Phone Usage, Rash Driving |

The ETA notebook also reports a baseline RMSE of approximately 14.82 minutes for comparison with the LSTM result.

## Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Machine Learning:** XGBoost, scikit-learn
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy
- **Model and artifact loading:** Joblib, Pickle
- **Image handling:** Pillow (PIL)

## Repository Structure

```text
RideFlow_AI/
│
├── models/
│   ├── cancel_features.pkl
│   ├── demand_features.pkl
│   ├── eta_features.pkl
│   ├── hotspot_enc.pkl
│   ├── hotspot_features.pkl
│   ├── price_features.pkl
│   ├── supply_features.pkl
│   ├── cancel_model.pkl
│   ├── demand_model.pkl
│   ├── encoders.pkl
│   ├── eta_artifact.pkl
│   ├── eta_model.keras
│   ├── hotspot_artifact.pkl
│   ├── hotspot_model.keras
│   ├── price_model.pkl
│   ├── supply_model.pkl
│   ├── new_cnn_artifact.pkl
│   ├── new_driver_behavior_cnn.keras
│   └── zone_hour_history.pkl
│
├── source_code/
│   ├── ML_model.ipynb
│   ├── DL_model.ipynb
│   └── driver_cnnn.ipynb
│
└── streamlit.py
```

## Dataset

The project uses **synthetic ride data** for model development and demonstration.

The driver-behaviour workflow uses image data for CNN training and image uploads for dashboard inference.


## Installation

### Prerequisites

- Python installed
- Git
- The complete project folder, including the `models/` directory and its required artifacts

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RideFlow_AI
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository URL.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install the Python packages

```bash
pip install streamlit pandas numpy scikit-learn xgboost tensorflow pillow joblib
```

If your environment uses a different TensorFlow/Python combination, install a compatible TensorFlow version for that environment.

## Run the Application

From the project root directory, run:

```bash
streamlit run test.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the dashboard.

**Important:** Keep the `models/` directory in the same project root as `streamlit.py`. The application loads its saved models and preprocessing artifacts from this directory.

## How to Use the Dashboard

The sidebar provides four sections:

1. **Home** — overview of the RideFlow AI dashboard.
2. **Price & ETA** — provide trip context and request price and ETA estimates.
3. **Demand & Operations** — explore demand, driver supply, hotspot classification, and cancellation probability.
4. **Driver Behaviour** — upload a supported image file to view the behaviour category and confidence score.

## Limitations and Future Improvements

- The current project uses synthetic data; validation with representative real-world data is a necessary next step.
- Predictions depend on the quality and availability of the saved models, encoders, feature lists, and supporting artifacts.
- Demand and supply forecasting use historical/lag features. The application expects the corresponding `zone_hour_history.pkl` artifact to support those features.
- Driver-behaviour predictions should be treated as model outputs for demonstration, not as a definitive safety assessment.
- The current interface is Streamlit.

---

**Project:**  RideFlow AI — End-to-End Intelligent Ride Optimization & Experience Platform  
**Authour:**  Sathyanarayanan. Data Analyst | Business Intelligence | Python | SQL | Power BI | Machine Learning | Deep Learning
**Github:**   https://github.com/sathyanarayananraman/
**LinkedIn:** https://www.linkedin.com/in/sathyanarayanan2001/


