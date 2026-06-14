# 💳 Credit Risk Prediction Pipeline

> End-to-end machine learning pipeline for predicting credit card default using an ensemble of XGBoost, Random Forest, and neural networks. Deployed via FastAPI with a Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2)](https://mlflow.org)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Modeling](#modeling)
- [Results](#results)
- [Deployment](#deployment)
- [MLflow Tracking](#mlflow-tracking)
- [Tech Stack](#tech-stack)

---

## Overview

This project predicts whether a credit card client will default next month, using the UCI Credit Card Default dataset. It implements two modeling approaches:

| Approach | AUC-ROC | Key Characteristics |
|----------|---------|---------------------|
| **Ensemble (4 models)** | **0.781** | XGBoost + Random Forest + 2 MLPs, simple average |
| **Vanilla MLP** | 0.777 | Focal Loss, 5-fold CV, Optuna-tuned |

The ensemble achieved the best recall (0.63) and F1 (0.546), making it the recommended model for production use.

---

## Features

### Data & Preprocessing
- ✅ 10 engineered features from 24 raw attributes
- ✅ RobustScaler normalization on continuous columns
- ✅ Class imbalance handling with Focal Loss & stratified CV
- ✅ 70/15/15 stratified train/val/test split

### Modeling
- ✅ 4-model ensemble (XGBoost, RF, MLP-A, MLP-B) with simple averaging
- ✅ Vanilla MLP baseline with Focal Loss
- ✅ 5-fold stratified cross-validation
- ✅ Optuna hyperparameter optimization
- ✅ OOF-calibrated decision thresholds
- ✅ Standardized metrics across all models

### Deployment
- ✅ FastAPI REST API with model selection (`?model=ensemble|vanilla`)
- ✅ Streamlit dashboard with interactive inputs
- ✅ Real-time predictions with model vote breakdown
- ✅ Pre-computed evaluation metrics, confusion matrix, and feature importance
- ✅ Scenario walkthroughs for each confusion-matrix quadrant

### Tracking
- ✅ MLflow experiment tracking (metrics, hyperparameters, artifacts)
- ✅ Reproducible runs with logged configurations

---

## Dataset

**Source:** [UCI Credit Card Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)  
**Samples:** 30,000 credit card clients from Taiwan (April–Sept 2005)  
**Features:** 24 raw → 10 engineered after EDA  
**Target:** Default payment next month (binary: 78% no-default, 22% default)

### Final Feature Set

| Feature | Description | Scaling |
|---------|-------------|---------|
| `LIMIT_BAL` | Credit limit (NTD) | RobustScaler |
| `PAY_SEP` … `PAY_APR` | Repayment status (-2 to 8) — Sep through Apr | None (ordinal) |
| `BILL_SEP` | September bill amount (NTD) | RobustScaler |
| `PAYMENT_SEP` | September payment amount (NTD) | RobustScaler |
| `TOTAL_DELAY_DAYS` | Sum of positive delays × 30 (engineered) | RobustScaler |

**Dropped during EDA:** `AGE`, `SEX`, `MARRIAGE`, `EDUCATION`, correlated bill/payment months, redundant delay features.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for MLflow server)

### Installation

```bash
# Clone the repository
git clone https://github.com/shaswot84/credit-risk-pipeline.git
cd credit-risk-pipeline

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync

# (Optional) Start MLflow tracking server
docker run -d \
  -p 5000:5000 \
  -v ~/mlflow-data:/mlflow \
  --name mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server --host 0.0.0.0 --port 5000
```

### Run the Complete Pipeline

```bash
# 1. EDA + preprocessing
uv run jupyter notebook data-mining/EDA+FE+cleaning.ipynb

# 2. Train vanilla MLP (~2 min)
uv run jupyter notebook data-mining/building-nns/local-notebooks/model-building-test.ipynb

# 3. Train ensemble (~10 min)
uv run jupyter notebook data-mining/building-nns/local-notebooks/stacking-ensemble.ipynb
```

### Start the API & Frontend

```bash
# Terminal 1: Backend API
uv run uvicorn backend.main:app --reload --port 8000

# Terminal 2: Streamlit frontend
uv run streamlit run frontend/app.py
```

- **API:** http://localhost:8000 — interactive docs at `/docs`
- **Frontend:** http://localhost:8501
- **MLflow UI:** http://localhost:5000

---

## Project Structure

```
credit-risk-pipeline/
├── backend/
│   ├── main.py                    # FastAPI app — model serving, prediction, metrics
│   ├── test_load.py               # Model-load sanity check
│   └── requirements.txt           # Backend-specific dependencies
├── frontend/
│   └── app.py                     # Streamlit dashboard — prediction UI, metrics, plots
├── data-mining/
│   ├── EDA+FE+cleaning.ipynb      # Full pipeline: EDA -> feature engineering -> split -> export
│   ├── building-nns/
│   │   ├── CONTEXT.md             # Modeling-specific context & architecture decisions
│   │   ├── colab-notebooks/       # Google Colab-ready versions
│   │   │   ├── model-building-test.ipynb
│   │   │   └── stacking-ensemble.ipynb
│   │   ├── local-notebooks/       # Local-only versions (hardcoded paths)
│   │   │   ├── model-building-test.ipynb
│   │   │   └── stacking-ensemble.ipynb
│   │   └── output/
│   │       ├── vanilla/           # Vanilla MLP: best_model.pt, results.csv
│   │       └── stacking-ensemble/ # Ensemble: xgb_final.pkl, rf_final.pkl, mlp_a/b_final.pt, config.pkl, metrics.csv
│   ├── graphs/
│   │   └── shapPLOT.png           # SHAP feature importance visualization
│   └── data/                      # Additional data-mining artifacts
├── data/
│   ├── raw/
│   │   └── UCI_Credit_Card.csv.zip  # Original dataset
│   └── processed/                 # Preprocessed data
│       ├── train.csv              # 21,000 samples (70%)
│       ├── val.csv                # 4,500 samples (15%)
│       ├── test.csv               # 4,500 samples (15%)
│       └── scaler.pkl             # Fitted RobustScaler
├── model/                         # (reserved for future model storage)
├── .venv/                         # Python virtual environment
├── .gitignore
├── .python-version                # Python 3.12
├── pyproject.toml                 # Project config + dependencies
├── uv.lock
├── CLAUDE.md                      # AI assistant context
├── CONTEXT.md                     # Developer handoff / project status
├── master-plan.txt                # Initial development plan
├── LICENSE
└── README.md                      # This file
```

---

## Data Pipeline

Key decisions made during preprocessing:

- **RobustScaler** chosen over StandardScaler -- resistant to the heavy right-skew of credit limits and bill amounts (some clients have 10x the median).
- **PAY_* columns left unscaled** -- already well-behaved ordinals (-2 to 8) with meaningful spacing.
- **BILL_SEP + PAYMENT_SEP preserved** -- the underpayment signal (payment < bill -> default) is a strong indicator best learned as raw dollar amounts.
- **TOTAL_DELAY_DAYS** engineered from PAY columns -- captures cumulative delay severity in a single interpretable feature.

---

## Modeling

### Vanilla MLP (Focal Loss)

Train a single neural network using Focal Loss to address class imbalance.

| Detail | Value |
|--------|-------|
| **Architecture** | `[64, 32, 16]` + BatchNorm + ReLU + Dropout |
| **Parameters** | ~3,500 |
| **Loss** | `FocalLoss(alpha=0.78, gamma=2.0)` |
| **Optimizer** | `Adam(lr=2.12e-3, weight_decay=5e-6)` |
| **Batch size** | 128 |
| **CV strategy** | 5-fold stratified |
| **Tuning** | Optuna (30 trials) |
| **Early stopping** | Patience=20 |

### Ensemble (Averaging)

Combine 4 diverse models via simple probability averaging.

| Base Model | Type | Architecture / Params |
|------------|------|-----------------------|
| **XGBoost** | Gradient boosting | Tuned: n_estimators, max_depth, learning_rate, subsample, colsample |
| **Random Forest** | Bagging | Tuned: n_estimators, max_depth, min_samples_leaf |
| **MLP-A** | Neural net | `[64, 32, 16]` + BatchNorm (same as vanilla MLP) |
| **MLP-B** | Neural net | `[256, 128, 64]` -- no BatchNorm (structurally diverse from MLP-A) |

The simple average outperformed a Logistic Regression meta-model (recall 0.62 vs 0.35), so the meta-model was removed.

---

## Results

### Test Set Performance (threshold = 0.5)

| Metric | Ensemble | Vanilla MLP |
|--------|----------|-------------|
| **AUC-ROC** | **0.781** | 0.777 |
| **Average Precision** | **0.559** | 0.548 |
| **Brier Score** | **0.180** | 0.208 |
| **Precision** | **0.482** | 0.477 |
| **Recall** | **0.630** | 0.594 |
| **F1-Score** | **0.546** | 0.530 |

### Individual Model AUCs (Ensemble Components)

| Model | AUC-ROC |
|-------|---------|
| Ensemble (Avg) | **0.781** |
| RandomForest | 0.780 |
| MLP-B [256,128,64] | 0.778 |
| XGBoost | 0.776 |
| MLP-A [64,32,16]+BN | 0.775 |

### Vanilla MLP -- Threshold Calibration

| Metric | @ 0.5 Threshold | @ Calibrated Threshold (0.525) |
|--------|-----------------|-------------------------------|
| Precision | 0.477 | **0.521** |
| Recall | 0.594 | **0.561** |
| F1 | 0.530 | **0.540** |

> **Takeaway:** The ensemble consistently outperforms the vanilla MLP across all metrics. The ensemble's higher recall (0.63) means it catches more actual defaulters -- critical for credit risk where false negatives are costly. The calibrated threshold on the vanilla MLP trades some recall for improved precision and F1.

---

## Deployment

### Backend API (FastAPI)

`http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/predict?model=ensemble\|vanilla` | POST | Predict default probability from 9 raw features |
| `/model-info` | GET | Pre-computed metrics, confusion matrix, feature importance, scenario cases |
| `/docs` | GET | Interactive Swagger UI |

**`POST /predict`** accepts 9 raw input features, derives `TOTAL_DELAY_DAYS` internally, applies RobustScaler, and returns:
- Default probability
- Classification (Default / Non-Default)
- Per-model probability breakdown (ensemble only)
- Computed delay days

### Frontend Dashboard (Streamlit)

`http://localhost:8501`

- **Input panel**: Credit limit, 6-month repayment history, latest bill & payment
- **Speedometer**: Real-time risk gauge with green/yellow/red zones and threshold marker
- **Model selector**: Toggle between Ensemble and Vanilla MLP
- **Model votes**: Bar chart showing per-model probabilities
- **Metrics table**: Side-by-side comparison of both models
- **Confusion matrix**: With counts, recall, and precision
- **Scenario walkthroughs**: 8 real-world cases across all 4 confusion-matrix quadrants
- **Feature importance**: Bar chart + table from RandomForest
- **Tooltips**: Explanatory hover text on every input field

### Startup

```bash
# Terminal 1: Backend API
uv run uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend UI
uv run streamlit run frontend/app.py

# (Optional) MLflow tracking UI
docker start mlflow
# -> http://localhost:5000
```

---

## MLflow Tracking

Experiment tracking at `http://localhost:5000` under experiment **`credit-risk-default`**.

Each training run logs:

| Category | What's logged |
|----------|--------------|
| **Parameters** | Hyperparameters per model (arch, lr, dropout, batch_size, focal_gamma, etc.) |
| **Metrics** | AUC-ROC, Average Precision, Brier Score, Precision/Recall/F1 @ both thresholds |
| **Per-fold metrics** | 5-fold CV OOF AUC |
| **Artifacts** | Trained model weights (`.pt`, `.pkl`), configuration dumps |

```bash
# Start MLflow server
docker run -d \
  -p 5000:5000 \
  -v ~/mlflow-data:/mlflow \
  --name mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server --host 0.0.0.0 --port 5000
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.12 |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |
| **Data Processing** | pandas, numpy, scikit-learn |
| **Modeling** | PyTorch, XGBoost, scikit-learn |
| **Hyperparameter Tuning** | Optuna |
| **Experiment Tracking** | MLflow |
| **Backend API** | FastAPI, uvicorn |
| **Frontend** | Streamlit, Plotly |
| **Explainability** | SHAP |
| **Visualization** | matplotlib, seaborn, plotly |

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
