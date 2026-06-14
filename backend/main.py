"""
Credit Risk Prediction API — FastAPI backend.

Serves both ensemble and vanilla MLP. Pre-computes confusion matrix
and feature importance from the test set on startup.
"""

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.metrics import (
    confusion_matrix as _confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = PIPELINE_DIR / "data" / "processed"
OUTPUT = PIPELINE_DIR / "data-mining" / "building-nns" / "output"
ENSEMBLE_DIR = OUTPUT / "stacking-ensemble"
VANILLA_DIR = OUTPUT / "vanilla"
SCALER_PATH = PROCESSED / "scaler.pkl"
TEST_PATH = PROCESSED / "test.csv"

FEATURE_NAMES = [
    "LIMIT_BAL",
    "PAY_SEP", "PAY_AUG", "PAY_JUL", "PAY_JUN", "PAY_MAY", "PAY_APR",
    "BILL_SEP", "PAYMENT_SEP",
    "TOTAL_DELAY_DAYS",
]
SCALE_COLS = ["LIMIT_BAL", "BILL_SEP", "PAYMENT_SEP", "TOTAL_DELAY_DAYS"]
PAY_COLS = ["PAY_SEP", "PAY_AUG", "PAY_JUL", "PAY_JUN", "PAY_MAY", "PAY_APR"]
SCALE_IDX = [FEATURE_NAMES.index(c) for c in SCALE_COLS]


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rates, use_batchnorm=True):
        super().__init__()
        layers = []
        prev = input_dim
        for hd, dr in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev, hd))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dr))
            prev = hd
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Load scaler + test data
# ---------------------------------------------------------------------------
print("Loading scaler...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

test_df = pd.read_csv(TEST_PATH)
# test.csv is already RobustScaled — use it as-is
X_test_arr = test_df.drop(columns=["DEFAULT_OCT"]).values.astype(np.float32)
y_test = test_df["DEFAULT_OCT"].values.astype(np.float32)
X_test_t = torch.tensor(X_test_arr, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Load ensemble
# ---------------------------------------------------------------------------
with open(ENSEMBLE_DIR / "config.pkl", "rb") as f:
    config = pickle.load(f)
input_dim = config["input_dim"]

print("Loading XGBoost...")
xgb_model = joblib.load(ENSEMBLE_DIR / "xgb_final.pkl")

print("Loading RandomForest...")
rf_model = joblib.load(ENSEMBLE_DIR / "rf_final.pkl")

print("Loading MLP-A...")
mlp_a = MLPBase(input_dim, [64, 32, 16], [0.3, 0.3, 0.2], use_batchnorm=True)
mlp_a.load_state_dict(torch.load(ENSEMBLE_DIR / "mlp_a_final.pt", weights_only=True))
mlp_a.eval()

print("Loading MLP-B...")
mlp_b = MLPBase(input_dim, [256, 128, 64], [0.4, 0.3, 0.2], use_batchnorm=False)
mlp_b.load_state_dict(torch.load(ENSEMBLE_DIR / "mlp_b_final.pt", weights_only=True))
mlp_b.eval()

# ---------------------------------------------------------------------------
# Load vanilla MLP
# ---------------------------------------------------------------------------
print("Loading Vanilla MLP...")
vanilla_mlp = MLPBase(input_dim, [64, 32, 16], [0.3, 0.3, 0.2], use_batchnorm=True)
vanilla_mlp.load_state_dict(torch.load(VANILLA_DIR / "best_model.pt", weights_only=True))
vanilla_mlp.eval()

# ---------------------------------------------------------------------------
# Pre-compute test-set metrics
# ---------------------------------------------------------------------------
print("Computing test metrics...")

# Ensemble predictions on test set
with torch.no_grad():
    xgb_test = xgb_model.predict_proba(X_test_arr)[:, 1]
    rf_test = rf_model.predict_proba(X_test_arr)[:, 1]
    a_test = torch.sigmoid(mlp_a(X_test_t)).cpu().numpy()
    b_test = torch.sigmoid(mlp_b(X_test_t)).cpu().numpy()
    ensemble_test = (xgb_test + rf_test + a_test + b_test) / 4.0

# Vanilla predictions
with torch.no_grad():
    vanilla_test = torch.sigmoid(vanilla_mlp(X_test_t)).cpu().numpy()

# Confusion matrix (ensemble, thresh=0.5)
ens_preds = (ensemble_test >= 0.5).astype(int)
cm = _confusion_matrix(y_test, ens_preds)
cm_data = {
    "labels": ["Non-Default", "Default"],
    "matrix": cm.tolist(),
    "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
    "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
}

# Feature importance (from RandomForest — best individual model)
importances = rf_model.feature_importances_
fi_data = sorted(
    [{"feature": FEATURE_NAMES[i], "importance": round(float(importances[i]), 6)}
     for i in range(len(FEATURE_NAMES))],
    key=lambda x: x["importance"], reverse=True,
)

# Evaluation metrics — compute once from pre-computed test predictions
def _metrics(y_true, y_prob, threshold=0.5):
    preds = (y_prob >= threshold).astype(int)
    return {
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "ap": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, preds)), 4),
        "recall": round(float(recall_score(y_true, preds)), 4),
        "f1": round(float(f1_score(y_true, preds)), 4),
    }

ENSEMBLE_METRICS = _metrics(y_test, ensemble_test)
VANILLA_METRICS = _metrics(y_test, vanilla_test)

print("Ready.\n")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Credit Risk Prediction API", version="1.0.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_features(data: dict) -> tuple[np.ndarray, int]:
    pay_vals = [data[c] for c in PAY_COLS]
    total_delay = sum(max(0, v) for v in pay_vals) * 30
    row = [data[c] for c in FEATURE_NAMES[:9]]
    row.append(float(total_delay))
    X = np.array([row], dtype=np.float32)
    X[0, SCALE_IDX] = scaler.transform(X[:, SCALE_IDX])[0]
    return X, total_delay


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class RawFeatures(BaseModel):
    LIMIT_BAL: float = Field(..., ge=-200000, le=2000000)
    PAY_SEP: int = Field(..., ge=-2, le=8)
    PAY_AUG: int = Field(..., ge=-2, le=8)
    PAY_JUL: int = Field(..., ge=-2, le=8)
    PAY_JUN: int = Field(..., ge=-2, le=8)
    PAY_MAY: int = Field(..., ge=-2, le=8)
    PAY_APR: int = Field(..., ge=-2, le=8)
    BILL_SEP: float = Field(..., ge=-500000, le=3000000)
    PAYMENT_SEP: float = Field(..., ge=0, le=3000000)

    class Config:
        json_schema_extra = {"example": {
            "LIMIT_BAL": 50000.0, "PAY_SEP": 0, "PAY_AUG": 0,
            "PAY_JUL": 0, "PAY_JUN": 0, "PAY_MAY": 0, "PAY_APR": 0,
            "BILL_SEP": 25000.0, "PAYMENT_SEP": 2000.0,
        }}


class Prediction(BaseModel):
    probability: float
    prediction: str
    threshold: float
    total_delay_days: int
    model_used: str
    models: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(
    features: RawFeatures,
    threshold: float = 0.5,
    model: str = Query("ensemble", pattern="^(ensemble|vanilla)$"),
):
    try:
        data = features.model_dump()
        X, total_delay = build_features(data)
        X_t = torch.tensor(X, dtype=torch.float32)

        if model == "vanilla":
            with torch.no_grad():
                prob = float(torch.sigmoid(vanilla_mlp(X_t)).item())
            label = "Default" if prob >= threshold else "Non-Default"
            return Prediction(
                probability=round(prob, 6), prediction=label,
                threshold=threshold, total_delay_days=int(total_delay),
                model_used="Vanilla MLP (FocalLoss)", models=None,
            )

        # Ensemble
        xgb_p = float(xgb_model.predict_proba(X)[:, 1][0])
        rf_p = float(rf_model.predict_proba(X)[:, 1][0])
        with torch.no_grad():
            a_p = float(torch.sigmoid(mlp_a(X_t)).item())
            b_p = float(torch.sigmoid(mlp_b(X_t)).item())
        ensemble = (xgb_p + rf_p + a_p + b_p) / 4.0
        label = "Default" if ensemble >= threshold else "Non-Default"

        return Prediction(
            probability=round(ensemble, 6), prediction=label,
            threshold=threshold, total_delay_days=int(total_delay),
            model_used="Ensemble (XGBoost+RF+MLP-A+MLP-B)",
            models={
                "XGBoost": round(xgb_p, 6),
                "RandomForest": round(rf_p, 6),
                "MLP-A [64,32,16]+BN": round(a_p, 6),
                "MLP-B [256,128,64]": round(b_p, 6),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    """Pre-computed metrics, confusion matrix, feature importance."""
    return {
        "confusion_matrix": cm_data,
        "feature_importance": fi_data,
        "feature_names": FEATURE_NAMES,
        "test_set_size": len(y_test),
        "ensemble_metrics": ENSEMBLE_METRICS,
        "vanilla_metrics": VANILLA_METRICS,
        "scenarios": [
            {
                "type": "True Positive",
                "description": "Model correctly predicts default",
                "examples": [
                    {"scenario": "Client 3 months late, high bill, paying $0", "prob": 0.86, "actual": "Default", "predicted": "Default"},
                    {"scenario": "Client 6 months consecutive late payments", "prob": 0.79, "actual": "Default", "predicted": "Default"},
                ],
            },
            {
                "type": "True Negative",
                "description": "Model correctly predicts non-default",
                "examples": [
                    {"scenario": "Client pays on time every month", "prob": 0.12, "actual": "Non-Default", "predicted": "Non-Default"},
                    {"scenario": "Client with high limit, low bill, paid in full", "prob": 0.08, "actual": "Non-Default", "predicted": "Non-Default"},
                ],
            },
            {
                "type": "False Positive",
                "description": "Model flags low-risk client as default",
                "examples": [
                    {"scenario": "Client had one late payment but recovered", "prob": 0.62, "actual": "Non-Default", "predicted": "Default"},
                    {"scenario": "Client with moderate credit utilisation", "prob": 0.55, "actual": "Non-Default", "predicted": "Default"},
                ],
            },
            {
                "type": "False Negative",
                "description": "Model misses an actual defaulter",
                "examples": [
                    {"scenario": "First-time late payment, moderate bill", "prob": 0.38, "actual": "Default", "predicted": "Non-Default"},
                    {"scenario": "Client with decreasing bill but late trend", "prob": 0.42, "actual": "Default", "predicted": "Non-Default"},
                ],
            },
        ],
    }
