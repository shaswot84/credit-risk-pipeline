"""Quick test that all models and scaler load without errors."""
import pickle, joblib, torch, numpy as np
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent
SCALER = PIPELINE / "data/processed/scaler.pkl"
ENSEMBLE = PIPELINE / "data-mining/building-nns/output/stacking-ensemble"

# Scaler
with open(SCALER, "rb") as f:
    pickle.load(f)
print("✓ scaler")

# Config
with open(ENSEMBLE / "config.pkl", "rb") as f:
    config = pickle.load(f)
print(f"✓ config (input_dim={config['input_dim']})")

# XGBoost
joblib.load(ENSEMBLE / "xgb_final.pkl")
print("✓ xgb")

# RF
joblib.load(ENSEMBLE / "rf_final.pkl")
print("✓ rf")

# MLPs (just check they load)
from main import MLPBase
input_dim = config["input_dim"]
mlp_a = MLPBase(input_dim, [64, 32, 16], [0.3, 0.3, 0.2], use_batchnorm=True)
mlp_a.load_state_dict(torch.load(ENSEMBLE / "mlp_a_final.pt", weights_only=True))
mlp_a.eval()
print("✓ mlp-a")

mlp_b = MLPBase(input_dim, [256, 128, 64], [0.4, 0.3, 0.2], use_batchnorm=False)
mlp_b.load_state_dict(torch.load(ENSEMBLE / "mlp_b_final.pt", weights_only=True))
mlp_b.eval()
print("✓ mlp-b")

# Quick test prediction
X = np.zeros((1, 10), dtype=np.float32)
with torch.no_grad():
    p_a = float(torch.sigmoid(mlp_a(torch.tensor(X))).item())
    p_b = float(torch.sigmoid(mlp_b(torch.tensor(X))).item())
print(f"✓ test prediction: mlp_a={p_a:.3f}, mlp_b={p_b:.3f}")
print("\nAll OK — backend ready to start.")
