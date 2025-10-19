# src/train.py
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import joblib
import json
import numpy as np
import pandas as pd  # ensure installed: pip install pandas

# -----------------------------
# v0.2 — RandomForestRegressor + High-Risk Flag
# -----------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
HIGH_RISK_PERCENTILE = 0.75  # top 25% progression considered "high risk"

# 1) Load data
Xy = load_diabetes(as_frame=True)
X = Xy.data
y = Xy.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Define improved model (no need for scaling)
model = RandomForestRegressor(
    n_estimators=400,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate RMSE using modern API
rmse = float(root_mean_squared_error(y_test, y_pred))

# -----------------------------
# Optional: "High-Risk" flag (binary threshold)
# -----------------------------
# Define threshold from training data (75th percentile of progression)
threshold = float(pd.Series(y_train).quantile(HIGH_RISK_PERCENTILE))

y_test_is_high = (y_test >= threshold).astype(int)
y_pred_is_high = (y_pred >= threshold).astype(int)

# Precision/Recall (guard if no positives)
if y_test_is_high.sum() == 0 or y_pred_is_high.sum() == 0:
    precision_at_thresh = None
    recall_at_thresh = None
else:
    precision_at_thresh = float(precision_score(y_test_is_high, y_pred_is_high))
    recall_at_thresh = float(recall_score(y_test_is_high, y_pred_is_high))

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(model, "model.pkl")

metrics = {
    "version": "v0.2",
    "model": "RandomForestRegressor",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "rmse": rmse,
    "high_risk_threshold": threshold,
    "high_risk_percentile": HIGH_RISK_PERCENTILE,
    "test_high_risk_prevalence": float(np.mean(y_test_is_high)),
    "precision_at_threshold": precision_at_thresh,
    "recall_at_threshold": recall_at_thresh,
    "notes": "v0.2 switches from LinearRegression to RandomForest; uses root_mean_squared_error for modern sklearn API."
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# -----------------------------
# Console output
# -----------------------------
print(f"[v0.2] RMSE: {rmse:.2f}")
if precision_at_thresh is not None:
    print(f"[v0.2] Precision@thr: {precision_at_thresh:.3f} | Recall@thr: {recall_at_thresh:.3f} (thr={threshold:.2f})")
else:
    print(f"[v0.2] Precision/Recall@thr: undefined (no positives or predictions at thr={threshold:.2f})")
