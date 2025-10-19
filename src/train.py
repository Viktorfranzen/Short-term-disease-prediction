# src/train.py
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split



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

# 2) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# 3) Model (trees don't require scaling)
model = RandomForestRegressor(
    n_estimators=400,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# 4) Train
model.fit(X_train, y_train)

# 5) Predict
y_pred = model.predict(X_test)

# 6) RMSE
rmse = compute_rmse(y_test, y_pred)

# 7) High-risk threshold + precision/recall
threshold = float(pd.Series(y_train).quantile(HIGH_RISK_PERCENTILE))
y_test_is_high = (y_test >= threshold).astype(int)
y_pred_is_high = (y_pred >= threshold).astype(int)

if y_test_is_high.sum() == 0 or y_pred_is_high.sum() == 0:
    precision_at_thresh = None
    recall_at_thresh = None
else:
    precision_at_thresh = float(precision_score(y_test_is_high, y_pred_is_high))
    recall_at_thresh = float(recall_score(y_test_is_high, y_pred_is_high))

# 8) Save artifacts
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
    "notes": "v0.2 switches from LinearRegression to RandomForest; uses RMSE with fallback for older sklearn.",
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 9) Console output (clean, deterministic)
positives = int(y_test_is_high.sum())
total = int(y_test_is_high.shape[0])
print(f"[v0.2] RMSE: {rmse:.2f}")
print(f"[v0.2] High-risk threshold (q={HIGH_RISK_PERCENTILE:.2f}): {threshold:.2f}")
if precision_at_thresh is None:
    print(f"[v0.2] Precision/Recall@thr: undefined (no positives or predictions at thr={threshold:.2f})")
else:
    print(
        f"[v0.2] Precision: {precision_at_thresh:.3f} | "
        f"Recall: {recall_at_thresh:.3f} | "
        f"Positives in test: {positives}/{total}"
    )
