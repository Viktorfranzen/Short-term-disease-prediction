# src/train.py
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import joblib
import json
import numpy as np

# 1) Load data
Xy = load_diabetes(as_frame=True)
X = Xy.data
y = Xy.target

# 2) Split (seed for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# 4) Train
model.fit(X_train, y_train)

# 5) Predict (regression)
preds = model.predict(X_test)

# 6) RMSE (regression metric)
mse = mean_squared_error(y_test, preds)
rmse = float(np.sqrt(mse))

# 7) Define a high-risk flag and compute precision/recall
#    Use the 75th percentile of TRAINING labels as the threshold (no leakage).
risk_quantile = 0.75
threshold = float(np.quantile(y_train, risk_quantile))

y_true_highrisk = (y_test >= threshold).astype(int)
y_pred_highrisk = (preds >= threshold).astype(int)

precision = float(precision_score(y_true_highrisk, y_pred_highrisk, zero_division=0))
recall = float(recall_score(y_true_highrisk, y_pred_highrisk, zero_division=0))

# (Optional) see how many positives there are in test
support = int(y_true_highrisk.sum())
n_test = int(y_test.shape[0])

# 8) Save model
joblib.dump(model, "model.pkl")

# 9) Log metrics (JSON-serializable)
metrics = {
    "rmse": rmse,
    "risk_quantile": risk_quantile,
    "threshold": threshold,
    "precision_highrisk": precision,
    "recall_highrisk": recall,
    "support_highrisk_test": support,
    "n_test": n_test
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 10) Console summary
print(f"RMSE: {rmse:.2f}")
print(f"High-risk threshold (q={risk_quantile:.2f}): {threshold:.2f}")
print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | Positives in test: {support}/{n_test}")
