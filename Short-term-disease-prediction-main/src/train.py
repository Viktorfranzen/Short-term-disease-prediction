# src/train.py

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib, json
import numpy as np

# Load the diabetes dataset
Xy = load_diabetes(as_frame=True)
x_input = Xy.data
y_target = Xy.target

print(y_target.min())


#Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    x_input, y_target, test_size=0.2, random_state=42
)


model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

#Train the model
model = model.fit(X_train, y_train)

#Ask the model to make predictions
preds = model.predict(X_test)

#Evaluate predictions using RMSE
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

#Save the model
joblib.dump(model, "model.pkl")
print(f"RMSE: {rmse:.2f}")

#store evaluation metric
with open("metrics.json", "w") as f:
    json.dump({"rmse": rmse}, f)

