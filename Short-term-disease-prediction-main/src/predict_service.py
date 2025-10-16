from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

try:
    model = joblib.load("model.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

app = FastAPI(title="Diabetes Progression Predictor")

class PatientData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

#Health endpoint
@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v0.1"}

# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    try:
        features = np.array([[data.age, data.sex, data.bmi, data.bp,
                              data.s1, data.s2, data.s3, data.s4, data.s5, data.s6]])
        prediction = float(model.predict(features)[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/")
def root():
    return {
        "message": "Diabetes predictor API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }
