# Diabetes Triage ML Service (v0.1)
Train: `python src/train.py`  
Run API: `uvicorn src.predict_service:app --reload` → http://127.0.0.1:8000/docs

Endpoints:
- GET /health → {"status":"ok","model_version":"v0.1"}
- POST /predict → {"prediction": <float>}

Example payload:
{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}
