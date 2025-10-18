# Diabetes Triage ML Service (v0.1)
Train: `python src/train.py`  
Run API: `uvicorn src.predict_service:app --reload` → http://127.0.0.1:8000/docs

Endpoints:
- GET /health → {"status":"ok","model_version":"v0.1"}
- POST /predict → {"prediction": <float>}

Example payload:
{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}


Valid inputs:
| Field   | Meaning                                             | Typical Range | Example  |
| ------- | --------------------------------------------------- | ------------- | -------- |
| **age** | Age of patient (standardized).                      | -0.1 → +0.1   | `0.02`   |
| **sex** | Biological sex (numeric; −≈ female, +≈ male).       | -0.06 → +0.06 | `-0.044` |
| **bmi** | Body mass index (kg/m², standardized).              | -0.1 → +0.2   | `0.06`   |
| **bp**  | Mean arterial blood pressure (mm Hg, standardized). | -0.1 → +0.1   | `-0.03`  |
| **s1**  | Serum total cholesterol (standardized).             | -0.1 → +0.1   | `-0.02`  |
| **s2**  | Low-density lipoproteins (LDL, standardized).       | -0.1 → +0.1   | `0.03`   |
| **s3**  | High-density lipoproteins (HDL, standardized).      | -0.1 → +0.1   | `-0.02`  |
| **s4**  | Total cholesterol / HDL ratio (standardized).       | -0.1 → +0.1   | `0.02`   |
| **s5**  | Log of serum triglycerides (standardized).          | -0.1 → +0.1   | `0.02`   |
| **s6**  | Blood sugar level (standardized).                   | -0.05 → +0.05 | `-0.001` |

