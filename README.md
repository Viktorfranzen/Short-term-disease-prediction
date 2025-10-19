# Diabetes Triage ML Service (v0.1)

## Run locally
```bash
python src/train.py
```

## Run API
```bash
uvicorn src.predict_service:app --reload
```
→ [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Run with Docker
```bash
docker build -t diabetes-triage .
docker run -p 8080:8080 diabetes-triage
```

---

## Endpoints
- **GET /health** → `{"status": "ok", "model_version": "v0.1"}`
- **POST /predict** → `{"prediction": <float>}`

### Example payload
```json
{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}
```

---

## Input fields

| Field | Meaning | Typical Range | Example |
|-------|----------|----------------|----------|
| **age** | Age (standardized) | -0.1 → +0.1 | 0.02 |
| **sex** | Biological sex (−≈ female, +≈ male) | -0.06 → +0.06 | -0.044 |
| **bmi** | Body mass index (standardized) | -0.1 → +0.2 | 0.06 |
| **bp** | Mean arterial pressure | -0.1 → +0.1 | -0.03 |
| **s1** | Serum total cholesterol | -0.1 → +0.1 | -0.02 |
| **s2** | LDL cholesterol | -0.1 → +0.1 | 0.03 |
| **s3** | HDL cholesterol | -0.1 → +0.1 | -0.02 |
| **s4** | Total/HDL cholesterol ratio | -0.1 → +0.1 | 0.02 |
| **s5** | Log serum triglycerides | -0.1 → +0.1 | 0.02 |
| **s6** | Blood sugar level | -0.05 → +0.05 | -0.001 |

---

## Training & Metrics
- Random seed: `42`
- Metrics stored in `metrics.json` (includes RMSE).
- Artifacts (`model.pkl`, `metrics.json`) uploaded by CI.

---

## CI/CD
- **On push/PR:** runs lint, unit tests, smoke test, and training (artifacts uploaded).
- **On tag (`v*`):** builds Docker image, runs container smoke test, pushes to GHCR, and publishes a GitHub Release with metrics & changelog.
