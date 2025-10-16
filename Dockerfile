# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code and the trained model (release workflow will create model.pkl first)
COPY src/ /app/src/
COPY model.pkl /app/model.pkl

# env + port
ENV MODEL_PATH=/app/model.pkl
EXPOSE 8080

# start API
CMD ["uvicorn", "src.predict_service:app", "--host", "0.0.0.0", "--port", "8080"]
