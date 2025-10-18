# tests/test_health.py
from fastapi.testclient import TestClient
from src.predict_service import app

def test_health_endpoint():
    """
    Basic health check to ensure the API starts and responds correctly.
    """
    client = TestClient(app)
    response = client.get("/health")

    # Check that request succeeded
    assert response.status_code == 200

    # Check the structure of the JSON response
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_version" in data
    assert data["model_version"] == "v0.1"

