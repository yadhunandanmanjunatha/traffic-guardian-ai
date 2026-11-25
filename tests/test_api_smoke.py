"""
Traffic Guardian AI - Smoke Tests

Run with: pytest tests/
Requires the trained model artifacts to already exist under outputs/models/
(they're included in this repo).
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_default_request():
    response = client.post("/api/predict", json={})
    assert response.status_code == 200
    body = response.json()
    assert "congestion_class" in body
    assert "traffic_health_score" in body


def test_landmarks():
    response = client.get("/api/landmarks")
    assert response.status_code == 200
    assert len(response.json()["landmarks"]) > 0
