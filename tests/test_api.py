import pytest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# ─── Mock model to avoid loading from MLflow ─────────────────
class MockModel:
    def predict(self, texts):
        return [1] * len(texts)

    def decision_function(self, texts):
        return [0.8] * len(texts)


# ─── Patch model before importing app ────────────────────────
@pytest.fixture
def client():
    with patch("src.serving.app.MODEL", MockModel()), \
         patch("src.serving.app.MODEL_INFO", {
             "model_name"   : "MentalHealthClassifier",
             "model_version": "1"
         }), \
         patch("src.serving.app.CONFIG", {
             "mlflow": {
                 "experiment_name": "mental_health_classifier_v2",
                 "tracking_uri"   : "file:///mlruns"
             }
         }):
        from src.serving.app import app
        with TestClient(app) as c:
            yield c


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_message(self, client):
        response = client.get("/")
        assert "message" in response.json()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_health_returns_model_info(self, client):
        response = client.get("/health")
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post(
            "/predict",
            json={"text": "I feel hopeless and sad"}
        )
        assert response.status_code == 200

    def test_predict_returns_label(self, client):
        response = client.post(
            "/predict",
            json={"text": "I feel hopeless and sad"}
        )
        data = response.json()
        assert "label" in data
        assert data["label"] in ["Depression", "Non-Depression"]

    def test_predict_returns_confidence(self, client):
        response = client.post(
            "/predict",
            json={"text": "I feel hopeless and sad"}
        )
        data = response.json()
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1

    def test_predict_empty_text_returns_422(self, client):
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        response = client.post(
            "/predict/batch",
            json={"texts": [
                "I feel hopeless",
                "Today was amazing"
            ]}
        )
        assert response.status_code == 200

    def test_batch_predict_returns_correct_count(self, client):
        response = client.post(
            "/predict/batch",
            json={"texts": [
                "I feel hopeless",
                "Today was amazing"
            ]}
        )
        data = response.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_predict_empty_list_returns_422(self, client):
        response = client.post(
            "/predict/batch",
            json={"texts": []}
        )
        assert response.status_code == 422