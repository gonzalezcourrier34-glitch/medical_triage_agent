"""Tests des endpoints FastAPI."""

import os

os.environ["INFERENCE_BACKEND"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


def test_health():
    """Vérifie la disponibilité du moteur simulé."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_ready": True,
    }


def test_version():
    """Vérifie les versions exposées par l'API."""
    with TestClient(app) as client:
        response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "api_version": "0.1.0",
        "model_version": "mock-0.1.0",
        "prompt_version": "triage-classification-v2",
    }


def test_triage_fr():
    """Vérifie une réponse de triage en français."""
    payload = {
        "language": "fr",
        "question": "Exemple fictif pour tester l'API.",
    }

    with TestClient(app) as client:
        response = client.post("/triage", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] == "P2"
    assert data["explanation"] == "Explication simulée pour les tests techniques."
    assert data["recommendation"] == "Évaluation médicale rapide nécessaire."
    assert data["model_version"] == "mock-0.1.0"
    assert data["status"] == "success"
    assert data["latency_ms"] >= 0
    assert data["request_id"]
    assert data["timestamp"]


def test_triage_en():
    """Vérifie une réponse de triage en anglais."""
    payload = {
        "language": "en",
        "question": "Fictional example for API testing.",
    }

    with TestClient(app) as client:
        response = client.post("/triage", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] == "P2"
    assert data["explanation"] == "Simulated explanation for technical testing."
    assert data["recommendation"] == "Prompt medical assessment required."


def test_metrics():
    """Vérifie l'exposition des métriques Prometheus."""
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_request" in response.text