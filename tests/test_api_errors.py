"""Tests des cas dégradés de l'API : moteur absent, erreur inattendue, redirection."""

import os

os.environ["INFERENCE_BACKEND"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from src.api.routes import health, triage  # noqa: E402

PAYLOAD = {"language": "fr", "question": "Cas fictif."}


class EngineNotReady:
    """Moteur chargé mais indisponible (ex. vLLM injoignable au démarrage)."""

    is_ready = False
    model_version = "test"


def test_root_redirects_to_docs():
    """La racine renvoie vers la documentation Swagger."""
    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_degraded(monkeypatch):
    """/health signale un moteur indisponible sans renvoyer d'erreur."""
    monkeypatch.setattr(health, "inference_service", EngineNotReady())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_ready": False}


def test_triage_engine_failure_returns_503(monkeypatch):
    """Moteur en panne : 503, sans détail technique dans la réponse."""

    async def failing_predict(request):
        raise RuntimeError("vLLM injoignable sur 10.0.0.12")

    monkeypatch.setattr(triage.inference_service, "predict", failing_predict)

    with TestClient(app) as client:
        response = client.post("/triage", json=PAYLOAD)

    assert response.status_code == 503
    assert response.json() == {"detail": "Moteur d'inférence indisponible."}
    assert "10.0.0.12" not in response.text


def test_unexpected_error_counted_in_metrics(monkeypatch):
    """Une erreur inattendue renvoie 500 et est comptée dans les métriques."""

    def broken_recommendation(priority, language):
        raise KeyError(priority)

    monkeypatch.setattr(triage, "get_recommendation", broken_recommendation)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/triage", json=PAYLOAD)
        metrics = client.get("/metrics").text

    assert response.status_code == 500
    assert 'status="500"' in metrics
