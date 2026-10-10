"""Tests de validation et de protection des données de l'API."""

import os

os.environ["INFERENCE_BACKEND"] = "mock"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture
def client():
    """Fournit un client API utilisant le moteur simulé."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "language",
    ["de", "es", "", "FR", None, 123],
)
def test_invalid_language(client, language):
    """Rejette les langues non autorisées."""
    response = client.post(
        "/triage",
        json={
            "language": language,
            "question": "Cas fictif.",
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "question",
    ["", None, 123, [], {}],
)
def test_invalid_question(client, question):
    """Rejette les questions vides ou de type incorrect."""
    response = client.post(
        "/triage",
        json={
            "language": "fr",
            "question": question,
        },
    )

    assert response.status_code == 422


def test_missing_required_fields(client):
    """Rejette une requête incomplète."""
    response = client.post("/triage", json={})

    assert response.status_code == 422


def test_extra_fields_rejected(client):
    """Rejette les champs non prévus par le contrat API."""
    response = client.post(
        "/triage",
        json={
            "language": "fr",
            "question": "Cas fictif.",
            "patient_name": "Patient Fictif",
        },
    )

    assert response.status_code == 422


def test_patient_data_not_echoed(client):
    """Vérifie que la réponse ne recopie pas les données saisies."""
    sensitive_marker = "PATIENT_TEST_CONFIDENTIEL_123"

    response = client.post(
        "/triage",
        json={
            "language": "fr",
            "question": sensitive_marker,
        },
    )

    assert response.status_code == 200
    assert sensitive_marker not in response.text


def test_invalid_request_not_exposed_in_metrics(client):
    """Vérifie que les métriques n'exposent pas le contenu médical."""
    sensitive_marker = "DONNEE_MEDICALE_FICTIVE_456"

    client.post(
        "/triage",
        json={
            "language": "fr",
            "question": sensitive_marker,
        },
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert sensitive_marker not in response.text