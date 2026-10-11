"""Tests du moteur simulé et des recommandations."""

import asyncio

import pytest

from src.api.schemas import TriageRequest
from src.api.services.inference_mock import MockInferenceService
from src.api.services.recommendations import get_recommendation


@pytest.mark.parametrize(
    ("language", "explanation"),
    [
        ("fr", "Explication simulée pour les tests techniques."),
        ("en", "Simulated explanation for technical testing."),
    ],
)
def test_mock_prediction(language, explanation):
    """Vérifie la classification et l'explication bilingue."""
    service = MockInferenceService()

    request = TriageRequest(
        language=language,
        question="Cas fictif de test.",
    )

    prediction = asyncio.run(service.predict(request))

    assert prediction == {
        "priority": "P2",
        "explanation": explanation,
    }


def test_mock_ready():
    """Vérifie la disponibilité du moteur simulé."""
    service = MockInferenceService()

    service.load()

    assert service.is_ready is True
    assert service.model_version == "mock-0.1.0"


@pytest.mark.parametrize(
    ("priority", "language", "expected"),
    [
        ("P1", "fr", "Prise en charge immédiate par une équipe médicale."),
        ("P2", "fr", "Évaluation médicale rapide nécessaire."),
        (
            "P3",
            "fr",
            "Évaluation médicale différée selon l'organisation du service.",
        ),
        ("P1", "en", "Immediate assessment by a medical team."),
        ("P2", "en", "Prompt medical assessment required."),
        (
            "P3",
            "en",
            "Deferred medical assessment according to department procedures.",
        ),
    ],
)
def test_recommendations(priority, language, expected):
    """Vérifie les six recommandations standardisées."""
    assert get_recommendation(priority, language) == expected
