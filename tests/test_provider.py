"""Tests du choix du moteur d'inférence (variable INFERENCE_BACKEND)."""

import importlib

import pytest

from src.api.services import provider
from src.api.services.inference_mock import MockInferenceService
from src.api.services.inference_vllm import VLLMInferenceService


@pytest.fixture(autouse=True)
def restore_mock_backend(monkeypatch):
    """Remet le moteur simulé après chaque test."""
    yield
    monkeypatch.setenv("INFERENCE_BACKEND", "mock")
    importlib.reload(provider)


def load_backend(monkeypatch, backend: str):
    """Recharge le module de sélection avec le moteur demandé."""
    monkeypatch.setenv("INFERENCE_BACKEND", backend)
    return importlib.reload(provider)


def test_mock_backend(monkeypatch):
    """Moteur simulé : tests, CI, développement."""
    module = load_backend(monkeypatch, "mock")

    assert module.BACKEND == "mock"
    assert isinstance(module.inference_service, MockInferenceService)


def test_vllm_backend(monkeypatch):
    """Moteur vLLM : déploiement (aucun appel réseau à l'import)."""
    module = load_backend(monkeypatch, "vllm")

    assert module.BACKEND == "vllm"
    assert isinstance(module.inference_service, VLLMInferenceService)


def test_local_backend(monkeypatch):
    """Moteur local : seulement si PyTorch et PEFT sont installés (pas en CI)."""
    pytest.importorskip("torch")
    pytest.importorskip("peft")

    module = load_backend(monkeypatch, "local")

    assert module.BACKEND == "local"
    assert module.inference_service.is_ready is False


def test_unknown_backend(monkeypatch):
    """Une valeur inconnue est refusée dès le démarrage."""
    with pytest.raises(ValueError, match="inconnu"):
        load_backend(monkeypatch, "gpu-magique")
