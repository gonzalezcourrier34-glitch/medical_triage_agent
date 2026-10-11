"""Tests du moteur vLLM sans serveur : les échanges HTTP sont simulés."""

import asyncio
import json

import httpx
import pytest

from src.api.schemas import TriageRequest
from src.api.services import inference_vllm
from src.api.services.inference_vllm import VLLMInferenceService

REQUEST = TriageRequest(language="fr", question="Douleur thoracique depuis 1 heure.")
REAL_ASYNC_CLIENT = httpx.AsyncClient


def completion(content: str) -> dict:
    """Réponse chat/completions minimale au format OpenAI."""
    return {"choices": [{"message": {"content": content}}]}


def use_transport(monkeypatch, handler) -> None:
    """Remplace le réseau par un transport simulé pour les appels asynchrones."""

    def client(**kwargs):
        return REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(inference_vllm.httpx, "AsyncClient", client)


def fake_get(response=None, error=None):
    """Simule httpx.get pour la vérification au démarrage."""

    def get(url, headers=None, timeout=None):
        if error:
            raise error
        response.request = httpx.Request("GET", url)
        return response

    return get


# ------------------------------------------------------------ démarrage


def test_load_ready_when_model_is_served(monkeypatch):
    """Le service est prêt si vLLM sert le modèle attendu."""
    served = httpx.Response(200, json={"data": [{"id": inference_vllm.VLLM_MODEL}]})
    monkeypatch.setattr(inference_vllm.httpx, "get", fake_get(served))

    service = VLLMInferenceService()
    service.load()

    assert service.is_ready is True


def test_load_not_ready_when_model_is_missing(monkeypatch):
    """Un autre modèle servi ne suffit pas."""
    served = httpx.Response(200, json={"data": [{"id": "autre-modele"}]})
    monkeypatch.setattr(inference_vllm.httpx, "get", fake_get(served))

    service = VLLMInferenceService()
    service.load()

    assert service.is_ready is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"error": httpx.ConnectError("connexion refusée")},
        {"response": httpx.Response(401, json={"error": "clé invalide"})},
        {"response": httpx.Response(200, content=b"pas du JSON")},
    ],
    ids=["injoignable", "non-autorise", "reponse-illisible"],
)
def test_load_not_ready_on_error(monkeypatch, kwargs):
    """Serveur absent, clé refusée ou réponse illisible : le service n'est pas prêt."""
    monkeypatch.setattr(inference_vllm.httpx, "get", fake_get(**kwargs))

    service = VLLMInferenceService()
    service.load()

    assert service.is_ready is False


# ------------------------------------------------------------ requête


def test_payload_matches_training_settings():
    """Génération déterministe, sans thinking, arrêt sur les fins de réponse apprises."""
    payload = VLLMInferenceService().payload(REQUEST)

    assert payload["model"] == inference_vllm.VLLM_MODEL
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["stop_token_ids"] == [151645, 151643]
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


def test_predict_parses_generation(monkeypatch):
    """La réponse générée est lue et la clé d'API est transmise."""
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["path"] = request.url.path
        sent["auth"] = request.headers.get("authorization")
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion("PRIORITY: P1\nEXPLANATION: Douleur aiguë."))

    monkeypatch.setattr(inference_vllm, "HEADERS", {"Authorization": "Bearer cle-de-test"})
    use_transport(monkeypatch, handler)

    service = VLLMInferenceService()
    result = asyncio.run(service.predict(REQUEST))

    assert result == {"priority": "P1", "explanation": "Douleur aiguë."}
    assert service.is_ready is True
    assert sent["path"].endswith("/chat/completions")
    assert sent["auth"] == "Bearer cle-de-test"
    assert sent["body"]["temperature"] == 0


def test_predict_marks_service_down_on_http_error(monkeypatch):
    """Une erreur du serveur est propagée et le service passe à « non prêt »."""
    use_transport(monkeypatch, lambda request: httpx.Response(500))

    service = VLLMInferenceService()
    service._ready = True

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(service.predict(REQUEST))

    assert service.is_ready is False


def test_predict_rejects_answer_without_priority(monkeypatch):
    """Une génération sans priorité valide est refusée."""
    use_transport(
        monkeypatch,
        lambda request: httpx.Response(200, json=completion("EXPLANATION: sans priorité")),
    )

    with pytest.raises(ValueError):
        asyncio.run(VLLMInferenceService().predict(REQUEST))
