"""Moteur d'inférence distant : serveur vLLM compatible OpenAI.

vLLM sert le modèle fusionné (Qwen3-1.7B + LoRA SFT/DPO). L'API envoie la même
conversation qu'à l'entraînement et lit la réponse « PRIORITY / EXPLANATION ».
"""

import os

import httpx

from src.api.schemas import TriageRequest
from src.api.services.prompting import MODEL_VERSION, build_messages, parse_completion

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1").rstrip("/")
VLLM_MODEL = os.getenv("VLLM_MODEL", "triage-qwen3")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TIMEOUT_S = float(os.getenv("VLLM_TIMEOUT_S", "60"))
# Clé du serveur vLLM (option --api-key), indispensable dès qu'il est exposé (RunPod).
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}


class VLLMInferenceService:
    """Interroge un serveur vLLM."""

    model_version = MODEL_VERSION

    def __init__(self) -> None:
        self._ready = False

    @property
    def is_ready(self) -> bool:
        """Indique si vLLM a répondu à la dernière vérification."""
        return self._ready

    def load(self) -> None:
        """Vérifie au démarrage que vLLM sert bien le modèle attendu."""
        try:
            response = httpx.get(f"{VLLM_BASE_URL}/models", headers=HEADERS, timeout=10)
            response.raise_for_status()
            served = {model["id"] for model in response.json().get("data", [])}
            self._ready = VLLM_MODEL in served
        except (httpx.HTTPError, ValueError):
            self._ready = False

    def payload(self, request: TriageRequest) -> dict:
        """Construit la requête chat/completions."""
        return {
            "model": VLLM_MODEL,
            "messages": build_messages(request),
            "temperature": 0,
            "max_tokens": MAX_NEW_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            # Extension vLLM : arrêt sur <|im_end|> (151645) ou <|endoftext|> (151643).
            "stop_token_ids": [151645, 151643],
        }

    async def predict(self, request: TriageRequest) -> dict[str, str]:
        """Retourne la priorité et l'explication générées par vLLM."""
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S, headers=HEADERS) as client:
                response = await client.post(
                    f"{VLLM_BASE_URL}/chat/completions",
                    json=self.payload(request),
                )
                response.raise_for_status()
        except httpx.HTTPError:
            self._ready = False
            raise

        self._ready = True
        content = response.json()["choices"][0]["message"]["content"]
        return parse_completion(content)


inference_service = VLLMInferenceService()
