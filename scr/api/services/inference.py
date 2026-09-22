from typing import Literal

from src.api.schemas import TriageRequest


Priority = Literal["P1", "P2", "P3"]


class InferenceService:
    """Gère l'inférence du modèle de triage."""

    model_version = "mock"

    async def predict(self, request: TriageRequest) -> Priority:
        """Retourne la priorité prédite."""

        # Mock temporaire avant intégration vLLM
        return "P2"


inference_service = InferenceService()