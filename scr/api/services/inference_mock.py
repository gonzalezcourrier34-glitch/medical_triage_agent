"""Service d'inférence simulé pour les tests techniques."""

from scr.api.schemas import TriageRequest


class MockInferenceService:
    """Simule une inférence sans modèle ni GPU."""

    model_version = "mock-0.1.0"

    @property
    def is_ready(self) -> bool:
        """Indique que le simulateur est disponible."""
        return True

    def load(self) -> None:
        """Aucun modèle à charger."""

    async def predict(self, request: TriageRequest) -> dict[str, str]:
        """Retourne une classification et une explication fictives."""
        explanations = {
            "fr": "Explication simulée pour les tests techniques.",
            "en": "Simulated explanation for technical testing.",
        }

        return {
            "priority": "P2",
            "explanation": explanations[request.language],
        }


inference_service = MockInferenceService()