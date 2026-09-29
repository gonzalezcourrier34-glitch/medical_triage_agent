from fastapi import APIRouter

from scr.api.schemas import HealthResponse
from scr.api.services.inference import inference_service


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Retourne l'état de disponibilité de l'API."""

    return HealthResponse(
        status="ok",
        model_ready=inference_service.is_ready
    )