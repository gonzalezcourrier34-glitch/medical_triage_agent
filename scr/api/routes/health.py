from fastapi import APIRouter

from src.api.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Retourne l'état de disponibilité de l'API."""

    return HealthResponse(
        status="ok",
        model_ready=False
    )