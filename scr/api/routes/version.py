from fastapi import APIRouter

from src.api.schemas import VersionResponse


router = APIRouter(tags=["version"])


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    """Retourne les versions du POC."""

    return VersionResponse(
        api_version="0.1.0",
        model_version="pending",
        prompt_version="triage-classification-v2"
    )