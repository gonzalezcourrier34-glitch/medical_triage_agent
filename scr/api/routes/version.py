from fastapi import APIRouter

from scr.api.schemas import VersionResponse
from scr.api.services.inference import inference_service


router = APIRouter(tags=["version"])


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    """Retourne les versions du POC."""

    return VersionResponse(
        api_version="0.1.0",
        model_version=inference_service.model_version,
        prompt_version="triage-classification-v2"
    )