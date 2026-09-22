from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter

from src.api.schemas import TriageRequest, TriageResponse
from src.api.services.inference import inference_service


router = APIRouter(tags=["triage"])


@router.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest) -> TriageResponse:
    """Retourne une priorité de triage."""

    start = perf_counter()
    priority = await inference_service.predict(request)
    latency_ms = (perf_counter() - start) * 1000

    return TriageResponse(
        request_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc),
        model_version=inference_service.model_version,
        priority=priority,
        latency_ms=round(latency_ms, 2),
        status="success"
    )