"""Route de classification du triage médical."""

from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import TriageRequest, TriageResponse
from src.api.services.provider import inference_service
from src.api.services.recommendations import get_recommendation

router = APIRouter(tags=["triage"])


@router.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest) -> TriageResponse:
    """Retourne une classification, une explication et une recommandation."""
    start = perf_counter()

    # Classification et explication fournies par le moteur.
    # Moteur injoignable ou réponse illisible : 503, sans détail technique ni médical.
    try:
        prediction = await inference_service.predict(request)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Moteur d'inférence indisponible.",
        ) from error

    # Recommandation déterministe fournie par l'API.
    recommendation = get_recommendation(
        priority=prediction["priority"],
        language=request.language,
    )

    latency_ms = (perf_counter() - start) * 1000

    return TriageResponse(
        request_id=str(uuid4()),
        timestamp=datetime.now(UTC),
        model_version=inference_service.model_version,
        priority=prediction["priority"],
        explanation=prediction["explanation"],
        recommendation=recommendation,
        latency_ms=round(latency_ms, 2),
        status="success",
    )
