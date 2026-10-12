"""Route de classification du triage médical."""

from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import TriageRequest, TriageResponse
from src.api.services import audit
from src.api.services.provider import BACKEND, inference_service
from src.api.services.recommendations import get_recommendation
from src.api.versions import API_VERSION, PROMPT_VERSION

router = APIRouter(tags=["triage"])


@router.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest) -> TriageResponse:
    """Retourne une classification, une explication et une recommandation."""
    start = perf_counter()
    request_id = str(uuid4())
    timestamp = datetime.now(UTC)

    def audit_entry(status_label: str, priority: str | None = None) -> float:
        latency = round((perf_counter() - start) * 1000, 2)
        audit.record(
            request_id=request_id,
            timestamp=timestamp,
            request=request,
            backend=BACKEND,
            model_version=inference_service.model_version,
            prompt_version=PROMPT_VERSION,
            api_version=API_VERSION,
            status=status_label,
            latency_ms=latency,
            priority=priority,
        )
        return latency

    # Classification et explication fournies par le moteur.
    # Moteur injoignable ou réponse illisible : 503, sans détail technique ni médical.
    try:
        prediction = await inference_service.predict(request)
    except Exception as error:
        audit_entry("engine_unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Moteur d'inférence indisponible.",
        ) from error

    # Recommandation déterministe fournie par l'API.
    recommendation = get_recommendation(
        priority=prediction["priority"],
        language=request.language,
    )

    # Traçabilité : une ligne d'audit par décision (sans le texte du patient).
    latency_ms = audit_entry("success", prediction["priority"])

    return TriageResponse(
        request_id=request_id,
        timestamp=timestamp,
        model_version=inference_service.model_version,
        priority=prediction["priority"],
        explanation=prediction["explanation"],
        recommendation=recommendation,
        latency_ms=latency_ms,
        status="success",
    )
