"""Schémas de validation des requêtes et réponses de l'API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TriageRequest(BaseModel):
    """Données reçues pour une demande de triage."""

    language: Literal["fr", "en"]
    question: str = Field(min_length=1)
    context: str = ""
    answer: str = ""


class TriageResponse(BaseModel):
    """Résultat retourné par l'API."""

    request_id: str
    timestamp: datetime
    model_version: str

    # Résultat du modèle Qwen3.
    priority: Literal["P1", "P2", "P3"]
    explanation: str

    # Recommandation déterministe fournie par l'API.
    recommendation: str

    # Traçabilité et performances.
    latency_ms: float = Field(ge=0)
    status: Literal["success"] = "success"

class TriageRequest(BaseModel):
    """Données reçues pour une demande de triage."""

    model_config = ConfigDict(extra="forbid")

    language: Literal["fr", "en"]
    question: str = Field(min_length=1)
    context: str = ""
    answer: str = ""
    
class HealthResponse(BaseModel):
    """État de disponibilité de l'API."""

    status: Literal["ok", "degraded"]
    model_ready: bool


class VersionResponse(BaseModel):
    """Versions exposées par l'API."""

    api_version: str
    model_version: str
    prompt_version: str