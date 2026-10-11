"""Recommandations standardisées selon la priorité."""

from typing import Literal

Priority = Literal["P1", "P2", "P3"]
Language = Literal["fr", "en"]

RECOMMENDATIONS = {
    "fr": {
        "P1": "Prise en charge immédiate par une équipe médicale.",
        "P2": "Évaluation médicale rapide nécessaire.",
        "P3": "Évaluation médicale différée selon l'organisation du service.",
    },
    "en": {
        "P1": "Immediate assessment by a medical team.",
        "P2": "Prompt medical assessment required.",
        "P3": "Deferred medical assessment according to department procedures.",
    },
}


def get_recommendation(priority: Priority, language: Language) -> str:
    """Retourne la recommandation associée à la priorité."""
    return RECOMMENDATIONS[language][priority]
