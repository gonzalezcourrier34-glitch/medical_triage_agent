class TriageConfidenceConfig:
    """Configuration du score de confiance."""

    HIGH = 0.80
    MEDIUM = 0.55


def triage_confidence_label(
    score: float
) -> str:
    """Retourne le niveau de confiance."""
    if score >= TriageConfidenceConfig.HIGH:
        return "high"

    if score >= TriageConfidenceConfig.MEDIUM:
        return "medium"

    return "low"