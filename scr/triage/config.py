from scr.config import SourceSchema


class TriageConfig:
    """Configuration de la classification TRIAGE."""

    PRIORITIES = {1, 2, 3}

    PRIORITY_LABELS = {
        1: "urgence_maximale",
        2: "urgence_moderee",
        3: "urgence_differee"
    }


class TriagePoolConfig:
    """Configuration du pool médical TRIAGE."""

    FAMILIES = SourceSchema.QCM_FAMILIES | {
        "open",
        "oeq"
    }