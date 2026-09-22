import json
from collections import Counter
from pathlib import Path


# ## Configuration du routage

# Centralise les routes et les champs utilisés par le routage.
# Le routage exploite patient et connaissance sans attribuer d'urgence.
class TriageRoutingConfig:
    """Configuration du routage clinique."""

    VERSION = "6.0"

    ALLOWED_ROUTES = {
        "protocol_annotation",
        "controlled_synthesis",
        "human_review",
        "triage_rejected"
    }

    PRIVACY_FLAGS = {
        "PII_CANDIDATE",
        "PII_REDACTION_EMAIL",
        "PII_REDACTION_PHONE",
        "PII_REDACTION_IP",
        "PII_REDACTION_ID",
        "PII_REDACTION_NAME"
    }

    PATIENT_EVIDENCE_FIELDS = {
        "age_days",
        "age_months",
        "age_years",
        "sex",
        "symptoms",
        "medical_history",
        "duration",
        "evolution",
        "critical_signs",
        "critical_sign_states"
    }

    PATIENT_INFORMATION_FIELDS = {
        "condition_mentions",
        "french_concept_states"
    }


# ## Accès aux extractions

# Isole l'accès aux blocs patient et connaissance médicale.
# Évite de disperser les clés JSON dans toute la logique métier.
def patient(record: dict) -> dict:
    """Retourne l'extraction patient."""
    return record.get("patient") or {}


def patient_extraction(record: dict) -> dict:
    """Retourne les métadonnées patient."""
    return patient(record).get("extraction") or {}


def medical_knowledge(record: dict) -> dict:
    """Retourne la connaissance médicale."""
    return record.get("medical_knowledge") or {}


# ## Champs extraits

# Sépare les preuves directement attribuables au patient des autres informations.
# Les constantes vitales restent reconnues via leur préfixe.
def extracted_fields(record: dict) -> list[str]:
    """Retourne les champs patient extraits."""
    return sorted(
        set(
            patient_extraction(record).get("extracted_fields")
            or []
        )
    )


def patient_evidence_fields(record: dict) -> list[str]:
    """Retourne les preuves patient."""
    return sorted(
        field
        for field in extracted_fields(record)
        if (
            field.startswith("vital_signs.")
            or field in TriageRoutingConfig.PATIENT_EVIDENCE_FIELDS
        )
    )


def patient_information_fields(record: dict) -> list[str]:
    """Retourne les informations patient complémentaires."""
    return sorted(
        field
        for field in extracted_fields(record)
        if field in TriageRoutingConfig.PATIENT_INFORMATION_FIELDS
    )


# ## Preuves patient

# Une preuve patient exige un élément explicitement attribuable au cas.
# La connaissance documentaire n'est jamais utilisée comme preuve patient.
def has_patient_measurement(record: dict) -> bool:
    """Indique si une mesure patient est disponible."""
    data = patient(record)

    return bool(
        data.get("age_days") is not None
        or data.get("age_months") is not None
        or data.get("age_years") is not None
        or data.get("sex")
        or data.get("vital_signs")
    )


def has_patient_evidence(record: dict) -> bool:
    """Indique si une preuve patient est exploitable."""
    return bool(
        patient_evidence_fields(record)
        or has_patient_measurement(record)
    )


# ## Motifs FRENCH

# Conserve séparément les motifs trouvés chez le patient et dans la connaissance.
# Leur union sert à mesurer la couverture globale du record.
def patient_motif_candidates(record: dict) -> list[dict]:
    """Retourne les motifs FRENCH patient."""
    return [
        candidate
        for candidate in (
            patient(record).get("french_motif_candidates")
            or []
        )
        if isinstance(candidate, dict)
    ]


def knowledge_motif_candidates(record: dict) -> list[dict]:
    """Retourne les motifs FRENCH de connaissance."""
    return [
        candidate
        for candidate in (
            medical_knowledge(record).get("french_motif_candidates")
            or []
        )
        if isinstance(candidate, dict)
    ]


def all_rule_ids(record: dict) -> list[str]:
    """Retourne tous les motifs FRENCH du record."""
    candidates = (
        patient_motif_candidates(record)
        + knowledge_motif_candidates(record)
    )

    return sorted({
        candidate.get("rule_id")
        for candidate in candidates
        if candidate.get("rule_id")
    })


def shared_rule_ids(record: dict) -> list[str]:
    """Retourne les motifs communs patient et connaissance."""
    patient_ids = {
        candidate.get("rule_id")
        for candidate in patient_motif_candidates(record)
        if candidate.get("rule_id")
    }

    knowledge_ids = {
        candidate.get("rule_id")
        for candidate in knowledge_motif_candidates(record)
        if candidate.get("rule_id")
    }

    return sorted(
        patient_ids & knowledge_ids
    )


# ## Connaissance médicale

# Distingue connaissance structurée et simple disponibilité d'une réponse correcte.
# La connaissance seule alimente la synthèse contrôlée, jamais l'urgence patient.
def has_medical_information(record: dict) -> bool:
    """Indique si une information médicale structurée existe."""
    return bool(
        medical_knowledge(record).get("has_medical_information")
    )


def has_medical_knowledge(record: dict) -> bool:
    """Indique si une connaissance médicale exploitable existe."""
    knowledge = medical_knowledge(record)

    return bool(
        knowledge.get("has_medical_information")
        and knowledge.get("has_correct_answer")
        and knowledge.get("correct_answers")
    )


# ## Blocages

# Les conflits proviennent uniquement des métadonnées patient.
# Les blocages de confidentialité restent prioritaires.
def has_clinical_conflict(record: dict) -> bool:
    """Détecte les conflits cliniques."""
    metadata = patient_extraction(record)

    return bool(
        metadata.get("critical_sign_conflicts")
        or metadata.get("french_concept_conflicts")
    )


def has_privacy_blocker(record: dict) -> bool:
    """Détecte les blocages de confidentialité."""
    flags = set(
        record.get("flags")
        or []
    )

    return bool(
        record.get("pii_matches")
        or flags & TriageRoutingConfig.PRIVACY_FLAGS
    )


# ## Décision de routage

# Utilise patient et connaissance tout en conservant leur niveau de preuve.
# Le protocole ne reçoit directement que les cas avec preuve patient et motif patient.
def decide_triage_route(record: dict) -> tuple[str, str]:
    """Détermine la destination du record."""
    patient_evidence = has_patient_evidence(record)
    patient_motifs = patient_motif_candidates(record)
    knowledge_motifs = knowledge_motif_candidates(record)
    knowledge = has_medical_knowledge(record)

    if record.get("reviewer_excluded"):
        return "triage_rejected", "HUMAN_EXCLUSION"

    if record.get("schema_error"):
        return "triage_rejected", "SCHEMA_ERROR"

    if has_privacy_blocker(record):
        return "human_review", "PRIVACY_REVIEW_REQUIRED"

    if has_clinical_conflict(record):
        return "human_review", "CLINICAL_CONFLICT"

    if patient_evidence and patient_motifs:
        return "protocol_annotation", "PATIENT_AND_FRENCH_EVIDENCE"

    if patient_evidence and knowledge_motifs:
        return "human_review", "PATIENT_WITH_KNOWLEDGE_MOTIF_ONLY"

    if patient_evidence:
        return "human_review", "PATIENT_WITHOUT_FRENCH_MOTIF"

    if knowledge and knowledge_motifs:
        return "controlled_synthesis", "MEDICAL_KNOWLEDGE_AND_FRENCH_MOTIF"

    if knowledge:
        return "controlled_synthesis", "MEDICAL_KNOWLEDGE_AVAILABLE"

    if patient_motifs or knowledge_motifs:
        return "human_review", "FRENCH_MOTIF_WITHOUT_PATIENT_EVIDENCE"

    if (
        has_medical_information(record)
        or patient_information_fields(record)
    ):
        return "human_review", "MEDICAL_INFORMATION_WITHOUT_TRIAGE_TARGET"

    return "triage_rejected", "NO_USABLE_TRIAGE_INFORMATION"


# ## Registre de routage

# Produit une vue compacte, traçable et compatible avec le nouveau schéma.
# Les motifs patient, connaissance, communs et globaux restent distingués.
def build_routing_record(record: dict) -> dict:
    """Construit une décision de routage."""
    route, reason = decide_triage_route(record)

    patient_data = patient(record)
    patient_meta = patient_extraction(record)
    knowledge = medical_knowledge(record)

    evidence_fields = patient_evidence_fields(record)
    information_fields = patient_information_fields(record)

    patient_motifs = patient_motif_candidates(record)
    knowledge_motifs = knowledge_motif_candidates(record)

    patient_rule_ids = sorted({
        candidate.get("rule_id")
        for candidate in patient_motifs
        if candidate.get("rule_id")
    })

    knowledge_rule_ids = sorted({
        candidate.get("rule_id")
        for candidate in knowledge_motifs
        if candidate.get("rule_id")
    })

    concept_states = (
        patient_data.get("french_concept_states")
        or {}
    )

    flags = set(
        record.get("flags")
        or []
    )

    review_reasons = []

    if "HIGH_RISK" in flags:
        review_reasons.append("HIGH_RISK")

    if has_clinical_conflict(record):
        review_reasons.append("CLINICAL_CONFLICT")

    if has_privacy_blocker(record):
        review_reasons.append("PRIVACY_REVIEW_REQUIRED")

    return {
        "id": record["id"],
        "group_id": record.get("group_id"),
        "source": record.get("source"),
        "dataset": record.get("dataset"),
        "source_id": record.get("source_id"),
        "source_split": record.get("source_split"),
        "family": record.get("family"),
        "language": record.get("language"),
        "clean_content_hash": record.get("clean_content_hash"),
        "triage_route": route,
        "route_reason": reason,
        "routing_version": TriageRoutingConfig.VERSION,
        "extracted_fields": extracted_fields(record),
        "patient_evidence_fields": evidence_fields,
        "patient_evidence_count": len(evidence_fields),
        "patient_information_fields": information_fields,
        "patient_information_count": len(information_fields),
        "has_patient_evidence": has_patient_evidence(record),
        "has_patient_measurement": has_patient_measurement(record),
        "condition_mentions": list(
            patient_data.get("condition_mentions")
            or []
        ),
        "symptoms": list(
            patient_data.get("symptoms")
            or []
        ),
        "critical_signs": list(
            patient_data.get("critical_signs")
            or []
        ),
        "french_concepts_present": sorted(
            concept
            for concept, state in concept_states.items()
            if state == "PRESENT"
        ),
        "patient_motif_count": len(patient_motifs),
        "patient_rule_ids": patient_rule_ids,
        "knowledge_motif_count": len(knowledge_motifs),
        "knowledge_rule_ids": knowledge_rule_ids,
        "shared_rule_ids": shared_rule_ids(record),
        "all_rule_ids": all_rule_ids(record),
        "all_motif_count": len(all_rule_ids(record)),
        "has_medical_information": has_medical_information(record),
        "has_medical_knowledge": has_medical_knowledge(record),
        "medical_knowledge_fields": list(
            knowledge.get("extracted_fields")
            or []
        ),
        "correct_answer_count": len(
            knowledge.get("correct_answers")
            or []
        ),
        "critical_sign_conflicts": list(
            patient_meta.get("critical_sign_conflicts")
            or []
        ),
        "french_concept_conflicts": list(
            patient_meta.get("french_concept_conflicts")
            or []
        ),
        "review_required": route == "human_review",
        "review_reasons": sorted(set(review_reasons)),
        "ready_for_priority_annotation": (
            route == "protocol_annotation"
        ),
        "ready_for_controlled_synthesis": (
            route == "controlled_synthesis"
        )
    }


# ## Création du registre de routage

# Lit le checkpoint enrichi et écrit un objet JSON par ligne.
# Les identifiants et routes sont validés avant remplacement atomique.
def create_triage_routing(
    source_path: Path,
    routing_path: Path
) -> Counter:
    """Crée le registre de routage."""
    routing_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = routing_path.with_suffix(
        ".jsonl.tmp"
    )

    route_counts = Counter()
    seen_ids = set()

    with (
        source_path.open(
            "r",
            encoding="utf-8"
        ) as source_file,
        temporary_path.open(
            "w",
            encoding="utf-8"
        ) as output_file
    ):
        for line_number, line in enumerate(
            source_file,
            1
        ):
            if not line.strip():
                continue

            record = json.loads(line)
            record_id = record.get("id")

            if not record_id:
                raise ValueError(
                    f"Identifiant absent ligne {line_number}"
                )

            if record_id in seen_ids:
                raise ValueError(
                    f"Identifiant dupliqué : {record_id}"
                )

            seen_ids.add(record_id)

            routing_record = build_routing_record(record)
            route = routing_record["triage_route"]

            if route not in TriageRoutingConfig.ALLOWED_ROUTES:
                raise ValueError(
                    f"Route inconnue : {route}"
                )

            output_file.write(
                json.dumps(
                    routing_record,
                    ensure_ascii=False
                )
                + "\n"
            )

            route_counts[route] += 1

    temporary_path.replace(
        routing_path
    )

    return route_counts
