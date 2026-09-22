from pathlib import Path
from dataclasses import dataclass
from typing import Any

from scr.data.cleaning import clean_text
from scr.utils.serialization import read_json



@dataclass(frozen=True)
class Branch:
    rule_id: str
    section_fr: str
    section_en: str
    motif_fr: str
    motif_en: str
    french_level: str
    priority: int
    priority_label: str
    criterion_fr: str
    criterion_en: str
    automation_status: str
    automation: dict[str, Any] | None


def explicit_branches(protocol: dict[str, Any]) -> list[Branch]:
    """Extrait uniquement les branches FRENCH explicitement décrites."""
    branches = []
    mapping = protocol["priority_mapping"]["french_to_target"]
    for motif in protocol["motifs"]:
        for level, payload in motif["triage_levels"].items():
            if not payload:
                continue
            priority = int(mapping[level])
            branches.append(
                Branch(
                    rule_id=motif["rule_id"],
                    section_fr=motif["section"]["fr"],
                    section_en=motif["section"]["en"],
                    motif_fr=motif["motif"]["fr"].replace("\n", " ").strip(),
                    motif_en=motif["motif"]["en"].replace("\n", " ").strip(),
                    french_level=level,
                    priority=priority,
                    priority_label=payload.get("priority_label", ""),
                    criterion_fr=payload["fr"].replace("\n", " ").strip(),
                    criterion_en=payload["en"].replace("\n", " ").strip(),
                    automation_status=payload.get("automation_status", "human_or_not_encoded"),
                    automation=payload.get("automation")
                )
            )
    return branches


def unresolved_motifs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Liste les motifs sans branche dont cvc et tri_m divergent."""
    unresolved = []
    for motif in protocol["motifs"]:
        has_branch = any(motif["triage_levels"].values())
        if not has_branch and motif.get("cvc") != motif.get("tri_m"):
            unresolved.append(
                {
                    "rule_id": motif["rule_id"],
                    "motif": motif["motif"],
                    "cvc": motif.get("cvc"),
                    "tri_m": motif.get("tri_m"),
                    "reason": "no_explicit_branch_and_cvc_tri_m_disagree"
                }
            )
    return unresolved


def motifs_without_explicit_branch(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Liste tous les motifs exclus faute de branche clinique détaillée."""
    rows = []
    for motif in protocol["motifs"]:
        if any(motif["triage_levels"].values()):
            continue
        rows.append(
            {
                "rule_id": motif["rule_id"],
                "motif": motif["motif"],
                "cvc": motif.get("cvc"),
                "tri_m": motif.get("tri_m"),
                "reason": "no_explicit_branch"
            }
        )
    return rows

class AutomaticTriageEngineConfig:
    """Configuration du moteur FRENCH."""

    LEVEL_ORDER = {
        "1": 0,
        "2": 1,
        "3A": 2,
        "3B": 3,
        "4": 4,
        "5": 5
    }

    OPERATORS = {
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "contains_any"
    }


def load_triage_protocol(
    path: Path
) -> dict:
    """Charge et valide le protocole TRIAGE."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Protocole TRIAGE introuvable : {path}."
        )

    protocol = read_json(path)

    if not isinstance(protocol, dict):
        raise ValueError(
            "Le protocole TRIAGE doit être un objet JSON."
        )

    required = {
        "protocol_name",
        "protocol_version",
        "languages",
        "priority_mapping",
        "triage_levels",
        "sections",
        "motifs",
        "rule_policy"
    }

    missing = required - protocol.keys()

    if missing:
        raise ValueError(
            f"Protocole TRIAGE incomplet : {sorted(missing)}."
        )

    mapping_source = protocol.get(
        "priority_mapping",
        {}
    ).get(
        "french_to_target"
    )

    if not isinstance(mapping_source, dict):
        raise ValueError(
            "Mapping FRENCH vers TRIAGE invalide."
        )

    try:
        mapping = {
            clean_text(level).upper(): int(priority)
            for level, priority in mapping_source.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Mapping FRENCH vers TRIAGE invalide."
        ) from error

    expected_levels = set(
        AutomaticTriageEngineConfig.LEVEL_ORDER
    )

    if set(mapping) != expected_levels:
        raise ValueError(
            "Niveaux FRENCH invalides."
        )

    if set(mapping.values()) != {1, 2, 3}:
        raise ValueError(
            "Priorités TRIAGE invalides."
        )

    languages = {
        clean_text(language).lower()
        for language in protocol["languages"]
    }

    if languages != {"fr", "en"}:
        raise ValueError(
            "Langues TRIAGE invalides."
        )

    motifs = protocol["motifs"]

    if not isinstance(motifs, list) or not motifs:
        raise ValueError(
            "Protocole TRIAGE sans motif clinique."
        )

    rule_ids = [
        clean_text(
            motif.get("rule_id")
        )
        for motif in motifs
        if isinstance(motif, dict)
    ]

    if (
        len(rule_ids) != len(motifs)
        or not all(rule_ids)
        or len(rule_ids) != len(set(rule_ids))
    ):
        raise ValueError(
            "Identifiants de motifs FRENCH invalides."
        )

    protocol["mapping"] = mapping

    return protocol