# scr/triage/knowledge_matching.py

import re
import unicodedata
from copy import deepcopy

from scr.triage.french_motif_aliases import (
    build_french_motif_aliases
)

# ## Configuration

# Exclut les termes trop génériques pour identifier seuls un motif FRENCH.
# Les expressions composées contenant ces mots restent utilisables.
FRENCH_MOTIF_GENERIC_TERMS = {
    "fr": {
        "douleur", "douleurs", "maladie", "infection", "traumatisme",
        "urgence", "patient", "symptome", "symptomes", "signe", "signes",
        "froid", "chaud", "rouge", "masse", "estomac", "intestins",
        "enfant", "comportement"
    },
    "en": {
        "pain", "painful", "disease", "infection", "trauma",
        "emergency", "patient", "symptom", "symptoms", "sign", "signs",
        "cold", "warm", "red", "mass", "stomach", "intestines",
        "child", "behavior"
    }
}


# ## Normalisation

def normalize_french_motif_term(value: str | None) -> str:
    """Normalise un texte pour le matching FRENCH."""
    if not value:
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )
    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value.casefold()
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def _normalize_language(
    language: str | None
) -> str | None:
    """Normalise la langue."""
    value = str(
        language or ""
    ).strip().casefold()

    if value.startswith("fr"):
        return "fr"

    if value.startswith("en"):
        return "en"

    return None


# ## Lexique

def build_french_motif_lexicon(
    protocol: dict
) -> dict[str, dict]:
    """Construit le lexique FRENCH bilingue."""
    aliases = build_french_motif_aliases(
        protocol
    )
    lexicon = {}

    for motif in protocol.get("motifs", []):
        rule_id = str(
            motif.get("rule_id") or ""
        )

        if not rule_id:
            continue

        names = motif.get("motif") or {}
        sections = motif.get("section") or {}
        keywords = motif.get("keywords") or {}
        extra = aliases.get(
            rule_id,
            {}
        )

        terms = {
            language: _prepare_terms([
                names.get(language),
                *keywords.get(language, []),
                *extra.get(language, [])
            ])
            for language in ("fr", "en")
        }

        if not terms["fr"] and not terms["en"]:
            continue

        lexicon[rule_id] = {
            "section": {
                "fr": sections.get("fr"),
                "en": sections.get("en")
            },
            "motif": {
                "fr": names.get("fr"),
                "en": names.get("en")
            },
            **terms
        }

    return lexicon


def _prepare_terms(
    values: str | list | tuple | set | None
) -> list[str]:
    """Prépare les termes d'un motif."""
    if values is None:
        return []

    values = (
        values
        if isinstance(
            values,
            (list, tuple, set)
        )
        else [values]
    )

    terms = {
        term
        for value in values
        if (
            term := normalize_french_motif_term(
                value
            )
        )
    }

    return sorted(
        terms,
        key=lambda term: (
            -len(term),
            term
        )
    )


# ## Matching

def french_motif_term_matches(
    normalized_text: str,
    normalized_term: str
) -> bool:
    """Teste un terme FRENCH."""
    if not normalized_text or not normalized_term:
        return False

    return bool(
        re.search(
            rf"(?<!\w)"
            rf"{re.escape(normalized_term)}"
            rf"(?!\w)",
            normalized_text
        )
    )


def match_french_motif_candidates(
    text: str,
    language: str | None,
    motif_lexicon: dict[str, dict]
) -> list[dict]:
    """Retourne les motifs compatibles."""
    text = normalize_french_motif_term(
        text
    )
    language = _normalize_language(
        language
    )

    if not text or language is None:
        return []

    generic = FRENCH_MOTIF_GENERIC_TERMS[
        language
    ]
    candidates = []

    for rule_id, motif in motif_lexicon.items():
        matched = {
            term
            for term in motif.get(
                language,
                []
            )
            if (
                term not in generic
                and french_motif_term_matches(
                    text,
                    term
                )
            )
        }

        if not matched:
            continue

        candidates.append({
            "rule_id": rule_id,
            "section": deepcopy(
                motif.get(
                    "section",
                    {}
                )
            ),
            "motif": deepcopy(
                motif.get(
                    "motif",
                    {}
                )
            ),
            "matched_terms": sorted(
                matched,
                key=lambda term: (
                    -len(term),
                    term
                )
            )
        })

    return sorted(
        candidates,
        key=_candidate_sort_key
    )


# ## Tri

def _candidate_sort_key(
    candidate: dict
) -> tuple:
    """Construit la clé de tri."""
    terms = candidate.get(
        "matched_terms",
        []
    )

    longest = max(
        (
            len(term)
            for term in terms
        ),
        default=0
    )

    return (
        -longest,
        str(
            candidate.get(
                "rule_id",
                ""
            )
        )
    )