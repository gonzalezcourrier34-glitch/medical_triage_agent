
# ## Configuration

# Termes trop génériques pour identifier seuls un motif FRENCH.
# Ils peuvent exister dans le protocole mais ne deviennent pas des alias.
ALIAS_EXCLUDED_TERMS = {
    "fr": {
        "douleur", "douleurs", "maladie", "infection", "traumatisme",
        "urgence", "patient", "symptome", "symptomes", "signe", "signes",
        "froid", "chaud", "rouge", "masse", "estomac", "intestins",
        "comportement", "enfant", "fièvre", "fievre"
    },
    "en": {
        "pain", "disease", "infection", "trauma", "emergency",
        "patient", "symptom", "symptoms", "sign", "signs",
        "painful", "cold", "warm", "red", "mass", "stomach", "intestines",
        "behavior", "child", "fever"
    }
}


# ## Extraction

# Récupère les expressions textuelles des règles automatiques.
# Les constantes numériques et autres critères structurés sont ignorés.
def _automation_terms(criteria: dict | list | None) -> set[str]:
    """Extrait les termes textuels d'une automation."""
    if isinstance(criteria, list):
        return {
            term
            for item in criteria
            for term in _automation_terms(item)
        }

    if not isinstance(criteria, dict):
        return set()

    terms = set()

    if (
        criteria.get("field") == "text"
        and criteria.get("operator") == "contains_any"
    ):
        terms.update(
            str(value).strip()
            for value in criteria.get("value") or []
            if str(value).strip()
        )

    for key in ("all", "any"):
        terms.update(
            _automation_terms(criteria.get(key))
        )

    return terms


# ## Alias

# Conserve uniquement les termes d'automation absents du motif et des keywords.
# La langue est déterminée par présence dans les vocabulaires FR/EN du motif.
def build_french_motif_aliases(
    protocol: dict
) -> dict[str, dict[str, set[str]]]:
    """Construit les alias supplémentaires FRENCH."""
    aliases = {}

    for motif in protocol.get("motifs", []):
        rule_id = str(
            motif.get("rule_id") or ""
        )

        if not rule_id:
            continue

        known = {
            language: {
                str(value).strip().casefold()
                for value in [
                    (motif.get("motif") or {}).get(language),
                    *((motif.get("keywords") or {}).get(language) or [])
                ]
                if value
            }
            for language in ("fr", "en")
        }

        automation_terms = set()

        for level in (motif.get("triage_levels") or {}).values():
            if not isinstance(level, dict):
                continue

            automation_terms.update(
                _automation_terms(
                    level.get("automation")
                )
            )

        rule_aliases = {
            "fr": set(),
            "en": set()
        }

        for term in automation_terms:
            normalized = term.strip().casefold()

            if not normalized:
                continue

            for language in ("fr", "en"):
                if normalized in known[language]:
                    continue

                if normalized in ALIAS_EXCLUDED_TERMS[language]:
                    continue

                # Une expression déjà connue dans l'autre langue
                # n'est pas dupliquée artificiellement.
                if normalized in known[
                    "en" if language == "fr" else "fr"
                ]:
                    continue

                if _looks_like_language(
                    term,
                    language
                ):
                    rule_aliases[language].add(term)

        rule_aliases = {
            language: values
            for language, values in rule_aliases.items()
            if values
        }

        if rule_aliases:
            aliases[rule_id] = rule_aliases

    return aliases


# ## Langue

# Utilise seulement quelques marqueurs simples.
# En cas de doute, aucun alias n'est créé automatiquement.
def _looks_like_language(
    value: str,
    language: str
) -> bool:
    """Estime la langue d'un alias."""
    text = value.casefold()

    french_markers = {
        "é", "è", "ê", "à", "â", "ç", "œ",
        "douleur", "fièvre", "membre", "grossesse",
        "saignement", "brûlure", "perte", "arrêt",
        "déficit", "courant", "domestique"
    }

    english_markers = {
        " pain", "fever", "blood", "bleeding", "loss",
        "heart", "pressure", "limb", "burn", "pregnancy",
        "shock", "seizure", "foreign body", "swelling"
    }

    markers = (
        french_markers
        if language == "fr"
        else english_markers
    )

    return any(
        marker in text
        for marker in markers
    )


# ## Validation

# Vérifie que les alias pointent uniquement vers des rule_id existants.
# Signale aussi les doublons déjà présents dans les keywords.
def validate_french_motif_aliases(
    protocol: dict,
    aliases: dict
) -> dict:
    """Valide les alias FRENCH."""
    motifs = {
        str(motif["rule_id"]): motif
        for motif in protocol.get("motifs", [])
        if motif.get("rule_id")
    }

    unknown = sorted(
        set(aliases) - set(motifs)
    )

    duplicates = []

    for rule_id, languages in aliases.items():
        motif = motifs.get(rule_id) or {}

        for language, values in languages.items():
            known = {
                str(value).strip().casefold()
                for value in (
                    (motif.get("keywords") or {}).get(
                        language,
                        []
                    )
                )
            }

            duplicates.extend(
                {
                    "rule_id": rule_id,
                    "language": language,
                    "alias": value
                }
                for value in values
                if value.casefold() in known
            )

    if unknown:
        raise ValueError(
            f"rule_id inconnus : {unknown}"
        )

    return {
        "rules_with_aliases": len(aliases),
        "aliases": sum(
            len(values)
            for languages in aliases.values()
            for values in languages.values()
        ),
        "duplicates": duplicates
    }