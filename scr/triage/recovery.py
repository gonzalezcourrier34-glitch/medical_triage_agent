import re
import unicodedata

from scr.data.cleaning import clean_text


# ## Configuration

class TriageRecoveryConfig:
    """Configuration du recovery FRENCH."""

    MIN_SCORE = 2.0
    MIN_MARGIN = 1.0

    GENERIC_TERMS = {
        "infection",
        "pain",
        "douleur",
        "fever",
        "fievre",
        "bleeding",
        "saignement"
    }

    PATIENT_FIELDS = (
        "symptoms",
        "medical_history",
        "critical_signs",
        "condition_mentions"
    )

    KNOWLEDGE_FIELDS = (
        "symptoms",
        "conditions",
        "complications",
        "diagnostic_tests",
        "laboratory_findings",
        "imaging_findings",
        "risk_factors",
        "treatments",
        "medications",
        "procedures",
        "pathogens",
        "anatomy"
    )


# ## Recovery FRENCH

class TriageRecoveryMatcher:
    """Récupère les motifs FRENCH manqués."""

    def __init__(self, motif_lexicon: dict) -> None:
        """Initialise le matcher."""
        self.lexicon = self._build_index(motif_lexicon)

    # ## Matching

    def match_patient(self, record: dict, patient: dict) -> dict:
        """Recherche un motif FRENCH patient."""
        return self._match(
            self._build_text(
                record,
                patient,
                TriageRecoveryConfig.PATIENT_FIELDS
            ),
            patient,
            TriageRecoveryConfig.PATIENT_FIELDS,
            "patient"
        )

    def match_knowledge(
        self,
        record: dict,
        medical_knowledge: dict
    ) -> dict:
        """Recherche un motif FRENCH documentaire."""
        return self._match(
            self._build_text(
                record,
                medical_knowledge,
                TriageRecoveryConfig.KNOWLEDGE_FIELDS,
                include_answers=True
            ),
            medical_knowledge,
            TriageRecoveryConfig.KNOWLEDGE_FIELDS,
            "medical_knowledge"
        )

    # ## Résolution

    def _match(
        self,
        text: str,
        data: dict,
        fields: tuple,
        scope: str
    ) -> dict:
        """Résout le meilleur motif FRENCH."""
        if not text:
            return self._result(scope, "no_match")

        text_tokens = set(text.split())
        padded_text = f" {text} "

        structured = {
            field: self._normalize(self._value_text(data.get(field)))
            for field in fields
            if data.get(field)
        }

        structured_tokens = {
            field: set(value.split())
            for field, value in structured.items()
        }

        matches = [
            self._score_motif(
                padded_text,
                text_tokens,
                structured,
                structured_tokens,
                motif
            )
            for motif in self.lexicon
        ]

        matches = [
            match
            for match in matches
            if match["score"] > 0
        ]

        if not matches:
            return self._result(scope, "no_match")

        matches.sort(
            key=lambda match: (-match["score"], match["rule_id"])
        )

        best = matches[0]
        second = matches[1]["score"] if len(matches) > 1 else 0.0

        if best["score"] < TriageRecoveryConfig.MIN_SCORE:
            return self._result(
                scope,
                "weak_match",
                best["score"],
                matches[:5]
            )

        if best["score"] - second < TriageRecoveryConfig.MIN_MARGIN:
            return self._result(
                scope,
                "ambiguous_match",
                best["score"],
                matches[:5]
            )

        return self._result(
            scope,
            "unique_match",
            best["score"],
            [best]
        )

    # ## Lexique

    def _build_index(self, motif_lexicon: dict) -> list[dict]:
        """Construit l'index des motifs."""
        motifs = []

        for rule_id, motif in motif_lexicon.items():
            if not isinstance(motif, dict):
                continue

            terms = self._motif_terms(motif)

            if terms:
                motifs.append({
                    "rule_id": clean_text(rule_id),
                    "terms": terms
                })

        return motifs

    # ## Lexique

    def _motif_terms(self, motif: dict) -> list[dict]:
        """Retourne les termes FR et EN."""
        values = []

        for language in ("fr", "en"):
            terms = motif.get(language) or []

            if isinstance(terms, str):
                values.append(terms)
            elif isinstance(terms, list):
                values.extend(terms)

        labels = motif.get("motif") or {}

        for language in ("fr", "en"):
            label = labels.get(language)

            if isinstance(label, str):
                values.append(label)

        terms = {
            self._normalize(value)
            for value in values
            if isinstance(value, str)
        }

        return [
            {
                "text": term,
                "words": tuple(
                    word
                    for word in term.split()
                    if len(word) >= 4
                )
            }
            for term in sorted(terms)
            if term
            and term not in TriageRecoveryConfig.GENERIC_TERMS
        ]

    # ## Texte

    def _build_text(
        self,
        record: dict,
        data: dict,
        fields: tuple,
        include_answers: bool = False
    ) -> str:
        """Construit le texte de recovery."""
        parts = [
            record.get("context"),
            record.get("question")
        ]

        if include_answers:
            parts.extend(data.get("correct_answers") or [])

        parts.extend(
            self._value_text(data.get(field))
            for field in fields
        )

        return self._normalize(
            " ".join(str(part) for part in parts if part)
        )

    def _value_text(self, value) -> str:
        """Convertit une valeur médicale en texte."""
        if isinstance(value, dict):
            value = list(value.values())

        if isinstance(value, list):
            value = " ".join(str(item) for item in value if item)

        return clean_text(value)

    # ## Score

    def _score_motif(
        self,
        padded_text: str,
        text_tokens: set,
        structured: dict,
        structured_tokens: dict,
        motif: dict
    ) -> dict:
        """Score un motif FRENCH."""
        matched_terms = []
        score = 0.0

        for term in motif["terms"]:
            term_score = self._term_score(
                padded_text,
                text_tokens,
                term
            )

            if term_score:
                score += term_score
                matched_terms.append(term["text"])

        structured_hits = []

        for field, value in structured.items():
            padded_value = f" {value} "
            tokens = structured_tokens[field]

            if any(
                self._term_score(padded_value, tokens, term)
                for term in motif["terms"]
            ):
                structured_hits.append(field)

        score += len(structured_hits) * 0.5

        return {
            "rule_id": motif["rule_id"],
            "score": round(score, 4),
            "matched_terms": matched_terms,
            "structured_hits": structured_hits,
            "source": "recovery"
        }

    def _term_score(
        self,
        padded_text: str,
        tokens: set,
        term: dict
    ) -> float:
        """Score la présence d'un terme."""
        if f" {term['text']} " in padded_text:
            return 2.0

        words = term["words"]

        if len(words) < 2:
            return 0.0

        matched = sum(
            word in tokens
            for word in words
        )

        return 1.0 if matched / len(words) >= 0.75 else 0.0

    # ## Résultat

    def _result(
        self,
        scope: str,
        status: str,
        score: float = 0.0,
        candidates: list[dict] | None = None
    ) -> dict:
        """Construit le résultat."""
        return {
            "used": True,
            "scope": scope,
            "status": status,
            "score": round(score, 4),
            "candidates": candidates or []
        }

    # ## Normalisation

    def _normalize(self, value) -> str:
        """Normalise un texte."""
        text = unicodedata.normalize(
            "NFKD",
            clean_text(value).casefold()
        )

        text = "".join(
            char
            for char in text
            if not unicodedata.combining(char)
        )

        return re.sub(r"[^\w]+", " ", text).strip()