import re

from scr.data.cleaning import clean_text, key_text
from scr.triage.knowledge_matching import match_french_motif_candidates
from scr.triage.medical_knowledge_patterns import (
    ANATOMY_TERMS,
    MEDICAL_KNOWLEDGE_PATTERNS,
    QUESTION_TAIL_PATTERN,
    VALUE_EXCLUSION_PATTERNS
)


# ## Configuration

class MedicalKnowledgeExtractionConfig:
    """Configuration de l'extraction médicale."""

    VERSION = "3.0"
    KNOWLEDGE_FIELDS = ("context", "question")


# ## Extracteur

class MedicalKnowledgeExtractor:
    """Extrait la connaissance médicale bilingue."""

    def __init__(self, motif_lexicon: dict) -> None:
        self.motif_lexicon = motif_lexicon

    # ## Extraction

    # Construit la knowledge depuis le texte source et la bonne réponse.
    # Les distracteurs restent traçables mais ne deviennent jamais des faits.
    def extract(self, record: dict) -> dict:
        """Extrait la connaissance médicale."""
        labels = self._correct_labels(record)
        answers = self._extract_correct_answers(record, labels)
        text = self._knowledge_text(record, answers)

        knowledge = {
            "question": clean_text(record.get("question")) or None,
            "context": clean_text(record.get("context")) or None,
            "source_answer": clean_text(record.get("answer")) or None,
            "correct_labels": labels,
            "correct_answers": answers,
            "distractors": self._extract_distractors(record, labels)
        }

        for field, patterns in MEDICAL_KNOWLEDGE_PATTERNS.items():
            knowledge[field] = self._extract_values(
                text,
                patterns
            )

        knowledge["anatomy"] = self._extract_anatomy(
            text,
            record.get("language")
        )

        knowledge["french_motif_candidates"] = (
            match_french_motif_candidates(
                text,
                record.get("language"),
                self.motif_lexicon
            )
        )

        fields = self._extracted_fields(knowledge)

        has_source_information = bool(
            answers or fields
        )

        knowledge.update({
            "has_correct_answer": bool(answers),
            "has_french_motif": bool(
                knowledge["french_motif_candidates"]
            ),
            "extracted_fields": fields,
            "has_structured_medical_information": bool(fields),
            "has_source_medical_information": has_source_information,
            "has_medical_information": has_source_information,
            "extraction": {
                "method": "deterministic_medical_knowledge_extraction_v3",
                "version": MedicalKnowledgeExtractionConfig.VERSION,
                "source_fields": list(
                    MedicalKnowledgeExtractionConfig.KNOWLEDGE_FIELDS
                ),
                "uses_correct_answers": bool(answers),
                "distractors_used_as_knowledge": False,
                "source_clean_content_hash": record.get(
                    "clean_content_hash"
                )
            }
        })

        return knowledge

    # ## Texte source

    # Utilise contexte, question et uniquement les réponses validées.
    # Les réponses fausses ne sont jamais injectées dans la knowledge.
    def _knowledge_text(
        self,
        record: dict,
        answers: list[str]
    ) -> str:
        """Construit le texte de connaissance."""
        values = [
            clean_text(record.get(field))
            for field in MedicalKnowledgeExtractionConfig.KNOWLEDGE_FIELDS
        ]

        values.extend(answers)

        return " ".join(
            value
            for value in values
            if value
        )

    # ## Extraction générique

    # Applique les regex externes et nettoie les fragments QCM.
    # Rejette les captures trop longues, génériques ou mal formées.
    def _extract_values(
        self,
        text: str,
        patterns: tuple[re.Pattern, ...]
    ) -> list[str]:
        """Extrait des valeurs médicales."""
        values = []

        for pattern in patterns:
            for match in pattern.finditer(text):
                value = clean_text(
                    match.group(1)
                )

                value = clean_text(
                    QUESTION_TAIL_PATTERN.sub(
                        "",
                        value
                    )
                )

                if self._valid_value(value):
                    values.append(value)

        return self._unique_texts(values)

    def _valid_value(
        self,
        value: str
    ) -> bool:
        """Valide une capture médicale."""
        if (
            not value
            or len(value) < 3
            or len(value) > 240
            or len(value.split()) > 30
        ):
            return False

        return not any(
            pattern.search(value)
            for pattern in VALUE_EXCLUSION_PATTERNS
        )

    # ## Anatomie

    # Recherche uniquement les structures anatomiques explicites.
    # Aucun organe absent du texte n'est ajouté par inférence.
    def _extract_anatomy(
        self,
        text: str,
        language: str | None
    ) -> list[str]:
        """Extrait les structures anatomiques."""
        language = clean_text(
            language
        ).casefold()

        if language.startswith("fr"):
            language = "fr"
        elif language.startswith("en"):
            language = "en"
        else:
            return []

        detected = []

        for term in ANATOMY_TERMS.get(
            language,
            set()
        ):
            if re.search(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                text,
                re.IGNORECASE
            ):
                detected.append(term)

        return sorted(detected)

    # ## Champs extraits

    # Recense uniquement les catégories médicales réellement renseignées.
    # Les champs documentaires restent exclus du comptage.
    def _extracted_fields(
        self,
        knowledge: dict
    ) -> list[str]:
        """Retourne les catégories extraites."""
        fields = list(
            MEDICAL_KNOWLEDGE_PATTERNS
        ) + ["anatomy"]

        return [
            field
            for field in fields
            if knowledge.get(field)
        ]

    # ## Labels

    def _correct_labels(
        self,
        record: dict
    ) -> list[str]:
        """Retourne les labels corrects."""
        raw = (
            record.get("labels")
            if record.get("labels") is not None
            else record.get("correct_answers")
        )

        if raw is None:
            return []

        values = (
            re.split(r"[,;/|\s]+", raw)
            if isinstance(raw, str)
            else list(raw)
            if isinstance(raw, (list, tuple, set))
            else [raw]
        )

        return list(dict.fromkeys(
            label
            for value in values
            if re.fullmatch(
                r"[A-E]",
                label := clean_text(value).upper()
            )
        ))

    # ## Réponses correctes

    def _extract_correct_answers(
        self,
        record: dict,
        labels: list[str]
    ) -> list[str]:
        """Extrait les réponses correctes."""
        choices = record.get("choices") or {}
        normalized = {}

        if isinstance(choices, dict):
            normalized = {
                clean_text(key).upper(): clean_text(value)
                for key, value in choices.items()
                if clean_text(value)
            }

        answers = [
            normalized[label]
            for label in labels
            if label in normalized
        ]

        direct = clean_text(
            record.get("answer")
        )

        if direct:
            answers.append(
                normalized.get(
                    direct.upper(),
                    direct
                )
            )

        return self._unique_texts(answers)

    # ## Distracteurs

    def _extract_distractors(
        self,
        record: dict,
        labels: list[str]
    ) -> list[dict]:
        """Extrait les distracteurs."""
        choices = record.get("choices") or {}

        if not isinstance(choices, dict):
            return []

        labels = set(labels)

        return [
            {
                "label": clean_text(label).upper(),
                "text": clean_text(value)
            }
            for label, value in choices.items()
            if (
                clean_text(value)
                and clean_text(label).upper() not in labels
            )
        ]

    # ## Résumé

    def summarize(
        self,
        knowledge: dict
    ) -> dict:
        """Résume l'extraction médicale."""
        return {
            "has_correct_answer": bool(
                knowledge.get("has_correct_answer")
            ),
            "has_french_motif": bool(
                knowledge.get("has_french_motif")
            ),
            "has_structured_medical_information": bool(
                knowledge.get(
                    "has_structured_medical_information"
                )
            ),
            "has_source_medical_information": bool(
                knowledge.get(
                    "has_source_medical_information"
                )
            ),
            "extracted_fields": list(
                knowledge.get("extracted_fields")
                or []
            )
        }

    # ## Déduplication

    def _unique_texts(
        self,
        values: list[str]
    ) -> list[str]:
        """Déduplique les textes."""
        unique = {}

        for value in values:
            value = clean_text(value)

            if value:
                unique.setdefault(
                    key_text(value).casefold(),
                    value
                )

        return list(unique.values())