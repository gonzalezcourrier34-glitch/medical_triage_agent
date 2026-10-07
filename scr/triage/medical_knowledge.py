import re

from scr.data.cleaning import clean_text, key_text
from scr.triage.knowledge_matching import match_french_motif_candidates
from scr.triage.medical_knowledge_patterns import (
    ANATOMY_TERMS,
    MEDICAL_KNOWLEDGE_PATTERNS,
    QUESTION_TAIL_PATTERN,
    VALUE_EXCLUSION_PATTERNS,
    QUESTION_INTENT_PATTERNS,
    QUESTION_SUBJECT_PATTERNS
)


# Configuration extraction médicale.
class MedicalKnowledgeExtractionConfig:
    """Configuration de l'extraction médicale."""

    VERSION = "3.1"
    KNOWLEDGE_FIELDS = ("context", "question")


# Extrait les connaissances médicales FR/EN sans utiliser les distracteurs.
class MedicalKnowledgeExtractor:
    """Extrait la connaissance médicale bilingue."""

    def __init__(self, motif_lexicon: dict) -> None:
        self.motif_lexicon = motif_lexicon

    # Détecte ce que demande explicitement la question source.
    def _detect_intent(self, record: dict) -> str | None:
        """Détecte l'intention médicale."""
        question = clean_text(record.get("question"))

        for intent, pattern in QUESTION_INTENT_PATTERNS:
            if pattern.search(question):
                return intent

        return None

    # Extrait la pathologie ou le sujet demandé par la question.
    def _extract_subject(self, record: dict) -> str | None:
        """Extrait le sujet médical de la question."""
        question = clean_text(record.get("question"))

        for pattern in QUESTION_SUBJECT_PATTERNS:
            if match := pattern.match(question):
                return clean_text(match.group(1)) or None

        return None

    # Complète les champs lorsque la réponse possède une structure exploitable.
    def _apply_intent(
        self,
        knowledge: dict,
        intent: str | None,
        answers: list[str]
    ) -> None:
        """Exploite les structures propres à l'intention."""
        if not intent or not answers:
            return

        text = " ".join(answers)

        if intent == "symptoms":
            values = self._extract_frequency_items(text)

            knowledge["symptoms"] = self._unique_texts(
                knowledge.get("symptoms", []) + values
            )
            
    # Construit knowledge depuis contexte, question et réponses valides.
    def extract(self, record: dict) -> dict:
        """Extrait la connaissance médicale."""
        labels = self._correct_labels(record)
        answers = self._extract_correct_answers(record, labels)
        intent = self._detect_intent(record)
        subject = self._extract_subject(record)
        text = self._knowledge_text(record, answers)

        knowledge = {
            "question": clean_text(record.get("question")) or None,
            "question_subject": subject,
            "context": clean_text(record.get("context")) or None,
            "source_answer": clean_text(record.get("answer")) or None,
            "intent": intent,
            "correct_labels": labels,
            "correct_answers": answers,
            "distractors": self._extract_distractors(record, labels)
        }

        # Extraction générale.
        for field, patterns in MEDICAL_KNOWLEDGE_PATTERNS.items():
            knowledge[field] = self._extract_values(
                text,
                patterns
            )

        # Extraction générale.
        for field, patterns in MEDICAL_KNOWLEDGE_PATTERNS.items():
            knowledge[field] = self._extract_values(
                text,
                patterns
            )

        # Extraction guidée par l'intention.
        self._apply_intent(
            knowledge,
            intent,
            answers
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
        has_source = bool(answers or fields)

        knowledge.update({
            "has_correct_answer": bool(answers),
            "has_french_motif": bool(
                knowledge["french_motif_candidates"]
            ),
            "extracted_fields": fields,
            "has_structured_medical_information": bool(fields),
            "has_source_medical_information": has_source,
            "has_medical_information": has_source,
            "extraction": {
                "method": "deterministic_medical_knowledge_extraction_v3_1",
                "version": MedicalKnowledgeExtractionConfig.VERSION,
                "intent": intent,
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

    # Assemble uniquement les sources autorisées pour les regex.
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

        return " ".join(value for value in values if value)

    # Extrait les items médicaux associés à une fréquence.
    def _extract_frequency_items(self, text: str) -> list[str]:
        """Extrait les items suivis d'une fréquence."""
        pattern = re.compile(
            r"([A-Za-z][A-Za-z /'-]{2,80}?)\s+"
            r"(?:\d+(?:\.\d+)?%|\d+/\d+)"
        )

        values = []

        for match in pattern.finditer(text):
            value = clean_text(match.group(1))

            if not self._valid_value(value):
                continue

            if "frequency of" in key_text(value):
                continue

            values.append(value)

        return self._unique_texts(values)
    
    # Applique les regex puis nettoie les captures parasites.
    def _extract_values(
        self,
        text: str,
        patterns: tuple[re.Pattern, ...]
    ) -> list[str]:
        """Extrait des valeurs médicales."""
        values = []

        for pattern in patterns:
            for match in pattern.finditer(text):
                value = clean_text(match.group(1))
                value = clean_text(
                    QUESTION_TAIL_PATTERN.sub("", value)
                )

                if self._valid_value(value):
                    values.append(value)

        return self._unique_texts(values)

    # Écarte les captures vides, excessives ou éditoriales.
    def _valid_value(self, value: str) -> bool:
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

    # Recherche seulement les structures anatomiques présentes dans le texte.
    def _extract_anatomy(
        self,
        text: str,
        language: str | None
    ) -> list[str]:
        """Extrait les structures anatomiques."""
        language = clean_text(language).casefold()

        if language.startswith("fr"):
            language = "fr"
        elif language.startswith("en"):
            language = "en"
        else:
            return []

        return sorted(
            term
            for term in ANATOMY_TERMS.get(language, set())
            if re.search(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                text,
                re.IGNORECASE
            )
        )

    # Retourne uniquement les catégories médicales réellement structurées.
    def _extracted_fields(self, knowledge: dict) -> list[str]:
        """Retourne les catégories extraites."""
        fields = list(MEDICAL_KNOWLEDGE_PATTERNS) + ["anatomy"]

        return [
            field
            for field in fields
            if knowledge.get(field)
        ]

    # Normalise les labels QCM corrects A-E.
    def _correct_labels(self, record: dict) -> list[str]:
        """Retourne les labels corrects."""
        raw = (
            record.get("labels")
            if record.get("labels") is not None
            else record.get("correct_answers")
        )

        if raw is None:
            return []

        if isinstance(raw, str):
            values = re.split(r"[,;/|\s]+", raw)
        elif isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            values = [raw]

        return list(dict.fromkeys(
            label
            for value in values
            if re.fullmatch(
                r"[A-E]",
                label := clean_text(value).upper()
            )
        ))

    # Résout les labels QCM puis ajoute la réponse directe si disponible.
    def _extract_correct_answers(
        self,
        record: dict,
        labels: list[str]
    ) -> list[str]:
        """Extrait les réponses correctes."""
        choices = record.get("choices") or {}

        normalized = {
            clean_text(key).upper(): clean_text(value)
            for key, value in choices.items()
            if clean_text(value)
        } if isinstance(choices, dict) else {}

        answers = [
            normalized[label]
            for label in labels
            if label in normalized
        ]

        direct = clean_text(record.get("answer"))

        if direct:
            answers.append(
                normalized.get(direct.upper(), direct)
            )

        return self._unique_texts(answers)

    # Conserve les distracteurs pour audit mais jamais comme connaissance.
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

    # Produit le résumé utilisé par l'extracteur général.
    def summarize(self, knowledge: dict) -> dict:
        """Résume l'extraction médicale."""
        return {
            "intent": knowledge.get("intent"),
            "has_correct_answer": bool(
                knowledge.get("has_correct_answer")
            ),
            "has_french_motif": bool(
                knowledge.get("has_french_motif")
            ),
            "has_structured_medical_information": bool(
                knowledge.get("has_structured_medical_information")
            ),
            "has_source_medical_information": bool(
                knowledge.get("has_source_medical_information")
            ),
            "extracted_fields": list(
                knowledge.get("extracted_fields") or []
            )
        }

    # Déduplique sans modifier le texte original conservé.
    def _unique_texts(self, values: list[str]) -> list[str]:
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