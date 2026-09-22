import ast

from scr.config import (
    DATASET_SPECS,
    SOURCE_REGISTRY,
    SourceSchema
)
from scr.data.cleaning import (
    assemble_input,
    audit_id,
    clean_text,
    content_hash,
    extract_completion,
    key_text,
    parse_labels
)


def canonical_qcm_content(
    raw: dict,
    family: str
) -> tuple[dict[str, str], list[str]]:
    """Canonicalise un QCM."""
    if family == "fmmcqa":
        options = raw.get("answers")

        if isinstance(options, str):
            options = ast.literal_eval(options)

        if isinstance(options, list) and len(options) <= 5:
            options = dict(zip("ABCDE", options))

        if not isinstance(options, dict):
            raise ValueError("Schéma des propositions non reconnu.")

        choices = {
            str(label).upper(): cleaned
            for label, option in options.items()
            if (
                str(label).upper() in "ABCDE"
                and (cleaned := clean_text(option))
            )
        }
    else:
        choices = {
            label: cleaned
            for label in "ABCDE"
            if (cleaned := clean_text(raw.get(f"answer_{label.lower()}")))
        }

    return choices, parse_labels(raw.get("correct_answers"))


def apply_dpo_content(record: dict, raw: dict) -> None:
    """Canonicalise une paire DPO."""
    record["chosen"] = extract_completion(
        raw.get("chosen"),
        record["question"]
    )
    record["rejected"] = extract_completion(
        raw.get("rejected"),
        record["question"]
    )
    record["label_type"] = clean_text(raw.get("label_type")) or "preference"
    record["feedback"] = clean_text(raw.get("feedback")) or None

    metadata = record["metadata"]

    for side in ("chosen", "rejected"):
        side_metadata = metadata.get(side)

        if isinstance(side_metadata, dict):
            record[f"{side}_score"] = side_metadata.get("score")
            record[f"{side}_rank"] = side_metadata.get("rank")

    if not record["chosen"] or not record["rejected"]:
        raise ValueError("Réponse DPO chosen/rejected manquante.")

    if key_text(record["chosen"]) == key_text(record["rejected"]):
        raise ValueError("DPO chosen et rejected identiques.")


class Canonicalizer:
    """Transforme une observation brute vers le schéma canonique."""

    def __init__(self, dataset: str) -> None:
        """Initialise le canonicaliseur."""
        self.dataset = dataset
        self.spec = DATASET_SPECS[dataset]
        self.source = self.spec.source
        self.family = self.spec.family
        self.language = SOURCE_REGISTRY[self.source].language

    def canonicalize(self, index: int, raw: dict) -> dict:
        """Canonicalise une observation."""
        source_id = self._source_id(index, raw)
        record = self._build_record(index, source_id, raw)

        try:
            self._apply_source_content(record, raw)
        except (ValueError, TypeError, SyntaxError, AttributeError) as error:
            record["flags"].add("SCHEMA_UNSUPPORTED")
            record["schema_error"] = f"{type(error).__name__}: {error}"

        record["raw_content_hash"] = content_hash(record)
        record["input"] = assemble_input(record)

        return record

    def _source_id(self, index: int, raw: dict) -> str:
        """Retourne l'identifiant source."""
        if self.spec.id_column:
            source_id = clean_text(raw.get(self.spec.id_column))

            if source_id:
                return source_id

        return f"{self.dataset}:{index}"

    def _build_record(
        self,
        index: int,
        source_id: str,
        raw: dict
    ) -> dict:
        """Construit le record canonique."""
        metadata = raw.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}

        return {
            "id": audit_id(self.dataset, index, source_id),
            "dataset": self.dataset,
            "source": self.source,
            "source_id": source_id,
            "source_index": int(index),
            "source_split": self.spec.source_split,
            "family": self.family,
            "task": "preference" if self.family == "dpo" else "medical_qa",
            "qtype": (
                clean_text(raw.get("qtype"))
                or SourceSchema.QTYPE_BY_FAMILY.get(self.family)
            ),
            "source_task": clean_text(raw.get("task")) or None,
            "source_qtype": clean_text(raw.get("qtype")) or None,
            "question_type": clean_text(raw.get("question_type")) or None,
            "language": self.language,
            "context": "",
            "question": "",
            "choices": {},
            "labels": [],
            "answer": "",
            "patient_age": None,
            "age_origin": None,
            "patient_sex": None,
            "sex_origin": None,
            "symptoms": [],
            "vital_signs": {},
            "medical_history": [],
            "duration": None,
            "evolution": None,
            "critical_signs": [],
            "priority_label": None,
            "priority_source": None,
            "chosen": "",
            "rejected": "",
            "feedback": None,
            "chosen_score": None,
            "chosen_rank": None,
            "rejected_score": None,
            "rejected_rank": None,
            "subject": "",
            "label_type": "",
            "metadata": metadata,
            "flags": set(),
            "notes": set(),
            "reviewer_excluded": False,
            "manual_allow": set(),
            "schema_error": None
        }

    def _apply_source_content(self, record: dict, raw: dict) -> None:
        """Applique le contenu source."""
        record["context"] = clean_text(raw.get("clinical_case"))

        question_column = SourceSchema.QUESTION_COLUMNS.get(
            self.family,
            "question"
        )
        record["question"] = clean_text(raw.get(question_column))
        record["subject"] = clean_text(
            raw.get("medical_subject", raw.get("subject_name"))
        )

        if self.family in SourceSchema.QCM_FAMILIES:
            self._apply_qcm_content(record, raw)
            return

        if self.family == "dpo":
            apply_dpo_content(record, raw)
            return

        answer_column = SourceSchema.ANSWER_COLUMNS.get(
            self.family,
            "answer"
        )
        record["answer"] = clean_text(raw.get(answer_column))

    def _apply_qcm_content(self, record: dict, raw: dict) -> None:
        """Applique le contenu QCM."""
        record["choices"], record["labels"] = canonical_qcm_content(
            raw,
            self.family
        )

        if self.family == "fmmcqa":
            declared = clean_text(raw.get("nbr_correct_answers"))

            if declared and int(declared) != len(record["labels"]):
                record["flags"].add("DECLARED_COUNT_MISMATCH")
