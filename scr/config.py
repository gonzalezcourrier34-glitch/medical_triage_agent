from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class PipelineConfig:
    """Paramètres stables du pipeline."""

    manual_rerun: bool = True
    manual_attempt_id: str = "attempt_01"
    seed: int = 42
    sft_target: int = 5_000
    dpo_target: int | None = None
    max_sft_per_group: int | None = None
    tokenizer_model: str = "Qwen/Qwen3-1.7B-Base"
    tokenizer_revision: str = "main"
    max_sequence_tokens: int = 2_048
    run_near_duplicates: bool = True
    near_threshold: float = 95.0
    lsh_threshold: float = 0.80
    lsh_num_perm: int = 64
    max_near_candidates: int = 1_000_000
    qa_sample_per_dataset: int = 200
    mode_llm: bool = False
    llm_batch_size: int = 20
    run_llm_judge: bool = False
    project_root_override: Path | None = None


CONFIG = PipelineConfig()


class SourceSchema:
    """Conventions des familles de datasets."""

    QCM_FAMILIES = {"fmmcqa", "mcqu", "mcqm"}
    QUESTION_COLUMNS = {"dpo": "prompt", "open": "Question", "oeq": "question"}
    ANSWER_COLUMNS = {"open": "Answer", "oeq": "answer"}
    QTYPE_BY_FAMILY = {
        "fmmcqa": "multiple_choice",
        "mcqu": "multiple_choice",
        "mcqm": "multiple_choice",
        "open": "open",
        "oeq": "open",
        "dpo": "preference"
    }


class TextConfig:
    """Champs textuels exploités par le pipeline."""

    TEXT_FIELDS = ("context", "question", "answer", "chosen", "rejected", "feedback")
    CLINICAL_INPUT_FIELDS = ("context", "question")


class PrivacyConfig:
    """Configure la détection Presidio."""

    SUPPORTED_LANGUAGES = (
        "fr",
        "en"
    )

    NLP_MODELS = {
        "fr": "fr_core_news_md",
        "en": "en_core_web_md"
    }

    SCORE_THRESHOLD = 0.60

    PRESIDIO_ENTITIES = (
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "IP_ADDRESS",
        "URL",
        "MEDICAL_LICENSE",
        "PERSON",
        "PATIENT_ID",
        "FR_NIR",
        "DATE_OF_BIRTH",
        "IDENTITY_INTRODUCTION"
    )

    AUTOMATIC_REDACTION_REPLACEMENTS = {
        "EMAIL_ADDRESS": "[EMAIL]",
        "PHONE_NUMBER": "[PHONE]",
        "IP_ADDRESS": "[IP]",
        "URL": "[URL]",
        "MEDICAL_LICENSE": "[IDENTIFIER]",
        "PATIENT_ID": "[IDENTIFIER]",
        "FR_NIR": "[IDENTIFIER]",
        "DATE_OF_BIRTH": "[DATE]",
        "IDENTITY_INTRODUCTION": "[NAME]"
    }

    AUTOMATIC_PII_CATEGORIES = set(
        AUTOMATIC_REDACTION_REPLACEMENTS
    )

    REVIEW_PII_CATEGORIES = {
        "PERSON"
    }

    CUSTOM_RECOGNIZERS = {
        "PATIENT_ID": {
            "score": 0.80,
            "patterns": [
                (
                    r"\b(?:patient\s*(?:id|identifier)|"
                    r"n[°ºo]\s*(?:de\s*)?dossier|"
                    r"dossier\s*(?:id|n[°ºo])|"
                    r"medical\s*record\s*(?:id|number)|mrn)"
                    r"\s*[:=#-]?\s*[A-Za-z0-9_-]{3,}\b"
                )
            ]
        },
        "FR_NIR": {
            "score": 0.85,
            "patterns": [
                r"\b[12]\s?\d{2}(?:\s?\d{2}){5}\s?\d{2}\b"
            ]
        },
        "DATE_OF_BIRTH": {
            "score": 0.80,
            "patterns": [
                (
                    r"\b(?:né|née)\s+(?:le\s+)?"
                    r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
                    r"\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"
                ),
                (
                    r"\b(?:born|date\s+of\s+birth|dob)"
                    r"\s*(?:on|:|=)?\s*"
                    r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
                    r"\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"
                )
            ]
        },
        "IDENTITY_INTRODUCTION": {
            "score": 0.85,
            "patterns": [
                (
                    r"\b(?:my\s+name\s+is|i\s+am\s+called|"
                    r"i['’]?m\s+called|je\s+m['’]appelle|"
                    r"mon\s+nom\s+est)\s+"
                    r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
                    r"(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’-]+){0,3}\b"
                )
            ]
        }
    }


class ClinicalPrivacyPatterns:
    """Règles cliniques utilisées pour calculer un âge."""

    BIRTH_DATE_FR = re.compile(
        r"(?i)\b(?:né|née)\s+(?:le\s+)?"
        r"(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
        r"\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"
    )

    BIRTH_DATE_EN = re.compile(
        r"(?i)\b(?:born|date\s+of\s+birth|dob)"
        r"\s*(?:on|:|=)?\s*"
        r"(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
        r"\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"
    )

    PATIENT_AGE_LIMITS = {
        "min": 0,
        "max": 120
    }


class QualityConfig:
    """Flags et contraintes de qualité."""

    OPEN_BLOCKING_FLAGS = {
        "UNINFORMATIVE_ANSWER",
        "RESOURCE_ONLY_ANSWER",
        "QUESTION_AS_ANSWER"
    }
    REVIEWABLE = {
        "PII_CANDIDATE",
        "PII_HUMAN_REVIEW",
        "NEEDS_CONTEXT",
        "EXTERNAL_REFERENCE",
        "TEXT_ANOMALY",
        "NEAR_REVIEW"
    }
    INFORMATIVE_FLAGS = {
        "HIGH_RISK",
        "DPO_LENGTH_BIAS",
        "LANGUAGE_MISMATCH"
    }


class ContextPatterns:
    """Patterns de dépendance et d'anomalie textuelle."""

    CONTEXT_REF = re.compile(
        r"(?i)\b(?:ci-dessus|ci-dessous|comme indiqué précédemment|"
        r"question précédente|résultats précédents|voir plus haut|"
        r"as described above|as mentioned above|previous question|"
        r"previous results|above-mentioned)\b"
    )
    EXTERNAL_REF = re.compile(
        r"(?i)\b(?:figure|image|illustration|tableau|annexe|fig\.?|table|appendix)"
        r"\s*(?:ci[- ]jointe?|ci[- ]dessous|suivant(?:e)?|above|below|"
        r"n[°ºo]?\s*\d+|\d+)"
    )
    ANOMALY = re.compile(
        r"\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f]|"
        r"(?i:\b(\w+)\s+\1\s+\1\b)|([^\w\s])\2{9,}"
    )


@dataclass(frozen=True)
class SourceSpec:
    """Décrit une source médicale utilisée par le pipeline."""

    repo: str
    license: str
    language: str
    allowed: bool
    upstream_repo: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Décrit un fichier source et sa place dans le pipeline."""

    relative_path: str
    source: str
    source_split: str
    family: str
    id_column: str | None


SOURCE_REGISTRY: dict[str, SourceSpec] = {
    "FrenchMedMCQA": SourceSpec(
        repo="qanastek/frenchmedmcqa",
        license="Apache-2.0",
        language="fr",
        allowed=True
    ),
    "MediQAl": SourceSpec(
        repo="ANR-MALADES/MediQAl",
        license="CC-BY-4.0",
        language="fr",
        allowed=True
    ),
    "MedQuad-KV": SourceSpec(
        repo="keivalya/MedQuad-MedicalQnADataset",
        upstream_repo="abachaa/MedQuAD",
        license="CC-BY-4.0",
        language="en",
        allowed=True
    ),
    "UltraMedical-Preference": SourceSpec(
        repo="TsinghuaC3I/UltraMedical-Preference",
        license="MIT",
        language="en",
        allowed=True
    )
}


DATASET_SPECS: dict[str, DatasetSpec] = {}

for source_split, filename in (
    ("train", "train.json"),
    ("validation", "dev.json"),
    ("test", "test.json")
):
    DATASET_SPECS[f"fmmcqa_{source_split}"] = DatasetSpec(
        relative_path=f"frenchmedmcqa/{filename}",
        source="FrenchMedMCQA",
        source_split=source_split,
        family="fmmcqa",
        id_column="id"
    )
    DATASET_SPECS[f"umed_{source_split}"] = DatasetSpec(
        relative_path=f"ultramedical_preference/data/{filename}",
        source="UltraMedical-Preference",
        source_split=source_split,
        family="dpo",
        id_column="prompt_id"
    )

for family in ("mcqu", "mcqm"):
    for source_split in ("train", "validation", "test"):
        DATASET_SPECS[f"mqal_{family}_{source_split}"] = DatasetSpec(
            relative_path=f"mediqal/{family}/{source_split}.json",
            source="MediQAl",
            source_split=source_split,
            family=family,
            id_column="id"
        )

DATASET_SPECS |= {
    "mqal_oeq_test": DatasetSpec(
        relative_path="mediqal/oeq/test.json",
        source="MediQAl",
        source_split="test",
        family="oeq",
        id_column="id"
    ),
    "mquad": DatasetSpec(
        relative_path="medquad/medDataset_processed.csv",
        source="MedQuad-KV",
        source_split="unsplit",
        family="open",
        id_column=None
    )
}
