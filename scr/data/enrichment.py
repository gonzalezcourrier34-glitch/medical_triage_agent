from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from scr.config import CONFIG
from scr.data.cleaning import (
    QualityValidator,
    clean_text,
    content_hash,
    key_text
)
from scr.data.privacy import PrivacyProcessor
from scr.utils.hashing import digest


DetectorFactory.seed = CONFIG.seed


class RecordEnricher:
    """Applique les enrichissements déterministes pré-admission."""

    def __init__(self, record: dict, medical: dict) -> None:
        """Initialise l'enrichisseur."""
        self.record = record
        self.medical = medical
        self.flags = set(record.get("flags") or [])
        self.record["flags"] = self.flags
        self.privacy = PrivacyProcessor(record)

    def process(self) -> None:
        """Enrichit le record."""
        self._preserve_original_keys()
        self._detect_language()
        self._process_privacy()
        self._validate_quality()
        self._apply_medical_details()

    def _preserve_original_keys(self) -> None:
        """Conserve les empreintes initiales."""
        self.record["original_input_key"] = digest(
            key_text(self.record.get("input"))
        )

        context = self.record.get("context")

        self.record["original_context_key"] = (
            digest(key_text(context))
            if context
            else None
        )

    def _process_privacy(self) -> None:
        """Anonymise le contenu."""
        self.privacy.apply_automatic_redactions()

    def _validate_quality(self) -> None:
        """Valide et hache le contenu."""
        QualityValidator(self.record).validate()

        self.record["clean_content_hash"] = content_hash(
            self.record
        )

    def _apply_medical_details(self) -> None:
        """Ajoute les signaux médicaux."""
        list_fields = (
            "triage_categories",
            "risk_categories",
            "sensitive_health_categories"
        )

        for field in list_fields:
            value = self.medical.get(field)
            self.record[field] = (
                list(value)
                if isinstance(value, list)
                else []
            )

        self.record.update({
            "medical_marker_count": int(
                self.medical.get("medical_marker_count") or 0
            ),
            "medical_status": self.medical.get(
                "medical_status",
                "NO_MARKER_DETECTED"
            ),
            "triage_marker_count": int(
                self.medical.get("triage_marker_count") or 0
            ),
            "triage_status": self.medical.get(
                "triage_status",
                "NO_TRIAGE_MARKER_DETECTED"
            ),
            "risk_flag": bool(
                self.medical.get("risk_flag")
            )
        })

        if self.record["risk_categories"]:
            self.flags.add("HIGH_RISK")

    def _detect_language(self) -> None:
        """Détecte la langue."""
        record_input = clean_text(
            self.record.get("input")
        )

        sample = (
            record_input
            if len(record_input) >= 80
            else self.privacy.all_text()
        )

        try:
            detected = (
                detect(sample)
                if len(sample) >= 20
                else "unknown"
            )
        except LangDetectException:
            detected = "unknown"

        expected = self.record.get("language")
        self.record["detected_language"] = detected

        if (
            detected != "unknown"
            and expected
            and detected != expected
        ):
            self.flags.add("LANGUAGE_MISMATCH")


def collect_remaining_automatic_pii(
    records: list[dict]
) -> list[dict]:
    """Retourne les PII automatiques restantes."""
    remaining_records = []

    for record in records:
        remaining = PrivacyProcessor(
            record
        ).remaining_automatic_pii()

        if not remaining:
            continue

        remaining_records.append({
            "id": record.get("id"),
            "dataset": record.get("dataset"),
            "family": record.get("family"),
            "language": record.get("language"),
            "pii": remaining
        })

    return remaining_records
