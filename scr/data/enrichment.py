from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from scr.config import CONFIG
from scr.data.cleaning import QualityValidator, clean_text, content_hash, key_text
from scr.data.privacy import PrivacyProcessor
from scr.utils.hashing import digest


# Rend la détection de langue reproductible.
DetectorFactory.seed = CONFIG.seed


class RecordEnricher:
    """Nettoie et enrichit un record canonique."""

    def __init__(self, record: dict, medical: dict) -> None:
        self.record = record
        self.medical = medical
        self.flags = set(record.get("flags") or [])
        self.privacy = PrivacyProcessor(record)

    def process(self) -> None:
        """Applique langue, confidentialité, qualité et données médicales."""
        self._preserve_keys()
        self._detect_language()

        # Anonymise les PII avant de calculer le hash final.
        self.privacy.apply_automatic_redactions()
        QualityValidator(self.record).validate()

        self.record["clean_content_hash"] = content_hash(self.record)
        self._add_medical_details()
        self.record["flags"] = sorted(self.flags)

    def _preserve_keys(self) -> None:
        """Conserve les empreintes du contenu original."""
        self.record["original_input_key"] = digest(
            key_text(self.record.get("input"))
        )

        context = self.record.get("context")
        self.record["original_context_key"] = (
            digest(key_text(context)) if context else None
        )

    def _detect_language(self) -> None:
        """Détecte la langue et signale les incohérences."""
        text = clean_text(self.record.get("input"))

        if len(text) < 80:
            text = self.privacy.all_text()

        try:
            detected = detect(text) if len(text) >= 20 else "unknown"
        except LangDetectException:
            detected = "unknown"

        self.record["detected_language"] = detected

        if (
            detected != "unknown"
            and self.record.get("language")
            and detected != self.record["language"]
        ):
            self.flags.add("LANGUAGE_MISMATCH")

    def _add_medical_details(self) -> None:
        """Ajoute les informations médicales déjà calculées."""
        for field in (
            "triage_categories",
            "risk_categories",
            "sensitive_health_categories"
        ):
            value = self.medical.get(field)
            self.record[field] = value if isinstance(value, list) else []

        self.record.update({
            "medical_marker_count": int(
                self.medical.get("medical_marker_count") or 0
            ),
            "medical_status": (
                self.medical.get("medical_status") or "NO_MARKER_DETECTED"
            ),
            "triage_marker_count": int(
                self.medical.get("triage_marker_count") or 0
            ),
            "triage_status": (
                self.medical.get("triage_status")
                or "NO_TRIAGE_MARKER_DETECTED"
            ),
            "risk_flag": bool(self.medical.get("risk_flag"))
        })

        if self.record["risk_categories"]:
            self.flags.add("HIGH_RISK")


def collect_remaining_automatic_pii(records: list[dict]) -> list[dict]:
    """Retourne uniquement les records contenant encore une PII."""
    remaining = []

    for record in records:
        pii = PrivacyProcessor(record).remaining_automatic_pii()

        if pii:
            remaining.append({
                "id": record.get("id"),
                "dataset": record.get("dataset"),
                "family": record.get("family"),
                "language": record.get("language"),
                "pii": pii
            })

    return remaining