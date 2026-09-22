# scr/triage/extraction_general.py

from copy import deepcopy

from scr.triage.extract_patient import PatientExtractor
from scr.triage.medical_knowledge import MedicalKnowledgeExtractor
from scr.triage.recovery import TriageRecoveryMatcher


# ## Extracteur général

class GeneralMedicalExtractor:
    """Coordonne patient, knowledge et recovery."""

    def __init__(self, motif_lexicon: dict) -> None:
        self.patient_extractor = PatientExtractor(motif_lexicon)
        self.knowledge_extractor = MedicalKnowledgeExtractor(
            motif_lexicon
        )
        self.recovery_matcher = TriageRecoveryMatcher(
            motif_lexicon
        )

    # ## Extraction

    # Extrait patient et knowledge indépendamment.
    # Le recovery complète uniquement les motifs FRENCH absents.
    def extract(self, record: dict) -> dict:
        """Enrichit un record médical."""
        enriched = deepcopy(record)

        patient = self._recover_patient(
            record,
            self.patient_extractor.extract(record)
        )

        knowledge = self._recover_knowledge(
            record,
            self.knowledge_extractor.extract(record)
        )

        enriched["patient"] = patient
        enriched["medical_knowledge"] = knowledge
        enriched["medical_extraction"] = self._metadata(
            record,
            patient,
            knowledge
        )

        return enriched

    # ## Recovery patient

    def _recover_patient(
        self,
        record: dict,
        patient: dict
    ) -> dict:
        """Complète les motifs patient absents."""
        if patient.get("french_motif_candidates"):
            return patient

        recovery = self.recovery_matcher.match_patient(
            record,
            patient
        )

        patient["french_motif_recovery"] = recovery

        if recovery.get("status") == "unique_match":
            patient["french_motif_candidates"] = (
                recovery.get("candidates") or []
            )
            patient["has_french_motif"] = True

        return patient

    # ## Recovery knowledge

    def _recover_knowledge(
        self,
        record: dict,
        knowledge: dict
    ) -> dict:
        """Complète les motifs knowledge absents."""
        if knowledge.get("french_motif_candidates"):
            return knowledge

        recovery = self.recovery_matcher.match_knowledge(
            record,
            knowledge
        )

        knowledge["french_motif_recovery"] = recovery

        if recovery.get("status") == "unique_match":
            knowledge["french_motif_candidates"] = (
                recovery.get("candidates") or []
            )
            knowledge["has_french_motif"] = True

        return knowledge

    # ## Métadonnées

    def _metadata(
        self,
        record: dict,
        patient: dict,
        knowledge: dict
    ) -> dict:
        """Construit les métadonnées globales."""
        patient_fields = (
            patient.get("extraction", {}).get("extracted_fields")
            or []
        )

        knowledge_fields = (
            knowledge.get("extracted_fields")
            or []
        )

        patient_recovery = (
            patient.get("french_motif_recovery")
            or {}
        )

        knowledge_recovery = (
            knowledge.get("french_motif_recovery")
            or {}
        )

        return {
            "method": "deterministic_general_medical_extraction_v3",
            "has_patient_information": bool(patient_fields),
            "has_medical_knowledge": bool(knowledge_fields),
            "patient_fields": patient_fields,
            "medical_knowledge_fields": knowledge_fields,
            "patient_french_motifs": self._rule_ids(patient),
            "knowledge_french_motifs": self._rule_ids(knowledge),
            "patient_recovery": self._recovery_summary(
                patient_recovery
            ),
            "knowledge_recovery": self._recovery_summary(
                knowledge_recovery
            ),
            "source_clean_content_hash": record.get(
                "clean_content_hash"
            )
        }

    # ## Recovery résumé

    def _recovery_summary(
        self,
        recovery: dict
    ) -> dict:
        """Résume un recovery."""
        return {
            "used": bool(recovery.get("used")),
            "status": recovery.get("status"),
            "score": recovery.get("score")
        }

    # ## Utilitaire

    def _rule_ids(
        self,
        data: dict
    ) -> list[str]:
        """Retourne les rule_id détectés."""
        return [
            candidate["rule_id"]
            for candidate in (
                data.get("french_motif_candidates")
                or []
            )
            if candidate.get("rule_id")
        ]