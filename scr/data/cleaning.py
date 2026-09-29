import ast
import hashlib
import math
import re
import unicodedata

import pandas as pd

from scr.config import ContextPatterns, PrivacyConfig, QualityConfig, SourceSchema
from scr.data.privacy import PrivacyProcessor
from scr.utils.hashing import digest


# Normalisation
# Je transforme les champs source en texte propre sans modifier leur contenu sémantique.

def text(value) -> str:
    """Convertit une valeur scalaire en texte."""
    if value is None or value is pd.NA or isinstance(value, float) and math.isnan(value):
        return ""
    if not isinstance(value, (str, int, float)):
        raise ValueError("Champ texte non scalaire.")
    return str(value)


def clean_text(value) -> str:
    """Normalise un texte."""
    value = unicodedata.normalize("NFC", text(value))
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")

    return "\n".join(
        re.sub(r"[ \t]+", " ", line).strip()
        for line in value.splitlines()
    ).strip()


def key_text(value) -> str:
    """Normalise un texte pour comparaison."""
    return re.sub(r"\s+", " ", clean_text(value)).strip()


# Identifiants
# Je conserve ici l'algorithme historique afin de garder les mêmes IDs entre les notebooks.

def audit_id(dataset, index, source_id) -> str:
    """Génère l'identifiant historique d'audit."""

    def normalize(value) -> str:
        value = unicodedata.normalize("NFKD", str(value).strip().lower())
        value = "".join(c for c in value if not unicodedata.combining(c))
        return re.sub(r"\s+", " ", value).strip()

    payload = "\u241f".join(normalize(v) for v in (dataset, index, source_id))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


# QCM et DPO
# Ces fonctions convertissent les formats sources en représentations canoniques non ambiguës.

def parse_labels(value) -> list[str]:
    """Normalise une réponse QCM."""
    if isinstance(value, str) and value.strip().startswith(("[", "(", "{")):
        try:
            value = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError) as error:
            raise ValueError("Structure de réponse QCM non interprétable.") from error

    items = value if isinstance(value, (list, tuple, set)) else [value]
    labels = []

    for item in items:
        label = text(item).strip().upper()

        if not re.fullmatch(r"[A-E](?:[ ,;/|+]*[A-E])*", label):
            raise ValueError("Réponses QCM non interprétables sans ambiguïté.")

        labels.extend(re.findall(r"[A-E]", label))

    if not labels:
        raise ValueError("Réponse QCM absente.")
    if len(labels) != len(set(labels)):
        raise ValueError("Labels QCM répétés.")

    return sorted(labels)


def extract_completion(value, prompt) -> str:
    """Extrait une réponse assistant DPO."""
    if isinstance(value, str):
        return clean_text(value)

    if isinstance(value, dict):
        if value.get("role") != "assistant":
            raise ValueError("Message DPO unique non assistant.")
        return clean_text(value.get("content"))

    if isinstance(value, list):
        if len(value) == 1:
            return extract_completion(value[0], prompt)

        valid = (
            len(value) == 2
            and all(isinstance(message, dict) for message in value)
            and value[0].get("role") == "user"
            and key_text(value[0].get("content")) == key_text(prompt)
            and value[1].get("role") == "assistant"
        )

        if valid:
            return clean_text(value[1].get("content"))

    raise ValueError("Structure DPO non prise en charge.")


# Entrée canonique
# Je rassemble contexte, question et choix pour obtenir le texte réellement audité.

def assemble_input(record: dict) -> str:
    """Construit l'entrée canonique."""
    if record.get("family") == "dpo":
        return clean_text(record.get("question"))

    header = "\n\n".join(
        value
        for value in (
            clean_text(record.get("context")),
            clean_text(record.get("question"))
        )
        if value
    )

    choices = "\n".join(
        f"{label}. {cleaned}"
        for label, value in sorted((record.get("choices") or {}).items())
        if (cleaned := clean_text(value))
    )

    return "\n".join(value for value in (header, choices) if value)


HASH_FIELDS = (
    "family", "task", "context", "question", "choices",
    "labels", "answer", "chosen", "rejected", "feedback"
)


def content_hash(record: dict) -> str:
    """Calcule l'empreinte du contenu canonique."""
    return digest({field: record.get(field) for field in HASH_FIELDS})


# Validation qualité
# Je contrôle chaque famille puis applique les vérifications communes de confidentialité et de texte.

class QualityValidator:
    """Valide la qualité d'un record canonique."""

    def __init__(self, record: dict) -> None:
        self.record = record
        self.privacy = PrivacyProcessor(record)

    def validate(self) -> None:
        """Exécute les contrôles qualité."""
        self.record["input"] = assemble_input(self.record)

        if not self.record.get("question"):
            self._flag("EMPTY_QUESTION")

        family = self.record["family"]

        if family in SourceSchema.QCM_FAMILIES:
            self._validate_qcm()
        elif family == "dpo":
            self._validate_dpo()
        else:
            self._validate_open_answer()

        self._validate_privacy()
        self._validate_text_quality()

    def _flag(self, name: str) -> None:
        """Ajoute un flag."""
        flags = set(self.record.get("flags") or [])
        flags.add(name)
        self.record["flags"] = flags

    def _validate_qcm(self) -> None:
        """Valide un QCM."""
        family = self.record["family"]
        choices = self.record["choices"]
        labels = self.record["labels"]
        allowed = {4, 5} if family == "mcqu" else {5}

        if len(choices) not in allowed or not labels or not set(labels) <= set(choices):
            self._flag("INVALID_QCM")

        if (family == "mcqu" and len(labels) != 1) or (family == "mcqm" and len(labels) < 2):
            self._flag("INVALID_QCM_MODE")

        if len({key_text(choice) for choice in choices.values()}) != len(choices):
            self._flag("DUPLICATE_OPTIONS")

        if (
            not self.record.get("context")
            and ContextPatterns.CONTEXT_REF.search(self.record.get("question", ""))
        ):
            self._flag("NEEDS_CONTEXT")

    def _validate_dpo(self) -> None:
        """Valide une paire DPO."""
        chosen = self.record.get("chosen", "")
        rejected = self.record.get("rejected", "")

        if not chosen or not rejected:
            self._flag("EMPTY_DPO_RESPONSE")
            return

        if key_text(chosen) == key_text(rejected):
            self._flag("IDENTICAL_DPO_RESPONSES")

        ratio = max(len(chosen), len(rejected)) / max(1, min(len(chosen), len(rejected)))
        self.record["length_ratio"] = round(ratio, 4)

        if ratio > 2 or clean_text(self.record.get("label_type")).casefold() == "length":
            self._flag("DPO_LENGTH_BIAS")

        metadata = self.record.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}

        if not metadata.get("golden_answer"):
            self.record["notes"].add("NO_GOLDEN_ANSWER")

    def _validate_open_answer(self) -> None:
        """Valide une réponse ouverte."""
        answer = clean_text(self.record.get("answer"))

        if not answer:
            self._flag("EMPTY_ANSWER")
            return

        normalized = key_text(answer).casefold()

        if normalized in {
            "topic", "topics", "sujet", "sujets",
            "resource", "resources", "ressource", "ressources"
        }:
            self._flag("UNINFORMATIVE_ANSWER")

        if normalized.startswith((
            "these resources address",
            "these resources from medlineplus",
            "ces ressources concernent",
            "les ressources suivantes"
        )):
            self._flag("RESOURCE_ONLY_ANSWER")

        if answer.endswith("?") and len(answer) < 120:
            self._flag("QUESTION_AS_ANSWER")

    def _validate_privacy(self) -> None:
        """Vérifie les PII restantes."""
        matches = self.privacy.pii_matches()
        categories = sorted({match["category"] for match in matches})

        automatic = [
            match for match in matches
            if match["category"] in PrivacyConfig.AUTOMATIC_PII_CATEGORIES
        ]

        self.record["pii_matches"] = matches
        self.record["pii_categories"] = categories

        if set(categories) & PrivacyConfig.REVIEW_PII_CATEGORIES:
            self._flag("PII_CANDIDATE")
            self._flag("PII_HUMAN_REVIEW")

        if automatic:
            self._flag("PII_REDACTION_FAILED")

    def _validate_text_quality(self) -> None:
        """Détecte les anomalies textuelles."""
        content = self.privacy.all_text()

        if ContextPatterns.ANOMALY.search(content):
            self._flag("TEXT_ANOMALY")

        chat_tokens = (
            r"<\|(?:im_start|im_end|endoftext|eot_id|"
            r"start_header_id|end_header_id)\|>|\[/?INST\]"
        )

        if re.search(chat_tokens, content):
            self._flag("CHAT_CONTROL_TOKEN")

        if ContextPatterns.EXTERNAL_REF.search(self.record.get("input", "")):
            self._flag("EXTERNAL_REFERENCE")


def unresolved(record: dict) -> set[str]:
    """Retourne les flags non résolus."""
    return (
        set(record.get("flags") or [])
        - QualityConfig.INFORMATIVE_FLAGS
        - set(record.get("manual_allow") or [])
    )