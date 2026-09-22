import ast
import hashlib
import math
import re
import unicodedata

import pandas as pd

from scr.config import (
    ContextPatterns,
    PrivacyConfig,
    QualityConfig,
    SourceSchema
)

from scr.utils.hashing import digest

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
        for line in value.split("\n")
    ).strip()


def key_text(value) -> str:
    """Normalise un texte pour comparaison."""
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def audit_id(dataset, index, source_id) -> str:
    """Génère l'identifiant historique d'audit."""
    def old_normalize(value) -> str:
        value = unicodedata.normalize("NFKD", str(value).strip().lower())
        value = "".join(char for char in value if not unicodedata.combining(char))
        return re.sub(r"\s+", " ", value).strip()

    payload = "\u241f".join(old_normalize(value) for value in (dataset, index, source_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


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
        valid_pair = (
            len(value) == 2
            and all(isinstance(message, dict) for message in value)
            and value[0].get("role") == "user"
            and key_text(value[0].get("content")) == key_text(prompt)
            and value[1].get("role") == "assistant"
        )
        if valid_pair:
            return clean_text(value[1].get("content"))
    raise ValueError("Structure DPO non prise en charge.")


def assemble_input(record: dict) -> str:
    """Construit l'entrée canonique."""
    if record.get("family") == "dpo":
        return clean_text(record.get("question"))
    header = "\n\n".join(
        value
        for value in (clean_text(record.get("context")), clean_text(record.get("question")))
        if value
    )
    choices = "\n".join(
        f"{label}. {cleaned}"
        for label, value in sorted((record.get("choices") or {}).items())
        if (cleaned := clean_text(value))
    )
    return f"{header}\n{choices}" if header and choices else header or choices


SOURCE_CONTENT_HASH_FIELDS = (
    "family", "task", "context", "question", "choices",
    "labels", "answer", "chosen", "rejected", "feedback"
)


def content_hash(record: dict) -> str:
    """Calcule l'empreinte du contenu canonique."""
    return digest({field: record.get(field) for field in SOURCE_CONTENT_HASH_FIELDS})


from scr.data.privacy import PrivacyProcessor


class QualityValidator:
    """Valide structure, texte et confidentialité d'un record."""

    # Initialise le validateur sur le record canonique courant.
    # Le processeur de confidentialité partage volontairement le même objet.
    def __init__(self, record: dict) -> None:
        """Initialise le validateur."""
        self.record = record
        self.privacy = PrivacyProcessor(record)

    # Exécute les contrôles propres à la famille puis les contrôles communs.
    # Les anomalies sont ajoutées aux flags sans modifier les cibles source.
    def validate(self) -> None:
        """Exécute les contrôles qualité."""
        self.record['input'] = assemble_input(self.record)

        if not self.record.get('question'):
            self._flag('EMPTY_QUESTION')

        family = self.record['family']

        if family in SourceSchema.QCM_FAMILIES:
            self._validate_qcm()
        elif family == 'dpo':
            self._validate_dpo()
        else:
            self._validate_open_answer()

        self._validate_privacy()
        self._validate_text_quality()

    # Ajoute un flag qualité au record courant.
    # Cette méthode centralise les écritures dans l'ensemble des validations.
    def _flag(self, name: str) -> None:
        """Ajoute un flag."""
        flags = set(
            self.record.get("flags") or []
        )

        flags.add(name)
        self.record["flags"] = flags

    # Contrôle la cohérence des choix, labels et modalités des QCM.
    # Vérifie aussi les références explicites à un contexte absent.
    def _validate_qcm(self) -> None:
        """Valide un QCM."""
        family = self.record['family']
        choices = self.record['choices']
        labels = self.record['labels']
        allowed_counts = {4, 5} if family == 'mcqu' else {5}

        if (
            len(choices) not in allowed_counts
            or not labels
            or not set(labels) <= set(choices)
        ):
            self._flag('INVALID_QCM')

        invalid_mode = (
            family == 'mcqu' and len(labels) != 1
            or family == 'mcqm' and len(labels) < 2
        )

        if invalid_mode:
            self._flag('INVALID_QCM_MODE')

        normalized_choices = {
            key_text(choice)
            for choice in choices.values()
        }

        if len(normalized_choices) != len(choices):
            self._flag('DUPLICATE_OPTIONS')

        if (
            not self.record.get('context')
            and ContextPatterns.CONTEXT_REF.search(
                self.record.get('question', '')
            )
        ):
            self._flag('NEEDS_CONTEXT')

    # Contrôle présence, distinction et éventuel biais de longueur de la paire.
    # L'orientation chosen/rejected reste strictement inchangée.
    def _validate_dpo(self) -> None:
        """Valide une paire DPO."""
        chosen = self.record.get('chosen', '')
        rejected = self.record.get('rejected', '')

        if not chosen or not rejected:
            self._flag('EMPTY_DPO_RESPONSE')
            return

        if key_text(chosen) == key_text(rejected):
            self._flag('IDENTICAL_DPO_RESPONSES')

        shorter = max(
            1,
            min(len(chosen), len(rejected))
        )
        ratio = max(
            len(chosen),
            len(rejected)
        ) / shorter

        self.record['length_ratio'] = round(
            ratio,
            4
        )

        label_type = clean_text(
            self.record.get('label_type')
        ).casefold()

        if ratio > 2 or label_type == 'length':
            self._flag('DPO_LENGTH_BIAS')

        metadata = self.record.get('metadata')

        if not isinstance(metadata, dict):
            metadata = {}

        if not metadata.get('golden_answer'):
            self.record['notes'].add(
                'NO_GOLDEN_ANSWER'
            )

    # Contrôle uniquement les défauts simples des réponses ouvertes.
    # Aucun jugement médical ou sémantique complexe n'est effectué ici.
    def _validate_open_answer(self) -> None:
        """Valide une réponse ouverte."""
        answer = clean_text(
            self.record.get('answer')
        )

        if not answer:
            self._flag('EMPTY_ANSWER')
            return

        normalized = key_text(
            answer
        ).casefold()

        if normalized in {
            'topic',
            'topics',
            'sujet',
            'sujets',
            'resource',
            'resources',
            'ressource',
            'ressources'
        }:
            self._flag('UNINFORMATIVE_ANSWER')

        if answer.casefold().startswith((
            'these resources address',
            'these resources from medlineplus',
            'ces ressources concernent',
            'les ressources suivantes'
        )):
            self._flag('RESOURCE_ONLY_ANSWER')

        if answer.endswith('?') and len(answer) < 120:
            self._flag('QUESTION_AS_ANSWER')

    # Vérifie les PII encore présentes après les anonymisations automatiques.
    # Distingue les candidats à revue des échecs de suppression automatique.
    def _validate_privacy(self) -> None:
        """Valide les résultats Presidio."""
        matches = self.privacy.pii_matches()

        categories = sorted({
            match["category"]
            for match in matches
        })

        automatic_remaining = [
            match
            for match in matches
            if (
                match["category"]
                in PrivacyConfig.AUTOMATIC_PII_CATEGORIES
            )
        ]

        review_categories = (
            set(categories)
            & PrivacyConfig.REVIEW_PII_CATEGORIES
        )

        self.record["pii_matches"] = matches
        self.record["pii_categories"] = categories

        if review_categories:
            self._flag("PII_CANDIDATE")
            self._flag("PII_HUMAN_REVIEW")

        if automatic_remaining:
            self._flag("PII_REDACTION_FAILED")

    # Recherche anomalies textuelles, tokens de chat et références externes.
    # Ces contrôles restent déterministes et indépendants du contenu médical.
    def _validate_text_quality(self) -> None:
        """Valide la qualité textuelle."""
        content = self.privacy.all_text()

        if ContextPatterns.ANOMALY.search(content):
            self._flag('TEXT_ANOMALY')

        chat_token_pattern = (
            r'<\|(?:im_start|im_end|endoftext|eot_id|'
            r'start_header_id|end_header_id)\|>|'
            r'\[/?INST\]'
        )

        if re.search(chat_token_pattern, content):
            self._flag('CHAT_CONTROL_TOKEN')

        if ContextPatterns.EXTERNAL_REF.search(
            self.record.get('input', '')
        ):
            self._flag('EXTERNAL_REFERENCE')


def unresolved(record: dict) -> set[str]:
    """Retourne les flags non résolus."""
    return (
        record.get('flags', set())
        - QualityConfig.INFORMATIVE_FLAGS
        - record.get('manual_allow', set())
    )
