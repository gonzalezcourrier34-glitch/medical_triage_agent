"""Tests du prompt et de la lecture des réponses du modèle."""

import pytest

from src.api.schemas import TriageRequest
from src.api.services.prompting import (
    INPUT_LABELS,
    SYSTEM_PROMPTS,
    build_messages,
    clinical_input,
    parse_completion,
)


@pytest.mark.parametrize("language", ["fr", "en"])
def test_messages_use_training_prompt(language):
    """La conversation reprend le prompt système et le label d'entrée de l'entraînement."""
    request = TriageRequest(language=language, question="Douleur thoracique depuis 1 heure.")
    messages = build_messages(request)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPTS[language]
    assert messages[1]["content"].startswith(f"{INPUT_LABELS[language]}:\n")


def test_clinical_input_joins_non_empty_fields():
    """Le contexte, la question et la réponse non vides sont assemblés dans l'ordre."""
    request = TriageRequest(language="fr", context="Homme de 70 ans.", question="Dyspnée.")
    text = clinical_input(request)

    assert text == "Informations médicales à analyser:\nHomme de 70 ans.\n\nDyspnée."


def test_parse_completion():
    """La priorité et l'explication sont extraites d'une réponse bien formée."""
    result = parse_completion("PRIORITY: P1\nEXPLANATION: Détresse respiratoire.<|im_end|>")

    assert result == {"priority": "P1", "explanation": "Détresse respiratoire."}


def test_parse_completion_ignores_thinking_block():
    """Un bloc de réflexion éventuel n'est pas pris pour la réponse."""
    text = "<think>\nP3 ?\n</think>\nPRIORITY: P2\nEXPLANATION: Douleur aiguë."

    assert parse_completion(text)["priority"] == "P2"


def test_parse_completion_without_priority():
    """Une réponse sans priorité valide est refusée."""
    with pytest.raises(ValueError):
        parse_completion("EXPLANATION: texte sans priorité")
