"""Prompt, entrée clinique et lecture des réponses, identiques à l'entraînement SFT/DPO.

Ce module ne dépend ni de PyTorch ni du GPU : il est partagé par les moteurs
« local » (Transformers + PEFT) et « vllm », et peut être testé en CI.
"""

import os
import re

from src.api.schemas import TriageRequest

# Modèle final : SFT run 20261009-154320 (checkpoint-450) + adapter DPO (checkpoint-150).
MODEL_VERSION = os.getenv("MODEL_VERSION", "qwen3-1.7b-sft154320-ckpt450-dpo-ckpt150")

# Longueur maximale utilisée à l'entraînement.
MAX_LENGTH = 2_048

LABELS = (1, 2, 3)

# Prompts système copiés à l'identique du notebook 05 (entraînement du modèle final).
SYSTEM_PROMPTS = {
    "fr": (
        "Tu es un assistant de triage médical destiné à un service d'urgence. "
        "Ta mission est d'organiser la prise en charge des patients selon la gravité actuelle "
        "de leur situation clinique et leur risque de gravité ou d'aggravation. "
        "Classe obligatoirement chaque patient dans une seule priorité : "
        "P1 = urgence maximale : situation grave ou à haut risque de gravité nécessitant une prise "
        "en charge immédiate ou prioritaire, notamment en cas de menace vitale ou fonctionnelle, "
        "d'instabilité, de détresse majeure ou de risque important d'évolution grave ; "
        "P2 = urgence modérée : situation nécessitant une prise en charge médicale rapide "
        "en raison "
        "de sa gravité, de symptômes significatifs ou d'un risque clinique réel, sans nécessité "
        "de prise en charge aussi immédiate qu'une P1 ; "
        "P3 = urgence différée : patient présent aux urgences dont la situation est peu grave, "
        "stable et sans risque important de gravité ou d'aggravation identifié dans les données. "
        "La priorité doit tenir compte à la fois de l'état actuel du patient et du risque clinique "
        "associé à la situation décrite. "
        "Évalue uniquement les informations disponibles dans le cas fourni. "
        "Ne suppose aucune information absente et n'invente aucun symptôme, diagnostic "
        "ou élément patient. "
        "Fournis une explication claire et concise justifiant la priorité à partir des éléments "
        "disponibles. "
        "L'explication ne doit pas dépasser 200 mots. "
        "Réponds exactement sur deux lignes, "
        "en remplaçant <P1/P2/P3> par une seule priorité :\n"
        "PRIORITY: <P1/P2/P3>\n"
        "EXPLANATION: <explication claire justifiant la priorité>"
    ),
    "en": (
        "You are a medical triage assistant for an emergency department. "
        "Your task is to organize patient care according to both the current severity of the "
        "clinical situation and the risk of severity or deterioration. "
        "Classify each patient into exactly one priority: "
        "P1 = maximum urgency: severe or high-risk situation requiring immediate or priority care, "
        "including a threat to life or function, instability, major distress, or a significant "
        "risk of severe deterioration; "
        "P2 = moderate urgency: situation requiring prompt medical care because of its severity, "
        "significant symptoms, or a genuine clinical risk, without requiring care as immediately "
        "as P1; "
        "P3 = deferred urgency: patient presenting to the emergency department with a "
        "low-severity, stable situation and no significant risk of severity or deterioration "
        "identified in the data. "
        "The priority must account for both the patient's current condition and the clinical "
        "risk associated with the situation described. "
        "Assess only the information available in the provided case. "
        "Do not assume missing information or invent symptoms, diagnoses, or patient details. "
        "Provide a clear and concise explanation justifying the priority from the available "
        "information. "
        "The explanation must not exceed 200 words. "
        "Answer exactly in the following format:\n"
        "Answer in exactly two lines, replacing <P1/P2/P3> with a single priority:\n"
        "PRIORITY: <P1/P2/P3>\n"
        "EXPLANATION: <clear explanation justifying the priority>"
    ),
}

INPUT_LABELS = {
    "fr": "Informations médicales à analyser",
    "en": "Medical information to analyze",
}

PRIORITY_PATTERN = re.compile(r"PRIORITY:\s*P([123])\b")
EXPLANATION_PATTERN = re.compile(r"EXPLANATION:\s*(.*)", re.S)
STOP_MARKERS = ("<|im_end|>", "<|endoftext|>", "\nPRIORITY:")


def case_text(request: TriageRequest) -> str:
    """Assemble le cas clinique transmis par le client."""
    parts = (request.context, request.question, request.answer)
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def clinical_input(request: TriageRequest) -> str:
    """Construit le message utilisateur au même format qu'à l'entraînement."""
    return f"{INPUT_LABELS[request.language]}:\n{case_text(request)}"


def build_messages(request: TriageRequest) -> list[dict[str, str]]:
    """Construit la conversation système + utilisateur."""
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[request.language]},
        {"role": "user", "content": clinical_input(request)},
    ]


def clean_explanation(text: str) -> str:
    """Coupe l'explication générée au premier marqueur de fin."""
    for marker in STOP_MARKERS:
        text = text.split(marker, 1)[0]
    return " ".join(text.split())


def parse_completion(text: str) -> dict[str, str]:
    """Extrait la priorité et l'explication d'une réponse générée."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    priority = PRIORITY_PATTERN.search(text)

    if priority is None:
        raise ValueError("Priorité absente de la réponse du modèle.")

    explanation = EXPLANATION_PATTERN.search(text[priority.end() :])

    return {
        "priority": f"P{priority.group(1)}",
        "explanation": clean_explanation(explanation.group(1)) if explanation else "",
    }
