from pathlib import Path
from typing import Literal

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scr.api.schemas import TriageRequest


Priority = Literal["P1", "P2", "P3"]

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
MODEL_PATH = Path("artifacts/model/qwen3_1_7b_triage")

CANDIDATES = {
    "P1": "PRIORITY: P1",
    "P2": "PRIORITY: P2",
    "P3": "PRIORITY: P3"
}

INPUT_LABELS = {
    "fr": ("Contexte", "Question", "Information médicale"),
    "en": ("Context", "Question", "Medical information")
}

SYSTEM_PROMPTS = {
    "fr": (
        "Tu es un assistant de triage médical destiné à un service d'urgence. "
        "Évalue la priorité à partir des informations médicales fournies. "
        "P1 = urgence élevée nécessitant une prise en charge rapide, "
        "P2 = urgence modérée, P3 = situation différable. "
        "Ne suppose aucune information absente. "
        "Réponds exactement avec : PRIORITY: P1, PRIORITY: P2 ou PRIORITY: P3."
    ),
    "en": (
        "You are a medical triage assistant for an emergency department. "
        "Assess priority from the medical information provided. "
        "P1 = high urgency requiring prompt care, "
        "P2 = moderate urgency, P3 = deferrable situation. "
        "Do not assume missing information. "
        "Answer exactly with: PRIORITY: P1, PRIORITY: P2 or PRIORITY: P3."
    )
}


class InferenceService:
    """Gère l'inférence du modèle de triage."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_version = "not_loaded"

    @property
    def is_ready(self) -> bool:
        """Indique si le modèle est disponible."""
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Charge Qwen3 puis l'adapter final."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype="auto",
            device_map="auto",
            attn_implementation="sdpa"
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            MODEL_PATH,
            is_trainable=False
        )

        self.model.config.use_cache = True
        self.model.eval()
        self.model_version = MODEL_PATH.name

    def clinical_input(self, request: TriageRequest) -> str:
        """Construit exactement l'entrée clinique du benchmark."""
        labels = INPUT_LABELS[request.language]
        values = (
            request.context,
            request.question,
            request.answer
        )

        return "\n\n".join(
            f"{label}: {value}"
            for label, value in zip(labels, values)
            if str(value).strip()
        )

    def build_prompt(self, request: TriageRequest) -> str:
        """Construit exactement le prompt du benchmark."""
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPTS[request.language]
            },
            {
                "role": "user",
                "content": self.clinical_input(request)
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    @torch.inference_mode()
    def score_candidate(self, prompt: str, candidate: str) -> float:
        """Score un candidat comme pendant le benchmark."""
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False
        )["input_ids"]

        target = self.tokenizer(
            candidate,
            add_special_tokens=False
        )["input_ids"]

        max_prompt = 1024 - len(target)
        ids = prompt_ids[-max_prompt:] + target

        input_ids = torch.tensor(
            [ids],
            device=self.model.device
        )

        logits = self.model(input_ids=input_ids).logits

        start = len(ids) - len(target) - 1
        logits = logits[0, start:start + len(target)].float()
        log_probs = torch.log_softmax(logits, dim=-1)

        target_ids = torch.tensor(
            target,
            device=self.model.device
        )

        index = torch.arange(
            len(target),
            device=self.model.device
        )

        return log_probs[index, target_ids].mean().item()

    async def predict(self, request: TriageRequest) -> Priority:
        """Retourne la priorité ayant le meilleur score."""
        if not self.is_ready:
            raise RuntimeError("Modèle non chargé.")

        prompt = self.build_prompt(request)

        scores = {
            priority: self.score_candidate(prompt, candidate)
            for priority, candidate in CANDIDATES.items()
        }

        return max(scores, key=scores.get)


inference_service = InferenceService()