"""Moteur d'inférence local : Qwen3 fine-tuné, sur GPU (Transformers).

MODEL_PATH peut désigner soit le modèle fusionné (artifacts/deployment/...),
soit l'adapter LoRA final appliqué à la base Qwen3 (PEFT).
La priorité est choisie par scoring conditionnel des trois réponses possibles,
exactement comme lors de l'évaluation du modèle. L'explication est ensuite générée
à partir de la priorité retenue.
"""

import asyncio
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.api.schemas import TriageRequest
from src.api.services.prompting import (
    LABELS,
    MAX_LENGTH,
    MODEL_VERSION,
    build_messages,
    clean_explanation,
)

# Base réelle de l'entraînement : unsloth/Qwen3-1.7B-unsloth-bnb-4bit = Qwen/Qwen3-1.7B.
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-1.7B")
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "artifacts/deployment/qwen3-triage-20261009-154320-aligned")
)

# Fins de réponse apprises : <|im_end|> puis <|endoftext|>.
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))


class InferenceService:
    """Gère l'inférence locale du modèle de triage."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.stop_ids: list[int] = []
        self.model_version = "not_loaded"

    @property
    def is_ready(self) -> bool:
        """Indique si le modèle est disponible."""
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Charge le modèle fusionné, ou la base Qwen3 avec l'adapter final."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        is_adapter = (MODEL_PATH / "adapter_config.json").is_file()

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL if is_adapter else MODEL_PATH,
            dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
        )

        if is_adapter:
            model = PeftModel.from_pretrained(model, MODEL_PATH, is_trainable=False)

        self.model = model.eval()
        self.stop_ids = self.tokenizer.convert_tokens_to_ids(list(STOP_TOKENS))
        self.model_version = MODEL_VERSION

    def _tokenize(self, text: str) -> list[int]:
        """Tokenise un texte déjà formaté."""
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _prompt_ids(self, request: TriageRequest) -> list[int]:
        """Construit le prompt avec le template Qwen3, sans thinking."""
        kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        messages = build_messages(request)

        try:
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)

        return self._tokenize(prompt)

    @torch.inference_mode()
    def _score(self, prompt_ids: list[int], priority: int) -> float:
        """Log-probabilité moyenne de « PRIORITY: Pn » après le prompt."""
        target = self._tokenize(f"PRIORITY: P{priority}")
        prompt_ids = prompt_ids[-(MAX_LENGTH - len(target)) :]
        ids = prompt_ids + target

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        logits = self.model(input_ids=input_ids).logits
        logits = logits[0, len(prompt_ids) - 1 : len(ids) - 1].float()
        log_probs = torch.log_softmax(logits, dim=-1)

        target_ids = torch.tensor(target, dtype=torch.long, device=self.model.device)
        index = torch.arange(len(target), device=self.model.device)

        return log_probs[index, target_ids].mean().item()

    @torch.inference_mode()
    def _explain(self, prompt_ids: list[int], priority: int) -> str:
        """Génère l'explication associée à la priorité retenue."""
        prefix = self._tokenize(f"PRIORITY: P{priority}\nEXPLANATION:")
        ids = (prompt_ids + prefix)[-(MAX_LENGTH - MAX_NEW_TOKENS) :]

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Les jetons spéciaux sont conservés pour couper au premier marqueur de fin.
        generated = self.tokenizer.decode(output[0, len(ids) :], skip_special_tokens=False)
        return clean_explanation(generated)

    def _predict(self, request: TriageRequest) -> dict[str, str]:
        """Classe puis explique un cas."""
        prompt_ids = self._prompt_ids(request)
        scores = {priority: self._score(prompt_ids, priority) for priority in LABELS}
        priority = max(scores, key=scores.get)

        return {
            "priority": f"P{priority}",
            "explanation": self._explain(prompt_ids, priority),
        }

    async def predict(self, request: TriageRequest) -> dict[str, str]:
        """Retourne la priorité et l'explication sans bloquer la boucle asynchrone."""
        if not self.is_ready:
            raise RuntimeError("Modèle non chargé.")

        return await asyncio.to_thread(self._predict, request)


inference_service = InferenceService()
