"""Sélection du moteur d'inférence.

- mock  : réponse simulée, sans modèle (tests, CI, développement) ;
- local : Qwen3 + adapter LoRA chargés dans le processus (GPU local) ;
- vllm  : serveur vLLM distant compatible OpenAI (déploiement).
"""

import os

BACKEND = os.getenv("INFERENCE_BACKEND", "mock")

if BACKEND == "mock":
    from src.api.services.inference_mock import inference_service

elif BACKEND == "local":
    from src.api.services.inference import inference_service

elif BACKEND == "vllm":
    from src.api.services.inference_vllm import inference_service

else:
    raise ValueError(f"Moteur d'inférence inconnu : {BACKEND}")

__all__ = ["BACKEND", "inference_service"]
