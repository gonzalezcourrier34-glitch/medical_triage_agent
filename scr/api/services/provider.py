"""Sélection du moteur d'inférence."""

import os

BACKEND = os.getenv("INFERENCE_BACKEND", "mock")

if BACKEND == "mock":
    from scr.api.services.inference_mock import inference_service

elif BACKEND == "local":
    from scr.api.services.inference import inference_service

else:
    raise ValueError(f"Moteur d'inférence inconnu : {BACKEND}")