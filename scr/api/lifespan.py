from contextlib import asynccontextmanager

from fastapi import FastAPI

from scr.api.services.inference import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'API."""
    inference_service.load()
    yield