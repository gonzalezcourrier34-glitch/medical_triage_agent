from fastapi import FastAPI

from scr.api.routes.health import router as health_router
from scr.api.routes.triage import router as triage_router
from scr.api.routes.version import router as version_router


app = FastAPI(
    title="Medical Triage API",
    description="POC d'assistance au triage médical par IA.",
    version="0.1.0"
)

app.include_router(health_router)
app.include_router(version_router)
app.include_router(triage_router)