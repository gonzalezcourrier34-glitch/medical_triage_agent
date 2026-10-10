"""Point d'entrée de l'API Medical Triage Agent."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator

from scr.api.lifespan import lifespan
from scr.api.routes.health import router as health_router
from scr.api.routes.triage import router as triage_router
from scr.api.routes.version import router as version_router


# Création de l'application FastAPI.
app = FastAPI(
    title="Medical Triage API",
    description="POC d'assistance au triage médical par IA.",
    version="0.1.0",
    lifespan=lifespan,
)


# Instrumentation automatique des requêtes HTTP.
Instrumentator(
    excluded_handlers=["/metrics"],
).instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)


# Redirection vers Swagger.
@app.get("/", include_in_schema=False)
async def root():
    """Redirige vers la documentation de l'API."""
    return RedirectResponse(url="/docs")


# Enregistrement des routes.
app.include_router(health_router)
app.include_router(version_router)
app.include_router(triage_router)