from fastapi import FastAPI
from fastapi.responses import RedirectResponse


from scr.api.lifespan import lifespan
from scr.api.routes.health import router as health_router
from scr.api.routes.triage import router as triage_router
from scr.api.routes.version import router as version_router


app = FastAPI(
    title="Medical Triage API",
    description="POC d'assistance au triage médical par IA.",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
async def root():
    """Redirige vers la documentation Swagger."""
    return RedirectResponse(url="/docs")

app.include_router(health_router)
app.include_router(version_router)
app.include_router(triage_router)