from fastapi import FastAPI
from app.api.routes_health import router as health_router
from app.api.routes_predict import router as predict_router
from app.core.logger import setup_logging


setup_logging()
app = FastAPI(title="Risk LLM API", version="1.0.0")


@app.on_event("startup")
async def on_startup():
    pass


app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(predict_router, prefix="/api/risk", tags=["risk"])