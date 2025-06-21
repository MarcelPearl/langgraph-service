from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging
from app.utils.lifespan import lifespan

from app.api.v1 import health, ai, execution, system, test

configure_logging()

app = FastAPI(
    title="FastAPI AI Engine - Workflow Node Processor",
    description="AI/ML node execution service for the dual-backend workflow system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc) if settings.DEBUG else "An error occurred"}
    )

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(ai.router, prefix="/api/v1", tags=["ai"])
app.include_router(execution.router, prefix="/api/v1", tags=["execution"])
app.include_router(system.router, prefix="/api/v1", tags=["checks"])
app.include_router(test.router, prefix="/api/v1", tags=["test"])

@app.on_event("startup")
async def startup_event():
    from app.core.config import settings
    import logging
    logger = logging.getLogger(__name__)
    logger.info("🚀 Configuration:")
    logger.info(f"- Debug: {settings.DEBUG}")
    logger.info(f"- Redis: {settings.REDIS_URL}")
    logger.info(f"- Kafka: {settings.KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"- Env: {settings.ENVIRONMENT}")
    logger.info(f"- Providers: {[k for k in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HUGGINGFACE_API_KEY'] if getattr(settings, k, None)]}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
