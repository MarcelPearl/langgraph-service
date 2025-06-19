from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.database import init_database, close_database
from app.core.config import settings
from app.api.v1 import workflows, health, ai


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_database()
    print("Application startup complete")
    yield
    await close_database()
    print("Application shutdown complete")

app = FastAPI(
    title="MarcelPearl Workflow Automation Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["workflows"])
app.include_router(ai.router, prefix="/api/v1", tags=["ai"])