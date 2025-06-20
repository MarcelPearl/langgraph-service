from datetime import time

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.database import init_database, close_database
from app.core.config import settings
from app.api.v1 import workflows, health, ai

# Kafka integration
from app.services.kafka.producer import event_producer
from app.schemas.events import EventType, create_execution_event

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with Kafka integration"""
    logger.info("Application startup initiated")

    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized")

        # Start Kafka producer
        await event_producer.start()
        logger.info("Kafka producer started")

        # Publish application startup event
        try:
            startup_event = create_execution_event(
                event_type=EventType.EXECUTION_STARTED,
                execution_id="system-startup",
                workflow_id="system",
                execution_status="running",
                metadata={
                    "event": "application_startup",
                    "version": "1.0.0",
                    "environment": getattr(settings, 'ENVIRONMENT', 'development')
                }
            )
            await event_producer.publish_event(startup_event)
            logger.info("Application startup event published")
        except Exception as e:
            logger.warning(f"Failed to publish startup event: {e}")

        logger.info("Application startup complete")

        yield

        # Shutdown phase
        logger.info("Application shutdown initiated")

        # Publish application shutdown event
        try:
            shutdown_event = create_execution_event(
                event_type=EventType.EXECUTION_COMPLETED,
                execution_id="system-shutdown",
                workflow_id="system",
                execution_status="completed",
                metadata={
                    "event": "application_shutdown",
                    "version": "1.0.0"
                }
            )
            await event_producer.publish_event(shutdown_event)
            logger.info("Application shutdown event published")
        except Exception as e:
            logger.warning(f"Failed to publish shutdown event: {e}")

        # Stop Kafka producer
        await event_producer.stop()
        logger.info("Kafka producer stopped")

        # Close database connections
        await close_database()
        logger.info("Database connections closed")

        logger.info("Application shutdown complete")

    except Exception as e:
        logger.error(f"Error during application lifecycle: {e}")
        raise


app = FastAPI(
    title="MarcelPearl Workflow Automation Platform",
    description="""
    Advanced AI Workflow Automation Platform with Event-Driven Architecture

    Features:
    - AI-powered workflow execution using LangGraph
    - Multi-provider AI model support (OpenAI, Anthropic, HuggingFace)
    - Event-driven architecture with Kafka
    - Real-time workflow monitoring and analytics
    - Tool integration and custom AI agents
    - Scalable execution engine with state persistence
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)


# Add middleware for request/response logging and metrics
@app.middleware("http")
async def log_requests(request, call_next):
    """Log requests and publish metrics events"""
    import time
    import uuid
    from datetime import datetime

    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Add request ID to state
    request.state.request_id = request_id

    logger.info(
        f"Request started: {request.method} {request.url.path} "
        f"[{request_id}]"
    )

    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    # Log response
    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"[{request_id}] {response.status_code} {process_time:.3f}s"
    )

    # Publish API metrics event (async, don't wait)
    try:
        if hasattr(event_producer, 'is_started') and event_producer.is_started:
            from app.schemas.events import BaseEventPayload, EventType

            metrics_event = BaseEventPayload(
                event_type=EventType.EXECUTION_COMPLETED,  # Reusing for API metrics
                metadata={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time_ms": int(process_time * 1000),
                    "event_category": "api_metrics"
                }
            )

            # Fire and forget
            import asyncio
            asyncio.create_task(event_producer.publish_event(metrics_event))

    except Exception as e:
        logger.warning(f"Failed to publish API metrics event: {e}")

    return response


# Health check enhancement with Kafka status
@app.get("/api/v1/health/detailed")
async def detailed_health_check():
    """Detailed health check including Kafka connectivity"""
    from app.core.database import get_db
    from sqlalchemy import text

    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "workflow-automation",
        "version": "1.0.0",
        "checks": {}
    }

    # Database check
    try:
        async with get_db().__anext__() as db:
            await db.execute(text("SELECT 1"))
            health_data["checks"]["database"] = {"status": "healthy"}
    except Exception as e:
        health_data["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_data["status"] = "unhealthy"

    # Kafka producer check
    try:
        if event_producer.is_started:
            health_data["checks"]["kafka_producer"] = {"status": "healthy", "connected": True}
        else:
            health_data["checks"]["kafka_producer"] = {"status": "unhealthy", "connected": False}
            health_data["status"] = "degraded"
    except Exception as e:
        health_data["checks"]["kafka_producer"] = {"status": "unhealthy", "error": str(e)}
        health_data["status"] = "unhealthy"

    # AI services check
    try:
        from app.services.ai.model_factory import model_factory
        providers = model_factory.get_available_providers()
        health_data["checks"]["ai_providers"] = {
            "status": "healthy" if providers else "degraded",
            "available_providers": providers,
            "count": len(providers)
        }
    except Exception as e:
        health_data["checks"]["ai_providers"] = {"status": "unhealthy", "error": str(e)}

    return health_data


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["workflows"])
app.include_router(ai.router, prefix="/api/v1", tags=["ai"])

# Kafka monitoring routes
from app.api.v1 import kafka_monitoring

app.include_router(kafka_monitoring.router, prefix="/api/v1", tags=["kafka-monitoring"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "MarcelPearl Workflow Automation Platform",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "AI-powered workflows",
            "Event-driven architecture",
            "Multi-provider AI support",
            "Real-time monitoring",
            "Tool integration"
        ],
        "documentation": "/api/docs",
        "health_check": "/api/v1/health"
    }


# Global exception handler with event publishing
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with error event publishing"""
    import traceback
    from fastapi.responses import JSONResponse

    error_id = str(uuid.uuid4())

    logger.error(
        f"Unhandled exception [{error_id}]: {exc}\n"
        f"Traceback: {traceback.format_exc()}"
    )

    # Publish error event
    try:
        if hasattr(event_producer, 'is_started') and event_producer.is_started:
            from app.schemas.events import BaseEventPayload, EventType, EventSeverity

            error_event = BaseEventPayload(
                event_type=EventType.EXECUTION_FAILED,
                severity=EventSeverity.ERROR,
                metadata={
                    "error_id": error_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "path": request.url.path,
                    "method": request.method,
                    "event_category": "api_error"
                }
            )

            import asyncio
            asyncio.create_task(event_producer.publish_event(error_event))

    except Exception as publish_error:
        logger.warning(f"Failed to publish error event: {publish_error}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "error_id": error_id,
            "message": "An unexpected error occurred. Please contact support.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )