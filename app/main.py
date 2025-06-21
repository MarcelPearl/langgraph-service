# app/main.py
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.v1 import health, ai

# Import new services
from app.services.messaging.kafka_service import kafka_service
from app.services.context.redis_service import redis_context_service
from app.services.execution.node_engine import node_execution_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log') if settings.DEBUG else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global background task for Kafka consumption
kafka_consumer_task: asyncio.Task = None


async def start_kafka_consumer():
    """Start Kafka message consumption"""
    global kafka_consumer_task
    try:
        kafka_consumer_task = asyncio.create_task(kafka_service.start_consuming())
        logger.info("Kafka consumer started")
    except Exception as e:
        logger.error(f"Failed to start Kafka consumer: {e}")


async def stop_kafka_consumer():
    """Stop Kafka message consumption"""
    global kafka_consumer_task
    if kafka_consumer_task and not kafka_consumer_task.done():
        kafka_consumer_task.cancel()
        try:
            await kafka_consumer_task
        except asyncio.CancelledError:
            pass
        logger.info("Kafka consumer stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting FastAPI AI Engine...")

    try:
        # Connect to Redis
        await redis_context_service.connect()
        logger.info("✅ Redis connected")

        # Start Kafka service
        await kafka_service.start()
        logger.info("✅ Kafka service started")

        # Start node execution engine
        await node_execution_engine.start()
        logger.info("✅ Node execution engine started")

        # Start Kafka consumer
        await start_kafka_consumer()
        logger.info("✅ Kafka consumer started")

        logger.info("🚀 FastAPI AI Engine startup complete")

        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    finally:
        # Cleanup on shutdown
        logger.info("Shutting down FastAPI AI Engine...")

        try:
            # Stop Kafka consumer
            await stop_kafka_consumer()

            # Stop services
            await node_execution_engine.stop()
            await kafka_service.stop()
            await redis_context_service.disconnect()

            logger.info("✅ Application shutdown complete")

        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="FastAPI AI Engine - Workflow Node Processor",
    description="AI/ML node execution service for the dual-backend workflow system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(ai.router, prefix="/api/v1", tags=["ai"])

# Import and include execution router
from app.api.v1.execution import router as execution_router

app.include_router(execution_router, prefix="/api/v1", tags=["execution"])


# New service status endpoints
@app.get("/api/v1/system/status")
async def get_system_status():
    """Get overall system status"""
    try:
        # Check all services
        kafka_health = await kafka_service.health_check()
        redis_health = await redis_context_service.health_check()

        overall_status = "healthy"
        if any(service["status"] != "healthy" for service in [kafka_health, redis_health]):
            overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "services": {
                "kafka": kafka_health,
                "redis": redis_health
            },
            "service_type": "fastapi_ai_engine",
            "version": "1.0.0",
            "node_types_supported": [
                "ai_decision",
                "ai_text_generator",
                "ai_data_processor"
            ]
        }

    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/api/v1/system/cleanup")
async def cleanup_old_executions(older_than_hours: int = 48):
    """Clean up old execution data"""
    try:
        await redis_context_service.cleanup_expired_executions(older_than_hours)
        return {"message": f"Cleanup completed for executions older than {older_than_hours} hours"}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Development and testing endpoints
if settings.DEBUG:
    @app.post("/api/v1/test/kafka")
    async def test_kafka():
        """Test Kafka connectivity"""
        try:
            from app.services.messaging.kafka_service import NodeExecutionMessage
            import uuid

            test_message = NodeExecutionMessage(
                execution_id=str(uuid.uuid4()),
                workflow_id=str(uuid.uuid4()),
                node_id="test_node",
                node_type="ai_decision",
                node_data={"prompt": "Test prompt", "options": ["yes", "no"]},
                context={"test": True},
                dependencies=[],
                timestamp=str(asyncio.get_event_loop().time())
            )

            await kafka_service.publish_node_execution(test_message)
            return {"message": "Kafka test message sent successfully"}

        except Exception as e:
            logger.error(f"Kafka test failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post("/api/v1/test/redis")
    async def test_redis():
        """Test Redis connectivity"""
        try:
            test_data = {"test": True, "timestamp": str(asyncio.get_event_loop().time())}
            await redis_context_service.set_execution_context("test_execution", test_data, ttl=60)

            retrieved = await redis_context_service.get_execution_context("test_execution")

            return {
                "message": "Redis test successful",
                "stored": test_data,
                "retrieved": retrieved
            }

        except Exception as e:
            logger.error(f"Redis test failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Add startup event
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info(f"FastAPI AI Engine starting with configuration:")
    logger.info(f"- Debug mode: {settings.DEBUG}")
    logger.info(f"- Redis URL: {settings.REDIS_URL}")
    logger.info(f"- Kafka servers: {settings.KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(
        f"- AI providers configured: {len([p for p in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HUGGINGFACE_API_KEY'] if getattr(settings, p, None)])}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )