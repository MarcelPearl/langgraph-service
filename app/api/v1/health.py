# app/api/v1/health.py
from fastapi import APIRouter, HTTPException
from app.services.context.redis_service import redis_context_service
from app.services.messaging.kafka_service import kafka_service
import time

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "fastapi-ai-engine",
        "version": "1.0.0"
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check for required services"""
    try:
        services_status = {}
        overall_status = "ready"

        # Check Redis connectivity
        try:
            redis_health = await redis_context_service.health_check()
            services_status["redis"] = redis_health["status"]
        except Exception as e:
            services_status["redis"] = "disconnected"
            overall_status = "not_ready"

        # Check Kafka connectivity  
        try:
            kafka_health = await kafka_service.health_check()
            services_status["kafka"] = kafka_health["status"]
        except Exception as e:
            services_status["kafka"] = "disconnected"
            overall_status = "not_ready"

        return {
            "status": overall_status,
            "services": services_status,
            "timestamp": time.time(),
            "service": "fastapi-ai-engine"
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": time.time()
            }
        )