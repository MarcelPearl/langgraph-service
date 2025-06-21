
from fastapi import APIRouter, HTTPException

from app.services.context.redis_service import redis_context_service
from app.services.messaging.kafka_service import kafka_service

import time

router = APIRouter(prefix="/system", tags=["checks"])

@router.get("/status")
async def get_system_status():
    """Get overall system status with detailed service info"""
    try:
        services = {}
        overall_status = "healthy"
        try:
            kafka_health = await kafka_service.health_check()
            services["kafka"] = kafka_health
            if kafka_health["status"] != "healthy":
                overall_status = "degraded"
        except Exception as e:
            services["kafka"] = {
                "service": "kafka",
                "status": "unhealthy",
                "error": str(e)
            }
            overall_status = "degraded"

        try:
            redis_health = await redis_context_service.health_check()
            services["redis"] = redis_health
            if redis_health["status"] != "healthy":
                overall_status = "degraded"
        except Exception as e:
            services["redis"] = {
                "service": "redis",
                "status": "unhealthy",
                "error": str(e)
            }
            overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "services": services,
            "service_type": "fastapi_ai_engine",
            "version": "1.0.0",
            "node_types_supported": [
                "ai_decision",
                "ai_text_generator",
                "ai_data_processor"
            ],
            "kafka_running": kafka_service.running,
            "kafka_consumers": len(kafka_service.consumers)
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

@router.post("/cleanup")
async def cleanup_old_executions(older_than_hours: int = 48):
    """Clean up old execution data"""
    try:
        await redis_context_service.cleanup_expired_executions(older_than_hours)
        return {"message": f"Cleanup completed for executions older than {older_than_hours} hours"}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))