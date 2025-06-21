#!/usr/bin/env python3
"""
Startup script for the FastAPI AI Workflow Service
"""
import asyncio
import logging
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def check_dependencies():
    """Check if all required services are available"""
    logger.info("🔍 Checking service dependencies...")

    issues = []

    # Check Redis
    try:
        from app.services.context.redis_service import redis_context_service
        await redis_context_service.connect()
        health = await redis_context_service.health_check()
        if health["status"] == "healthy":
            logger.info("✅ Redis connection successful")
        else:
            issues.append("Redis health check failed")
        await redis_context_service.disconnect()
    except Exception as e:
        issues.append(f"Redis connection failed: {e}")

    # Check Kafka
    try:
        from app.services.messaging.kafka_service import kafka_service
        await kafka_service.start()
        health = await kafka_service.health_check()
        if health["status"] == "healthy":
            logger.info("✅ Kafka connection successful")
        else:
            issues.append("Kafka health check failed")
        await kafka_service.stop()
    except Exception as e:
        issues.append(f"Kafka connection failed: {e}")

    # Check AI services
    try:
        from app.services.ai.huggingface_service import hf_api
        test_result = await hf_api.generate_text(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            prompt="Hello",
            parameters={"max_tokens": 10}
        )
        if test_result and "text" in test_result:
            logger.info("✅ HuggingFace API working")
        else:
            issues.append("HuggingFace API test failed")
    except Exception as e:
        issues.append(f"HuggingFace API failed: {e}")

    if issues:
        logger.warning("⚠️  Some dependencies have issues:")
        for issue in issues:
            logger.warning(f"   - {issue}")
        logger.warning("Service will start but may have limited functionality")
    else:
        logger.info("✅ All dependencies are healthy")

    return len(issues) == 0


async def initialize_services():
    """Initialize all services"""
    logger.info("🚀 Initializing services...")

    try:
        # Initialize event producer
        from app.services.kafka.producer import initialize_event_producer
        await initialize_event_producer()
        logger.info("✅ Event producer initialized")
    except Exception as e:
        logger.warning(f"⚠️  Event producer initialization failed: {e}")

    logger.info("✅ Service initialization complete")


def run_fastapi_server():
    """Run the FastAPI server"""
    import uvicorn

    logger.info(f"🌟 Starting FastAPI server on port 8000")
    logger.info(f"📊 Debug mode: {settings.DEBUG}")
    logger.info(f"🌍 Environment: {settings.ENVIRONMENT}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )


async def main():
    """Main startup function"""
    logger.info("=" * 60)
    logger.info("🤖 FastAPI AI Workflow Service")
    logger.info("=" * 60)

    try:
        # Check dependencies
        all_healthy = await check_dependencies()

        # Initialize services
        await initialize_services()

        if not all_healthy:
            logger.warning("⚠️  Starting with some dependency issues...")

        logger.info("🎯 Ready to start server!")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)


def cli():
    """Command line interface"""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "check":
            # Just run dependency checks
            asyncio.run(check_dependencies())
            return
        elif command == "test":
            # Run the test script
            from test import test_enhanced_fastapi_service
            asyncio.run(test_enhanced_fastapi_service())
            return
        elif command == "worker":
            # Run as event processor worker
            from app.services.kafka.event_processor_service import main as worker_main
            asyncio.run(worker_main())
            return

    # Default: run full startup and server
    asyncio.run(main())
    run_fastapi_server()


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}")
        sys.exit(1)