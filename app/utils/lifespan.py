import asyncio
from contextlib import asynccontextmanager
from app.services.messaging.kafka_service import kafka_service
from app.services.context.redis_service import redis_context_service
from app.services.execution.node_engine import node_execution_engine
import logging

logger = logging.getLogger(__name__)

kafka_consumer_task = None

async def start_kafka_consumer():
    global kafka_consumer_task
    if kafka_service.running and kafka_service.consumers:
        kafka_consumer_task = asyncio.create_task(kafka_service.start_consuming())
        logger.info("Kafka consumer started")

async def stop_kafka_consumer():
    global kafka_consumer_task
    if kafka_consumer_task and not kafka_consumer_task.done():
        kafka_consumer_task.cancel()
        try:
            await asyncio.wait_for(kafka_consumer_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logger.info("Kafka consumer stopped")

@asynccontextmanager
async def lifespan(app):
    logger.info("Starting FastAPI AI Engine...")

    services = {'redis': False, 'kafka': False, 'node_engine': False, 'kafka_consumer': False}

    try:
        try:
            await redis_context_service.connect()
            services['redis'] = True
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.error(f"❌ Redis error: {e}")

        try:
            await kafka_service.start()
            services['kafka'] = True
            logger.info("✅ Kafka service started")
        except Exception as e:
            logger.error(f"❌ Kafka error: {e}")

        try:
            await node_execution_engine.start()
            services['node_engine'] = True
            logger.info("✅ Node engine started")
        except Exception as e:
            logger.error(f"❌ Node engine error: {e}")

        if services['kafka']:
            try:
                await start_kafka_consumer()
                services['kafka_consumer'] = True
            except Exception as e:
                logger.error(f"❌ Kafka consumer error: {e}")

        yield

    finally:
        logger.info("Shutting down...")
        if services['kafka_consumer']:
            await stop_kafka_consumer()
        if services['node_engine']:
            await node_execution_engine.stop()
        if services['kafka']:
            await kafka_service.stop()
        if services['redis']:
            await redis_context_service.disconnect()
        logger.info("✅ Shutdown complete")
