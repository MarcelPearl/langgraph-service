#!/usr/bin/env python3
"""
Background Event Processor Service

This service runs as a separate container to process Kafka events.
It provides:
- Event consumption from Kafka topics
- Event processing and database updates
- Error handling and dead letter queue processing
- Health monitoring and graceful shutdown
"""

import asyncio
import signal
import logging
import sys
import json
from typing import Optional
from contextlib import asynccontextmanager

# Add the app directory to the Python path
sys.path.append('/app')

from app.core.database import async_session_maker, init_database
from app.services.kafka.consumer import EventProcessor
from app.schemas.events import EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/event_processor.log')
    ]
)

logger = logging.getLogger(__name__)


class EventProcessorService:
    """Main event processor service"""

    def __init__(self):
        self.processor: Optional[EventProcessor] = None
        self.is_running = False
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the event processor service"""
        logger.info("Starting Event Processor Service")

        try:
            # Initialize database
            await init_database()
            logger.info("Database initialized")

            # Create event processor
            self.processor = EventProcessor(self._get_db_session)

            # Register custom event handlers
            await self._register_custom_handlers()

            # Start the processor
            await self.processor.start()
            self.is_running = True

            logger.info("Event Processor Service started successfully")

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error(f"Failed to start Event Processor Service: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Stop the event processor service"""
        logger.info("Stopping Event Processor Service")

        if self.processor:
            await self.processor.stop()

        self.is_running = False
        logger.info("Event Processor Service stopped")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self._shutdown_event.set()

    @asynccontextmanager
    async def _get_db_session(self):
        """Get database session"""
        async with async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def _register_custom_handlers(self):
        """Register custom event handlers"""
        if not self.processor:
            return

        # Register analytics handler
        self.processor.register_handler(
            EventType.EXECUTION_COMPLETED,
            self._handle_execution_analytics
        )

        # Register performance monitoring
        self.processor.register_handler(
            EventType.AI_REQUEST_COMPLETED,
            self._handle_ai_performance_tracking
        )

        # Register error alerting
        self.processor.register_error_handler(
            self._handle_processing_errors
        )

        logger.info("Custom event handlers registered")

    async def _handle_execution_analytics(self, payload, message):
        """Handle execution completion for analytics"""
        try:
            logger.info(f"Processing analytics for execution: {payload.execution_id}")

            # Here you would typically:
            # 1. Update analytics tables
            # 2. Calculate metrics (execution time, success rate, etc.)
            # 3. Update workflow performance statistics
            # 4. Trigger notifications if needed

            execution_time = payload.execution_time_ms
            if execution_time:
                logger.info(f"Execution time: {execution_time}ms")

            tokens_used = payload.tokens_used
            if tokens_used:
                logger.info(f"AI tokens used: {tokens_used}")

            # Update workflow statistics
            async with self._get_db_session() as db:
                from sqlalchemy import text

                # Example: Update workflow execution statistics
                await db.execute(
                    text("""
                        INSERT INTO workflow_stats (workflow_id, executions_count, total_tokens, avg_execution_time)
                        VALUES (:workflow_id, 1, :tokens, :exec_time)
                        ON CONFLICT (workflow_id) DO UPDATE SET
                            executions_count = workflow_stats.executions_count + 1,
                                                         total_tokens = workflow_stats.total_tokens + :tokens,
                            avg_execution_time = (workflow_stats.avg_execution_time + :exec_time) / 2,
                            updated_at = NOW()
                    """),
                    {
                        'workflow_id': payload.workflow_id,
                        'tokens': tokens_used or 0,
                        'exec_time': execution_time or 0
                    }
                )
                await db.commit()

        except Exception as e:
            logger.error(f"Error processing execution analytics: {e}")

    async def _handle_ai_performance_tracking(self, payload, message):
        """Handle AI request performance tracking"""
        try:
            logger.debug(f"Tracking AI performance: {payload.ai_provider}/{payload.ai_model}")

            # Track AI model performance metrics
            response_time = payload.response_time_ms
            tokens_used = payload.total_tokens
            cost = payload.cost_estimate

            # Here you would typically:
            # 1. Store performance metrics
            # 2. Update model performance statistics
            # 3. Detect anomalies or performance degradation
            # 4. Update cost tracking

            if response_time and response_time > 30000:  # 30 seconds
                logger.warning(f"Slow AI response detected: {response_time}ms")

            if cost and cost > 1.0:  # $1.00
                logger.warning(f"High cost AI request: ${cost}")

        except Exception as e:
            logger.error(f"Error tracking AI performance: {e}")

    async def _handle_processing_errors(self, message, error):
        """Handle event processing errors"""
        try:
            logger.error(f"Event processing error: {error}")

            # Here you would typically:
            # 1. Send alerts for critical errors
            # 2. Update error tracking metrics
            # 3. Determine if manual intervention is needed
            # 4. Log detailed error information

            error_data = {
                'topic': message.topic,
                'partition': message.partition,
                'offset': message.offset,
                'error': str(error),
                'timestamp': message.timestamp
            }

            # Log structured error data
            logger.error(f"Structured error data: {error_data}")

            # Update error metrics
            async with self._get_db_session() as db:
                from sqlalchemy import text

                await db.execute(
                    text("""
                        INSERT INTO event_processing_errors (topic, partition_id, offset_id, error_message, created_at)
                        VALUES (:topic, :partition, :offset, :error, NOW())
                    """),
                    {
                        'topic': message.topic,
                        'partition': message.partition,
                        'offset': message.offset,
                        'error': str(error)
                    }
                )
                await db.commit()

        except Exception as e:
            logger.error(f"Error handling processing error: {e}")


def setup_signal_handlers(service: EventProcessorService):
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, service.signal_handler)
    signal.signal(signal.SIGTERM, service.signal_handler)


async def main():
    """Main entry point"""
    logger.info("Initializing Event Processor Service")

    service = EventProcessorService()
    setup_signal_handlers(service)

    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Service error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)