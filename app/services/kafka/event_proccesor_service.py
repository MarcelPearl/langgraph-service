# app/services/kafka/event_processor_service.py
import asyncio
import json
import logging
import signal
import sys
from typing import Dict, Any
from aiokafka import AIOKafkaConsumer
from datetime import datetime

# Add the app directory to Python path
sys.path.append('/app')

from app.core.config import settings
from app.services.context.redis_service import redis_context_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventProcessor:
    """Background service to process workflow events"""

    def __init__(self):
        self.bootstrap_servers = getattr(settings, 'KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.consumer: AIOKafkaConsumer = None
        self.running = False

        # Topics to consume
        self.topics = [
            "workflow-events",
            "node-completion-events",
            "workflow-state-updates"
        ]

    async def start(self):
        """Start the event processor"""
        try:
            # Connect to Redis
            await redis_context_service.connect()
            logger.info("✅ Connected to Redis")

            # Setup Kafka consumer
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id="event-processor",
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=5000
            )

            await self.consumer.start()
            logger.info("✅ Connected to Kafka")

            self.running = True
            logger.info(f"🚀 Event processor started, consuming from topics: {self.topics}")

            # Start consuming
            await self.consume_events()

        except Exception as e:
            logger.error(f"❌ Failed to start event processor: {e}")
            raise

    async def stop(self):
        """Stop the event processor"""
        self.running = False

        if self.consumer:
            await self.consumer.stop()
            logger.info("Kafka consumer stopped")

        await redis_context_service.disconnect()
        logger.info("Redis connection closed")

        logger.info("✅ Event processor stopped")

    async def consume_events(self):
        """Main event consumption loop"""
        try:
            async for message in self.consumer:
                if not self.running:
                    break

                try:
                    await self.process_event(message)
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    # Continue processing other events

        except Exception as e:
            logger.error(f"Error in event consumption loop: {e}")
            raise

    async def process_event(self, message):
        """Process a single event"""
        try:
            event_data = message.value
            topic = message.topic

            logger.debug(f"Processing event from topic {topic}: {event_data.get('event_type', 'unknown')}")

            if topic == "workflow-events":
                await self.handle_workflow_event(event_data)
            elif topic == "node-completion-events":
                await self.handle_node_completion_event(event_data)
            elif topic == "workflow-state-updates":
                await self.handle_state_update_event(event_data)
            else:
                logger.warning(f"Unknown topic: {topic}")

        except Exception as e:
            logger.error(f"Error processing event: {e}")

    async def handle_workflow_event(self, event_data: Dict[str, Any]):
        """Handle workflow lifecycle events"""
        try:
            event_type = event_data.get("event_type")
            execution_id = event_data.get("execution_id")

            if not execution_id:
                logger.warning("Workflow event missing execution_id")
                return

            # Update execution context with event information
            context_update = {
                "last_event": {
                    "type": event_type,
                    "timestamp": event_data.get("timestamp"),
                    "data": event_data
                }
            }

            await redis_context_service.update_execution_context(execution_id, context_update)

            # Handle specific event types
            if event_type == "execution_started":
                await self.handle_execution_started(execution_id, event_data)
            elif event_type == "execution_completed":
                await self.handle_execution_completed(execution_id, event_data)
            elif event_type == "execution_failed":
                await self.handle_execution_failed(execution_id, event_data)

            logger.info(f"Processed workflow event: {event_type} for execution {execution_id}")

        except Exception as e:
            logger.error(f"Error handling workflow event: {e}")

    async def handle_node_completion_event(self, event_data: Dict[str, Any]):
        """Handle node completion events"""
        try:
            execution_id = event_data.get("execution_id")
            node_id = event_data.get("node_id")
            status = event_data.get("status")

            if not all([execution_id, node_id, status]):
                logger.warning("Node completion event missing required fields")
                return

            # Update node status in Redis
            if status == "completed":
                await redis_context_service.mark_completed(execution_id, node_id)

            # Store result if present
            result = event_data.get("result")
            if result:
                await redis_context_service.store_node_result(execution_id, node_id, result)

            logger.info(f"Processed node completion: {node_id} -> {status} in execution {execution_id}")

        except Exception as e:
            logger.error(f"Error handling node completion event: {e}")

    async def handle_state_update_event(self, event_data: Dict[str, Any]):
        """Handle workflow state updates"""
        try:
            execution_id = event_data.get("execution_id")
            status = event_data.get("status")
            data = event_data.get("data", {})

            if not execution_id:
                logger.warning("State update event missing execution_id")
                return

            # Update workflow status
            await redis_context_service.set_workflow_status(
                execution_id=execution_id,
                status=status,
                metadata=data
            )

            logger.debug(f"Updated workflow status: {execution_id} -> {status}")

        except Exception as e:
            logger.error(f"Error handling state update event: {e}")

    async def handle_execution_started(self, execution_id: str, event_data: Dict[str, Any]):
        """Handle execution started event"""
        try:
            # Initialize execution tracking
            context_update = {
                "execution_started_at": event_data.get("timestamp"),
                "execution_status": "running",
                "events_processed": 1
            }

            await redis_context_service.update_execution_context(execution_id, context_update)

            logger.info(f"Execution {execution_id} started tracking")

        except Exception as e:
            logger.error(f"Error handling execution started: {e}")

    async def handle_execution_completed(self, execution_id: str, event_data: Dict[str, Any]):
        """Handle execution completed event"""
        try:
            # Update final status
            context_update = {
                "execution_completed_at": event_data.get("timestamp"),
                "execution_status": "completed",
                "final_output": event_data.get("output_data"),
                "total_tokens_used": event_data.get("tokens_used", 0),
                "execution_time_ms": event_data.get("execution_time_ms", 0)
            }

            await redis_context_service.update_execution_context(execution_id, context_update)

            logger.info(f"Execution {execution_id} completed successfully")

        except Exception as e:
            logger.error(f"Error handling execution completed: {e}")

    async def handle_execution_failed(self, execution_id: str, event_data: Dict[str, Any]):
        """Handle execution failed event"""
        try:
            # Update failure status
            context_update = {
                "execution_failed_at": event_data.get("timestamp"),
                "execution_status": "failed",
                "error_data": event_data.get("error_data"),
                "execution_time_ms": event_data.get("execution_time_ms", 0)
            }

            await redis_context_service.update_execution_context(execution_id, context_update)

            logger.error(f"Execution {execution_id} failed: {event_data.get('error_data')}")

        except Exception as e:
            logger.error(f"Error handling execution failed: {e}")

    async def cleanup_old_executions(self):
        """Periodic cleanup of old execution data"""
        try:
            # Run cleanup every hour
            await redis_context_service.cleanup_expired_executions(older_than_hours=24)
            logger.info("Completed periodic cleanup of old executions")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global instance
event_processor = EventProcessor()


async def run_periodic_cleanup():
    """Run periodic cleanup tasks"""
    while event_processor.running:
        try:
            await asyncio.sleep(3600)  # Wait 1 hour
            await event_processor.cleanup_old_executions()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")


async def signal_handler():
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    await event_processor.stop()


async def main():
    """Main function"""
    # Setup signal handlers
    loop = asyncio.get_event_loop()

    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, lambda: asyncio.create_task(signal_handler()))

    try:
        # Start event processor
        processor_task = asyncio.create_task(event_processor.start())

        # Start periodic cleanup
        cleanup_task = asyncio.create_task(run_periodic_cleanup())

        # Wait for tasks
        await asyncio.gather(processor_task, cleanup_task)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Event processor failed: {e}")
        raise
    finally:
        await event_processor.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Event processor stopped by user")
    except Exception as e:
        print(f"\n💥 Event processor failed: {e}")
        sys.exit(1)