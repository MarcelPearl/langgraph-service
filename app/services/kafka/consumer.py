import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from aiokafka import AIOKafkaConsumer
from app.core.kafka_config import get_consumer_config, kafka_settings
from app.schemas.events import EventType, EventPayload, ExecutionEventPayload
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

logger = logging.getLogger(__name__)


class WorkflowEventConsumer:
    """Async Kafka consumer for workflow events"""

    def __init__(self, db_session_factory: Callable[[], AsyncSession]):
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.is_running = False
        self.db_session_factory = db_session_factory
        self._config = get_consumer_config()
        self._handlers: Dict[EventType, List[Callable]] = {}
        self._error_handlers: List[Callable] = []

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default event handlers"""
        self.register_handler(EventType.EXECUTION_STARTED, self._handle_execution_started)
        self.register_handler(EventType.EXECUTION_COMPLETED, self._handle_execution_completed)
        self.register_handler(EventType.EXECUTION_FAILED, self._handle_execution_failed)
        self.register_handler(EventType.NODE_STARTED, self._handle_node_started)
        self.register_handler(EventType.NODE_COMPLETED, self._handle_node_completed)
        self.register_handler(EventType.NODE_FAILED, self._handle_node_failed)

    async def start(self, topics: Optional[List[str]] = None):
        """Start the Kafka consumer"""
        if self.is_running:
            return

        if topics is None:
            topics = [
                kafka_settings.KAFKA_WORKFLOW_EVENTS_TOPIC,
                kafka_settings.KAFKA_EXECUTION_EVENTS_TOPIC,
                kafka_settings.KAFKA_NODE_EVENTS_TOPIC
            ]

        try:
            self.consumer = AIOKafkaConsumer(
                *topics,
                **self._config,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )

            await self.consumer.start()
            self.is_running = True
            logger.info(f"Kafka consumer started, subscribed to topics: {topics}")

        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise

    async def stop(self):
        """Stop the Kafka consumer"""
        if self.consumer and self.is_running:
            try:
                await self.consumer.stop()
                self.is_running = False
                logger.info("Kafka consumer stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Kafka consumer: {e}")

    async def consume(self):
        """Main consumer loop"""
        if not self.is_running:
            logger.error("Consumer not started")
            return

        logger.info("Starting event consumption loop")

        try:
            async for message in self.consumer:
                await self._process_message(message)

        except Exception as e:
            logger.error(f"Error in consume loop: {e}")
            # Attempt to restart consumer
            await self._restart_consumer()

    async def _process_message(self, message):
        """Process a single Kafka message"""
        try:
            # Extract message data
            topic = message.topic
            partition = message.partition
            offset = message.offset
            key = message.key.decode('utf-8') if message.key else None
            value = message.value
            timestamp = message.timestamp

            logger.debug(
                f"Processing message: topic={topic}, partition={partition}, "
                f"offset={offset}, key={key}"
            )

            # Parse event data
            event_data = value
            event_type_str = event_data.get('event_type')

            if not event_type_str:
                logger.warning(f"Message missing event_type: {event_data}")
                await self._commit_message(message)
                return

            try:
                event_type = EventType(event_type_str)
            except ValueError:
                logger.warning(f"Unknown event type: {event_type_str}")
                await self._commit_message(message)
                return

            # Create event payload object
            event_payload = self._create_event_payload(event_type, event_data)

            # Process event with registered handlers
            await self._handle_event(event_type, event_payload, message)

            # Commit message after successful processing
            await self._commit_message(message)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._handle_processing_error(message, e)

    def _create_event_payload(self, event_type: EventType, event_data: Dict[str, Any]) -> EventPayload:
        """Create typed event payload from raw data"""
        try:
            # Import here to avoid circular imports
            from app.schemas.events import (
                ExecutionEventPayload, NodeEventPayload,
                WorkflowEventPayload, AIRequestEventPayload,
                ToolCallEventPayload, CheckpointEventPayload
            )

            if event_type.value.startswith('execution.'):
                return ExecutionEventPayload(**event_data)
            elif event_type.value.startswith('node.'):
                return NodeEventPayload(**event_data)
            elif event_type.value.startswith('workflow.'):
                return WorkflowEventPayload(**event_data)
            elif event_type.value.startswith('ai.'):
                return AIRequestEventPayload(**event_data)
            elif event_type.value.startswith('tool.'):
                return ToolCallEventPayload(**event_data)
            elif event_type.value.startswith('checkpoint.'):
                return CheckpointEventPayload(**event_data)
            else:
                # Fallback to base payload
                from app.schemas.events import BaseEventPayload
                return BaseEventPayload(**event_data)

        except Exception as e:
            logger.error(f"Error creating event payload: {e}")
            # Return a basic payload if parsing fails
            from app.schemas.events import BaseEventPayload
            return BaseEventPayload(
                event_type=event_type,
                event_id=event_data.get('event_id', str(uuid.uuid4()))
            )

    async def _handle_event(self, event_type: EventType, payload: EventPayload, message):
        """Handle event with registered handlers"""
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug(f"No handlers registered for event type: {event_type}")
            return

        for handler in handlers:
            try:
                await handler(payload, message)
            except Exception as e:
                logger.error(f"Error in event handler {handler.__name__}: {e}")
                # Continue with other handlers

    async def _commit_message(self, message):
        """Commit message offset"""
        try:
            await self.consumer.commit({
                message.topic_partition(): message.offset + 1
            })
        except Exception as e:
            logger.error(f"Error committing message: {e}")

    async def _handle_processing_error(self, message, error: Exception):
        """Handle message processing errors"""
        logger.error(f"Failed to process message: {error}")

        # Call error handlers
        for error_handler in self._error_handlers:
            try:
                await error_handler(message, error)
            except Exception as handler_error:
                logger.error(f"Error in error handler: {handler_error}")

        # For now, commit the message to avoid infinite retry
        # In production, you might want to send to a dead letter queue
        await self._commit_message(message)

    async def _restart_consumer(self):
        """Restart consumer after error"""
        logger.info("Attempting to restart consumer...")
        try:
            await self.stop()
            await asyncio.sleep(5)  # Wait before restart
            await self.start()
            logger.info("Consumer restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart consumer: {e}")

    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info(f"Registered handler {handler.__name__} for {event_type}")

    def register_error_handler(self, handler: Callable):
        """Register error handler"""
        self._error_handlers.append(handler)
        logger.info(f"Registered error handler {handler.__name__}")

    # Default event handlers
    async def _handle_execution_started(self, payload: ExecutionEventPayload, message):
        """Handle execution started event"""
        logger.info(f"Execution started: {payload.execution_id}")

        # Update database record
        async with self.db_session_factory() as db:
            try:
                from sqlalchemy import select, update
                from app.models.workflow import WorkflowExecution

                await db.execute(
                    update(WorkflowExecution)
                    .where(WorkflowExecution.id == payload.execution_id)
                    .values(
                        status="running",
                        started_at=payload.timestamp,
                        current_step=payload.current_step
                    )
                )
                await db.commit()

            except Exception as e:
                logger.error(f"Error updating execution record: {e}")
                await db.rollback()

    async def _handle_execution_completed(self, payload: ExecutionEventPayload, message):
        """Handle execution completed event"""
        logger.info(f"Execution completed: {payload.execution_id}")

        # Update database record
        async with self.db_session_factory() as db:
            try:
                from sqlalchemy import update
                from app.models.workflow import WorkflowExecution

                await db.execute(
                    update(WorkflowExecution)
                    .where(WorkflowExecution.id == payload.execution_id)
                    .values(
                        status="completed",
                        completed_at=payload.timestamp,
                        output_data=payload.output_data,
                        steps_completed=payload.steps_completed,
                        ai_tokens_used=payload.tokens_used
                    )
                )
                await db.commit()

            except Exception as e:
                logger.error(f"Error updating execution record: {e}")
                await db.rollback()

    async def _handle_execution_failed(self, payload: ExecutionEventPayload, message):
        """Handle execution failed event"""
        logger.error(f"Execution failed: {payload.execution_id}")

        # Update database record
        async with self.db_session_factory() as db:
            try:
                from sqlalchemy import update
                from app.models.workflow import WorkflowExecution

                await db.execute(
                    update(WorkflowExecution)
                    .where(WorkflowExecution.id == payload.execution_id)
                    .values(
                        status="failed",
                        completed_at=payload.timestamp,
                        error_data=payload.error_data,
                        steps_completed=payload.steps_completed
                    )
                )
                await db.commit()

            except Exception as e:
                logger.error(f"Error updating execution record: {e}")
                await db.rollback()

    async def _handle_node_started(self, payload, message):
        """Handle node started event"""
        logger.debug(f"Node started: {payload.node_id}")

    async def _handle_node_completed(self, payload, message):
        """Handle node completed event"""
        logger.debug(f"Node completed: {payload.node_id}")

    async def _handle_node_failed(self, payload, message):
        """Handle node failed event"""
        logger.warning(f"Node failed: {payload.node_id}")


class EventProcessor:
    """Background service to run the event consumer"""

    def __init__(self, db_session_factory: Callable[[], AsyncSession]):
        self.consumer = WorkflowEventConsumer(db_session_factory)
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the event processor"""
        if self._task and not self._task.done():
            return

        await self.consumer.start()
        self._task = asyncio.create_task(self.consumer.consume())
        logger.info("Event processor started")

    async def stop(self):
        """Stop the event processor"""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self.consumer.stop()
        logger.info("Event processor stopped")

    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        self.consumer.register_handler(event_type, handler)

    def register_error_handler(self, handler: Callable):
        """Register error handler"""
        self.consumer.register_error_handler(handler)