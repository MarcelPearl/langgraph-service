import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError
from aiokafka import AIOKafkaProducer
from app.core.kafka_config import get_producer_config, kafka_settings
from app.schemas.events import EventPayload, EventType, KafkaMessage
import uuid

logger = logging.getLogger(__name__)


class WorkflowEventProducer:
    """Async Kafka producer for workflow events"""

    def __init__(self):
        self.producer: Optional[AIOKafkaProducer] = None
        self.is_started = False
        self._config = get_producer_config()
        self._dead_letter_topic = kafka_settings.KAFKA_DEAD_LETTER_TOPIC

    async def start(self):
        """Start the Kafka producer"""
        if self.is_started:
            return

        try:
            self.producer = AIOKafkaProducer(
                **self._config,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )

            await self.producer.start()
            self.is_started = True
            logger.info("Kafka producer started successfully")

        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise

    async def stop(self):
        """Stop the Kafka producer"""
        if self.producer and self.is_started:
            try:
                await self.producer.stop()
                self.is_started = False
                logger.info("Kafka producer stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")

    async def publish_event(
            self,
            event: EventPayload,
            topic: Optional[str] = None,
            key: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Publish a single event to Kafka"""
        if not self.is_started:
            logger.warning("Producer not started, attempting to start...")
            await self.start()

        try:
            # Determine topic based on event type if not specified
            if topic is None:
                topic = self._get_topic_for_event_type(event.event_type)

            # Use workflow_id or execution_id as key for partitioning
            if key is None:
                key = self._generate_key(event)

            # Prepare headers
            if headers is None:
                headers = {}

            headers.update({
                'event_type': event.event_type.value,
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'source_service': event.source_service
            })

            # Convert event to dictionary
            event_dict = event.model_dump()

            # Send message
            future = await self.producer.send(
                topic=topic,
                value=event_dict,
                key=key,
                headers=[(k, v.encode('utf-8')) for k, v in headers.items()]
            )

            # Get record metadata
            record_metadata = await future

            logger.debug(
                f"Event published successfully: "
                f"topic={record_metadata.topic}, "
                f"partition={record_metadata.partition}, "
                f"offset={record_metadata.offset}, "
                f"event_type={event.event_type.value}"
            )

            return True

        except KafkaTimeoutError as e:
            logger.error(f"Kafka timeout publishing event {event.event_id}: {e}")
            await self._send_to_dead_letter_queue(event, str(e))
            return False

        except KafkaError as e:
            logger.error(f"Kafka error publishing event {event.event_id}: {e}")
            await self._send_to_dead_letter_queue(event, str(e))
            return False

        except Exception as e:
            logger.error(f"Unexpected error publishing event {event.event_id}: {e}")
            await self._send_to_dead_letter_queue(event, str(e))
            return False

    async def publish_batch(
            self,
            events: List[EventPayload],
            topic: Optional[str] = None
    ) -> Dict[str, int]:
        """Publish a batch of events"""
        results = {"success": 0, "failed": 0}

        if not events:
            return results

        logger.info(f"Publishing batch of {len(events)} events")

        # Create tasks for concurrent publishing
        tasks = []
        for event in events:
            task = asyncio.create_task(
                self.publish_event(event, topic)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, bool) and result:
                    results["success"] += 1
                else:
                    results["failed"] += 1

        except Exception as e:
            logger.error(f"Error in batch publishing: {e}")
            results["failed"] = len(events)

        logger.info(
            f"Batch publishing complete: "
            f"{results['success']} success, {results['failed']} failed"
        )

        return results

    async def publish_workflow_event(
            self,
            event_type: EventType,
            workflow_id: str,
            payload: Dict[str, Any]
    ) -> bool:
        """Convenience method for workflow events"""
        from app.schemas.events import create_execution_event

        event = create_execution_event(
            event_type=event_type,
            execution_id=payload.get('execution_id', str(uuid.uuid4())),
            workflow_id=workflow_id,
            execution_status=payload.get('status', 'unknown'),
            **payload
        )

        return await self.publish_event(event)

    def _get_topic_for_event_type(self, event_type: EventType) -> str:
        """Get appropriate topic for event type"""
        if event_type.value.startswith('workflow.'):
            return kafka_settings.KAFKA_WORKFLOW_EVENTS_TOPIC
        elif event_type.value.startswith('execution.'):
            return kafka_settings.KAFKA_EXECUTION_EVENTS_TOPIC
        elif event_type.value.startswith('node.'):
            return kafka_settings.KAFKA_NODE_EVENTS_TOPIC
        else:
            return kafka_settings.KAFKA_WORKFLOW_EVENTS_TOPIC

    def _generate_key(self, event: EventPayload) -> str:
        """Generate partition key for event"""
        # Use workflow_id for workflow events
        if hasattr(event, 'workflow_id'):
            return event.workflow_id

        # Use execution_id for execution events
        if hasattr(event, 'execution_id'):
            return event.execution_id

        # Fallback to event_id
        return event.event_id

    async def _send_to_dead_letter_queue(self, event: EventPayload, error: str):
        """Send failed event to dead letter queue"""
        try:
            dead_letter_payload = {
                'original_event': event.model_dump(),
                'error': error,
                'failed_at': datetime.utcnow().isoformat(),
                'retry_count': 0
            }

            await self.producer.send(
                topic=self._dead_letter_topic,
                value=dead_letter_payload,
                key=event.event_id
            )

            logger.info(f"Event {event.event_id} sent to dead letter queue")

        except Exception as dlq_error:
            logger.error(f"Failed to send event to dead letter queue: {dlq_error}")


class SyncWorkflowEventProducer:
    """Synchronous Kafka producer for compatibility"""

    def __init__(self):
        self.producer: Optional[KafkaProducer] = None
        self.is_started = False
        self._config = get_producer_config()

    def start(self):
        """Start the sync Kafka producer"""
        if self.is_started:
            return

        try:
            self.producer = KafkaProducer(
                **self._config,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )

            self.is_started = True
            logger.info("Sync Kafka producer started successfully")

        except Exception as e:
            logger.error(f"Failed to start sync Kafka producer: {e}")
            raise

    def stop(self):
        """Stop the sync Kafka producer"""
        if self.producer and self.is_started:
            try:
                self.producer.close()
                self.is_started = False
                logger.info("Sync Kafka producer stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping sync Kafka producer: {e}")

    def publish_event_sync(
            self,
            event: EventPayload,
            topic: Optional[str] = None,
            timeout: int = 10
    ) -> bool:
        """Publish event synchronously"""
        if not self.is_started:
            self.start()

        try:
            if topic is None:
                topic = self._get_topic_for_event_type(event.event_type)

            key = self._generate_key(event)
            event_dict = event.model_dump()

            future = self.producer.send(topic, value=event_dict, key=key)
            record_metadata = future.get(timeout=timeout)

            logger.debug(f"Sync event published: topic={record_metadata.topic}")
            return True

        except Exception as e:
            logger.error(f"Sync publish error: {e}")
            return False

    def _get_topic_for_event_type(self, event_type: EventType) -> str:
        """Get appropriate topic for event type"""
        if event_type.value.startswith('workflow.'):
            return kafka_settings.KAFKA_WORKFLOW_EVENTS_TOPIC
        elif event_type.value.startswith('execution.'):
            return kafka_settings.KAFKA_EXECUTION_EVENTS_TOPIC
        elif event_type.value.startswith('node.'):
            return kafka_settings.KAFKA_NODE_EVENTS_TOPIC
        else:
            return kafka_settings.KAFKA_WORKFLOW_EVENTS_TOPIC

    def _generate_key(self, event: EventPayload) -> str:
        """Generate partition key for event"""
        if hasattr(event, 'workflow_id'):
            return event.workflow_id
        if hasattr(event, 'execution_id'):
            return event.execution_id
        return event.event_id


# Global instances
event_producer = WorkflowEventProducer()
sync_event_producer = SyncWorkflowEventProducer()