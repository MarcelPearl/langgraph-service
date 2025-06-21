# app/services/kafka/producer.py
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaTimeoutError
from app.core.config import settings
from app.schemas.events import BaseEvent
from datetime import datetime

logger = logging.getLogger(__name__)


class EventProducer:
    """Kafka event producer for publishing workflow events"""

    def __init__(self):
        self.bootstrap_servers = getattr(settings, 'KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.producer: Optional[AIOKafkaProducer] = None
        self.running = False

        # Topic for workflow events
        self.events_topic = "workflow-events"

    async def start(self):
        """Start the Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                compression_type="gzip",
                request_timeout_ms=30000,
                retry_backoff_ms=1000,
                max_request_size=1048576,  # 1MB
                acks='all',  # Wait for all replicas to acknowledge
                retries=3
            )

            await self.producer.start()
            self.running = True
            logger.info("Event producer started successfully")

        except Exception as e:
            logger.error(f"Failed to start event producer: {e}")
            raise

    async def stop(self):
        """Stop the Kafka producer"""
        self.running = False

        if self.producer:
            await self.producer.stop()
            logger.info("Event producer stopped")

    async def publish_event(self, event: BaseEvent, topic: Optional[str] = None):
        """Publish an event to Kafka"""
        if not self.producer or not self.running:
            logger.warning("Event producer not running, event will be logged only")
            logger.info(f"Event: {event.dict()}")
            return

        topic = topic or self.events_topic

        try:
            # Convert event to dictionary
            event_data = event.dict()

            # Add metadata
            event_data["published_at"] = datetime.utcnow().isoformat()
            event_data["producer_service"] = "fastapi"

            # Publish to Kafka
            await self.producer.send_and_wait(topic, event_data)

            logger.debug(f"Published event {event.event_type} to topic {topic}")

        except KafkaTimeoutError:
            logger.error(f"Timeout publishing event {event.event_type} to {topic}")
        except KafkaError as e:
            logger.error(f"Kafka error publishing event {event.event_type}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error publishing event {event.event_type}: {e}")

    async def publish_raw_event(self, event_data: Dict[str, Any], topic: Optional[str] = None):
        """Publish raw event data to Kafka"""
        if not self.producer or not self.running:
            logger.warning("Event producer not running, event will be logged only")
            logger.info(f"Raw event: {event_data}")
            return

        topic = topic or self.events_topic

        try:
            # Add metadata
            event_data["published_at"] = datetime.utcnow().isoformat()
            event_data["producer_service"] = "fastapi"

            # Publish to Kafka
            await self.producer.send_and_wait(topic, event_data)

            logger.debug(f"Published raw event to topic {topic}")

        except Exception as e:
            logger.error(f"Error publishing raw event: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check producer health"""
        return {
            "service": "event_producer",
            "status": "healthy" if self.running and self.producer else "unhealthy",
            "producer_connected": self.producer is not None,
            "timestamp": datetime.utcnow().isoformat()
        }


class MockEventProducer:
    """Mock event producer for testing without Kafka"""

    def __init__(self):
        self.running = False
        self.events = []  # Store events for testing

    async def start(self):
        """Start mock producer"""
        self.running = True
        logger.info("Mock event producer started")

    async def stop(self):
        """Stop mock producer"""
        self.running = False
        logger.info("Mock event producer stopped")

    async def publish_event(self, event: BaseEvent, topic: Optional[str] = None):
        """Mock publish event"""
        if not self.running:
            return

        event_data = event.dict()
        event_data["published_at"] = datetime.utcnow().isoformat()
        event_data["topic"] = topic or "workflow-events"

        self.events.append(event_data)
        logger.info(f"Mock published event: {event.event_type}")

    async def publish_raw_event(self, event_data: Dict[str, Any], topic: Optional[str] = None):
        """Mock publish raw event"""
        if not self.running:
            return

        event_data["published_at"] = datetime.utcnow().isoformat()
        event_data["topic"] = topic or "workflow-events"

        self.events.append(event_data)
        logger.info(f"Mock published raw event")

    async def health_check(self) -> Dict[str, Any]:
        """Mock health check"""
        return {
            "service": "mock_event_producer",
            "status": "healthy" if self.running else "unhealthy",
            "events_published": len(self.events),
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_events(self):
        """Get all published events (for testing)"""
        return self.events.copy()

    def clear_events(self):
        """Clear all events (for testing)"""
        self.events.clear()


# Global instance - will be initialized based on environment
event_producer = None


async def get_event_producer() -> EventProducer:
    """Get the global event producer instance"""
    global event_producer

    if event_producer is None:
        # Use mock producer in test/dev environments without Kafka
        if getattr(settings, 'ENVIRONMENT', 'development') == 'test':
            event_producer = MockEventProducer()
        else:
            event_producer = EventProducer()

        await event_producer.start()

    return event_producer


async def initialize_event_producer():
    """Initialize the global event producer"""
    await get_event_producer()


async def shutdown_event_producer():
    """Shutdown the global event producer"""
    global event_producer

    if event_producer:
        await event_producer.stop()
        event_producer = None