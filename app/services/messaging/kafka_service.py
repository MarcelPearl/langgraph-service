import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable, List
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer, TopicPartition
from aiokafka.errors import KafkaError, KafkaTimeoutError
from app.core.config import settings
from dataclasses import asdict, dataclass
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class NodeExecutionMessage:
    """Message format for node execution"""
    execution_id: str
    workflow_id: str
    node_id: str
    node_type: str
    node_data: Dict[str, Any]
    context: Dict[str, Any]
    dependencies: List[str]
    timestamp: str
    priority: str = "normal"


@dataclass
class CompletionEvent:
    """Message format for completion events"""
    execution_id: str
    node_id: str
    status: str  # completed, failed, retry
    result: Dict[str, Any]
    next_possible_nodes: List[str]
    execution_time_ms: int
    service: str
    error: Optional[str] = None


class KafkaService:
    """Enhanced Kafka service for workflow coordination - FIXED VERSION"""

    def __init__(self):
        self.bootstrap_servers = getattr(settings, 'KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.running = False
        self.message_handlers: Dict[str, Callable] = {}

        # Topic definitions from architecture
        self.topics = {
            'coordination': 'workflow-coordination',
            'fastapi_queue': 'fastapi-execution-queue',
            'spring_queue': 'spring-execution-queue',
            'completion': 'node-completion-events',
            'state_updates': 'workflow-state-updates'
        }

    async def start(self):
        """Start Kafka producer and consumers"""
        try:
            # Initialize producer with better error handling
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                compression_type="gzip",
                request_timeout_ms=30000,
                retry_backoff_ms=1000,
                max_request_size=1048576,  # 1MB
                acks=1,  # Changed from 'all' to reduce timeout issues
                retries=3,
                enable_idempotence=False  # Disable to avoid potential issues
            )
            await self.producer.start()

            # Initialize consumers for FastAPI queues with better error handling
            await self._setup_consumers()

            self.running = True
            logger.info("Kafka service started successfully")

        except Exception as e:
            logger.error(f"Failed to start Kafka service: {e}")
            # Don't raise - allow graceful degradation
            self.running = False

    async def stop(self):
        """Stop Kafka producer and consumers"""
        self.running = False

        # Stop producer
        if self.producer:
            try:
                await self.producer.stop()
                logger.info("Kafka producer stopped")
            except Exception as e:
                logger.error(f"Error stopping producer: {e}")

        # Stop all consumers
        for consumer_name, consumer in self.consumers.items():
            try:
                await consumer.stop()
                logger.info(f"Kafka consumer {consumer_name} stopped")
            except Exception as e:
                logger.error(f"Error stopping consumer {consumer_name}: {e}")

        logger.info("Kafka service stopped")

    async def _setup_consumers(self):
        """Setup consumers for AI execution queues"""
        try:
            # Consumer for AI/ML node execution
            ai_consumer = AIOKafkaConsumer(
                self.topics['fastapi_queue'],
                bootstrap_servers=self.bootstrap_servers,
                group_id='fastapi-ai-workers',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                request_timeout_ms=30000,
                retry_backoff_ms=1000
            )

            await ai_consumer.start()
            self.consumers['ai_execution'] = ai_consumer
            logger.info("AI execution consumer started")

            # Consumer for coordination messages
            coord_consumer = AIOKafkaConsumer(
                self.topics['coordination'],
                bootstrap_servers=self.bootstrap_servers,
                group_id='fastapi-coordination',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                request_timeout_ms=30000,
                retry_backoff_ms=1000
            )

            await coord_consumer.start()
            self.consumers['coordination'] = coord_consumer
            logger.info("Coordination consumer started")

        except Exception as e:
            logger.error(f"Failed to setup consumers: {e}")
            # Continue without consumers

    async def publish_node_execution(self, message: NodeExecutionMessage, topic: str = None):
        """Publish node execution message"""
        if not self.producer or not self.running:
            logger.warning("Kafka producer not available, skipping message publish")
            return

        topic = topic or self.topics['fastapi_queue']

        try:
            message_dict = asdict(message)
            await self.producer.send_and_wait(topic, message_dict)
            logger.debug(f"Published execution message for node {message.node_id}")

        except KafkaTimeoutError:
            logger.error(f"Timeout publishing to {topic}")
            raise
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise

    async def publish_completion_event(self, event: CompletionEvent):
        """Publish node completion event"""
        if not self.producer or not self.running:
            logger.warning("Kafka producer not available, skipping completion event")
            return

        try:
            event_dict = asdict(event)
            await self.producer.send_and_wait(self.topics['completion'], event_dict)
            logger.debug(f"Published completion event for node {event.node_id}")

        except Exception as e:
            logger.error(f"Failed to publish completion event: {e}")
            # Don't raise - completion events are not critical

    async def publish_state_update(self, execution_id: str, status: str, data: Dict[str, Any]):
        """Publish workflow state update"""
        if not self.producer or not self.running:
            logger.warning("Kafka producer not available, skipping state update")
            return

        message = {
            'execution_id': execution_id,
            'status': status,
            'data': data,
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'fastapi'
        }

        try:
            await self.producer.send_and_wait(self.topics['state_updates'], message)
            logger.debug(f"Published state update for execution {execution_id}")

        except Exception as e:
            logger.error(f"Failed to publish state update: {e}")
            # Don't raise - state updates are not critical

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message types"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type}")

    async def start_consuming(self):
        """Start consuming messages from all subscribed topics"""
        if not self.running or not self.consumers:
            logger.warning("Kafka service not properly started or no consumers available")
            return

        # Start consumer tasks
        tasks = []

        if 'ai_execution' in self.consumers:
            tasks.append(asyncio.create_task(
                self._consume_ai_execution_messages()
            ))

        if 'coordination' in self.consumers:
            tasks.append(asyncio.create_task(
                self._consume_coordination_messages()
            ))

        if not tasks:
            logger.warning("No consumer tasks to start")
            return

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in consumer tasks: {e}")

    async def _consume_ai_execution_messages(self):
        """Consume AI execution messages"""
        consumer = self.consumers.get('ai_execution')
        if not consumer:
            logger.error("AI execution consumer not available")
            return

        try:
            async for message in consumer:
                try:
                    execution_msg = NodeExecutionMessage(**message.value)

                    # Route to appropriate handler
                    handler = self.message_handlers.get('ai_execution')
                    if handler:
                        await handler(execution_msg)
                    else:
                        logger.warning("No handler registered for ai_execution messages")

                except Exception as e:
                    logger.error(f"Error processing AI execution message: {e}")
                    # Could implement dead letter queue here

        except Exception as e:
            logger.error(f"Error in AI execution consumer: {e}")

    async def _consume_coordination_messages(self):
        """Consume coordination messages"""
        consumer = self.consumers.get('coordination')
        if not consumer:
            logger.error("Coordination consumer not available")
            return

        try:
            async for message in consumer:
                try:
                    coord_msg = message.value

                    handler = self.message_handlers.get('coordination')
                    if handler:
                        await handler(coord_msg)
                    else:
                        logger.warning("No handler registered for coordination messages")

                except Exception as e:
                    logger.error(f"Error processing coordination message: {e}")

        except Exception as e:
            logger.error(f"Error in coordination consumer: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check Kafka service health - FIXED VERSION"""
        health = {
            'service': 'kafka',
            'status': 'healthy' if self.running else 'unhealthy',
            'producer_connected': self.producer is not None,
            'consumers_count': len(self.consumers),
            'timestamp': datetime.utcnow().isoformat()
        }

        if self.producer and self.running:
            try:
                # FIXED: Test producer by attempting to send to a test topic
                # This is more reliable than trying to get metadata
                test_topic = 'health-check-topic'
                test_message = {
                    'test': True,
                    'timestamp': datetime.utcnow().isoformat()
                }

                # Try to send a test message with a short timeout
                await asyncio.wait_for(
                    self.producer.send(test_topic, test_message),
                    timeout=5.0
                )

                health['status'] = 'healthy'
                health['test_message_sent'] = True

            except asyncio.TimeoutError:
                health['status'] = 'degraded'
                health['error'] = 'Timeout sending test message'
            except Exception as e:
                health['status'] = 'degraded'
                health['error'] = str(e)

        return health


# Global instance
kafka_service = KafkaService()