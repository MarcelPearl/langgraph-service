"""
Kafka Integration Tests

Tests for Kafka event publishing, consumption, and processing
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

# Import test utilities
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.kafka.producer import WorkflowEventProducer
from app.services.kafka.consumer import WorkflowEventConsumer
from app.services.kafka.admin import KafkaAdministrator
from app.schemas.events import (
    EventType, create_execution_event, create_node_event,
    create_ai_request_event, create_tool_call_event
)


class TestKafkaProducer:
    """Test Kafka event producer"""

    @pytest.fixture
    async def producer(self):
        """Create test producer"""
        producer = WorkflowEventProducer()
        # Mock the actual Kafka producer for testing
        producer.producer = AsyncMock()
        producer.is_started = True
        yield producer
        # Cleanup
        producer.is_started = False

    @pytest.mark.asyncio
    async def test_publish_execution_event(self, producer):
        """Test publishing execution event"""
        event = create_execution_event(
            event_type=EventType.EXECUTION_STARTED,
            execution_id=str(uuid.uuid4()),
            workflow_id=str(uuid.uuid4()),
            execution_status="running",
            input_data={"query": "test query"}
        )

        # Mock successful publish
        future_mock = AsyncMock()
        future_mock.get.return_value = AsyncMock(topic="execution-events", partition=0, offset=123)
        producer.producer.send.return_value = future_mock

        result = await producer.publish_event(event)

        assert result is True
        producer.producer.send.assert_called_once()

        # Verify call arguments
        call_args = producer.producer.send.call_args
        assert call_args[1]['topic'] == 'execution-events'
        assert call_args[1]['key'] == event.execution_id
        assert 'value' in call_args[1]

    @pytest.mark.asyncio
    async def test_publish_batch_events(self, producer):
        """Test batch event publishing"""
        events = [
            create_execution_event(
                event_type=EventType.EXECUTION_STARTED,
                execution_id=str(uuid.uuid4()),
                workflow_id=str(uuid.uuid4()),
                execution_status="running"
            ) for _ in range(5)
        ]

        # Mock successful publishes
        future_mock = AsyncMock()
        future_mock.get.return_value = AsyncMock(topic="execution-events", partition=0, offset=123)
        producer.producer.send.return_value = future_mock

        results = await producer.publish_batch(events)

        assert results['success'] == 5
        assert results['failed'] == 0
        assert producer.producer.send.call_count == 5

    @pytest.mark.asyncio
    async def test_publish_event_failure(self, producer):
        """Test handling of publish failures"""
        event = create_execution_event(
            event_type=EventType.EXECUTION_FAILED,
            execution_id=str(uuid.uuid4()),
            workflow_id=str(uuid.uuid4()),
            execution_status="failed"
        )

        # Mock Kafka error
        from kafka.errors import KafkaTimeoutError
        producer.producer.send.side_effect = KafkaTimeoutError("Timeout")

        # Mock dead letter queue send
        dlq_mock = AsyncMock()
        producer.producer.send = dlq_mock

        result = await producer.publish_event(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_topic_routing(self, producer):
        """Test correct topic routing based on event type"""
        test_cases = [
            (EventType.EXECUTION_STARTED, "execution-events"),
            (EventType.WORKFLOW_CREATED, "workflow-events"),
            (EventType.NODE_COMPLETED, "node-events"),
        ]

        for event_type, expected_topic in test_cases:
            topic = producer._get_topic_for_event_type(event_type)
            assert topic == expected_topic


class TestKafkaConsumer:
    """Test Kafka event consumer"""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session factory"""
        session_mock = AsyncMock()
        session_mock.execute.return_value = None
        session_mock.commit.return_value = None
        session_mock.rollback.return_value = None

        async def mock_session_factory():
            return session_mock

        return mock_session_factory

    @pytest.fixture
    def consumer(self, mock_db_session):
        """Create test consumer"""
        consumer = WorkflowEventConsumer(mock_db_session)
        consumer.consumer = AsyncMock()
        consumer.is_running = True
        return consumer

    @pytest.mark.asyncio
    async def test_event_handler_registration(self, consumer):
        """Test event handler registration"""
        test_handler = AsyncMock()

        consumer.register_handler(EventType.EXECUTION_COMPLETED, test_handler)

        assert EventType.EXECUTION_COMPLETED in consumer._handlers
        assert test_handler in consumer._handlers[EventType.EXECUTION_COMPLETED]

    @pytest.mark.asyncio
    async def test_execution_started_handler(self, consumer):
        """Test execution started event handler"""
        event_payload = create_execution_event(
            event_type=EventType.EXECUTION_STARTED,
            execution_id=str(uuid.uuid4()),
            workflow_id=str(uuid.uuid4()),
            execution_status="running"
        )

        # Mock message
        message_mock = AsyncMock()
        message_mock.topic = "execution-events"
        message_mock.partition = 0
        message_mock.offset = 123

        # Call handler
        await consumer._handle_execution_started(event_payload, message_mock)

        # Verify database update was attempted
        # Note: In real test, you'd verify the actual SQL execution

    @pytest.mark.asyncio
    async def test_message_processing(self, consumer):
        """Test message processing pipeline"""
        # Create mock message
        message_mock = AsyncMock()
        message_mock.topic = "execution-events"
        message_mock.partition = 0
        message_mock.offset = 123
        message_mock.key = b"test-execution-id"
        message_mock.value = {
            "event_type": "execution.started",
            "event_id": str(uuid.uuid4()),
            "execution_id": "test-execution-id",
            "workflow_id": "test-workflow-id",
            "execution_status": "running",
            "timestamp": datetime.utcnow().isoformat()
        }
        message_mock.timestamp = datetime.utcnow().timestamp()

        # Mock commit
        consumer._commit_message = AsyncMock()
        consumer._handle_event = AsyncMock()

        # Process message
        await consumer._process_message(message_mock)

        # Verify event handling was called
        consumer._handle_event.assert_called_once()
        consumer._commit_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, consumer):
        """Test error handling in message processing"""
        # Create mock message that will cause an error
        message_mock = AsyncMock()
        message_mock.topic = "execution-events"
        message_mock.value = {"invalid": "data"}  # Missing required fields

        # Mock error handlers
        consumer._handle_processing_error = AsyncMock()

        # Process message (should handle error gracefully)
        await consumer._process_message(message_mock)

        # Verify error handler was called
        consumer._handle_processing_error.assert_called_once()


class TestKafkaAdmin:
    """Test Kafka administration"""

    @pytest.fixture
    def admin(self):
        """Create test admin"""
        admin = KafkaAdministrator()
        admin.admin_client = AsyncMock()
        admin.sync_admin_client = AsyncMock()
        return admin

    @pytest.mark.asyncio
    async def test_topic_creation(self, admin):
        """Test topic creation"""
        # Mock successful topic creation
        admin.sync_admin_client.create_topics.return_value = {
            'workflow-events': AsyncMock()
        }

        # Mock future result
        future_mock = AsyncMock()
        future_mock.result.return_value = None
        admin.sync_admin_client.create_topics.return_value = {
            'workflow-events': future_mock
        }

        results = await admin.create_topics()

        # Verify topics were attempted to be created
        admin.sync_admin_client.create_topics.assert_called_once()

    @pytest.mark.asyncio
    async def test_cluster_health_monitoring(self, admin):
        """Test cluster health monitoring"""
        # Mock cluster metadata
        admin.admin_client.describe_cluster.return_value = AsyncMock(
            cluster_id="test-cluster",
            controller=AsyncMock(id=1),
            brokers=[AsyncMock(id=1), AsyncMock(id=2), AsyncMock(id=3)]
        )

        admin.admin_client.list_topics.return_value = [
            'workflow-events', 'execution-events', 'node-events'
        ]

        # Mock topic descriptions
        admin.admin_client.describe_topics.return_value = {
            'workflow-events': AsyncMock(
                partitions=[
                    AsyncMock(replicas=[1, 2, 3])
                ]
            )
        }

        health = await admin.monitor_cluster_health()

        assert 'overall_status' in health
        assert 'cluster_info' in health
        assert 'topic_health' in health


class TestEventIntegration:
    """Test end-to-end event flow"""

    @pytest.mark.asyncio
    async def test_workflow_execution_event_flow(self):
        """Test complete event flow for workflow execution"""
        execution_id = str(uuid.uuid4())
        workflow_id = str(uuid.uuid4())

        # Mock producer and consumer
        producer = AsyncMock()
        consumer = AsyncMock()

        # Simulate workflow execution with events
        events = [
            create_execution_event(
                event_type=EventType.EXECUTION_STARTED,
                execution_id=execution_id,
                workflow_id=workflow_id,
                execution_status="running"
            ),
            create_node_event(
                event_type=EventType.NODE_STARTED,
                execution_id=execution_id,
                workflow_id=workflow_id,
                node_id="process_input",
                node_type="input_processor"
            ),
            create_ai_request_event(
                event_type=EventType.AI_REQUEST_STARTED,
                execution_id=execution_id,
                workflow_id=workflow_id,
                ai_provider="openai",
                ai_model="gpt-4"
            ),
            create_execution_event(
                event_type=EventType.EXECUTION_COMPLETED,
                execution_id=execution_id,
                workflow_id=workflow_id,
                execution_status="completed"
            )
        ]

        # Verify event ordering and content
        assert events[0].event_type == EventType.EXECUTION_STARTED
        assert events[-1].event_type == EventType.EXECUTION_COMPLETED
        assert all(event.execution_id == execution_id for event in events)
        assert all(event.workflow_id == workflow_id for event in events)

    @pytest.mark.asyncio
    async def test_ai_request_with_tools_event_flow(self):
        """Test event flow for AI request with tool calls"""
        execution_id = str(uuid.uuid4())
        workflow_id = str(uuid.uuid4())

        events = [
            create_ai_request_event(
                event_type=EventType.AI_REQUEST_STARTED,
                execution_id=execution_id,
                workflow_id=workflow_id,
                ai_provider="openai",
                ai_model="gpt-4"
            ),
            create_tool_call_event(
                event_type=EventType.TOOL_CALL_STARTED,
                execution_id=execution_id,
                workflow_id=workflow_id,
                tool_name="calculator",
                tool_args={"expression": "2 + 2"}
            ),
            create_tool_call_event(
                event_type=EventType.TOOL_CALL_COMPLETED,
                execution_id=execution_id,
                workflow_id=workflow_id,
                tool_name="calculator",
                tool_result="4"
            ),
            create_ai_request_event(
                event_type=EventType.AI_REQUEST_COMPLETED,
                execution_id=execution_id,
                workflow_id=workflow_id,
                ai_provider="openai",
                ai_model="gpt-4",
                total_tokens=150
            )
        ]

        # Verify tool call flow
        tool_start = next(e for e in events if e.event_type == EventType.TOOL_CALL_STARTED)
        tool_end = next(e for e in events if e.event_type == EventType.TOOL_CALL_COMPLETED)

        assert tool_start.tool_name == tool_end.tool_name
        assert tool_end.tool_result == "4"


# Test utilities and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_kafka_config():
    """Mock Kafka configuration for testing"""
    with patch('app.core.kafka_config.kafka_settings') as mock_settings:
        mock_settings.KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
        mock_settings.KAFKA_WORKFLOW_EVENTS_TOPIC = 'workflow-events'
        mock_settings.KAFKA_EXECUTION_EVENTS_TOPIC = 'execution-events'
        mock_settings.KAFKA_NODE_EVENTS_TOPIC = 'node-events'
        mock_settings.KAFKA_DEAD_LETTER_TOPIC = 'dead-letter-queue'
        yield mock_settings


# Performance tests
class TestKafkaPerformance:
    """Performance tests for Kafka operations"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_batch_publish_performance(self):
        """Test batch publishing performance"""
        import time

        # Create large batch of events
        events = [
            create_execution_event(
                event_type=EventType.EXECUTION_COMPLETED,
                execution_id=str(uuid.uuid4()),
                workflow_id=str(uuid.uuid4()),
                execution_status="completed"
            ) for _ in range(1000)
        ]

        # Mock producer
        producer = WorkflowEventProducer()
        producer.producer = AsyncMock()
        producer.is_started = True

        # Mock successful publish
        future_mock = AsyncMock()
        future_mock.get.return_value = AsyncMock(topic="execution-events")
        producer.producer.send.return_value = future_mock

        # Measure batch publish time
        start_time = time.time()
        results = await producer.publish_batch(events)
        end_time = time.time()

        publish_time = end_time - start_time

        assert results['success'] == 1000
        assert publish_time < 5.0  # Should complete within 5 seconds

        # Calculate throughput
        throughput = len(events) / publish_time
        assert throughput > 200  # Should handle at least 200 events/second


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])