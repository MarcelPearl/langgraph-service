
"""
Kafka services for event-driven workflow automation

This module provides:
- Event publishing for workflow events
- Event consumption and processing
- Event-driven architecture patterns
- Integration with workflow execution engine
"""

from .producer import event_producer, sync_event_producer
from .consumer import WorkflowEventConsumer, EventProcessor
from .admin import kafka_admin, dlq_manager

__all__ = [
    'event_producer',
    'sync_event_producer',
    'WorkflowEventConsumer',
    'EventProcessor',
    'kafka_admin',
    'dlq_manager'
]