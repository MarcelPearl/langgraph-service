from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum
import uuid


class EventType(str, Enum):
    """Workflow event types"""
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_UPDATED = "workflow.updated"
    WORKFLOW_DELETED = "workflow.deleted"

    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_CANCELLED = "execution.cancelled"
    EXECUTION_RETRY = "execution.retry"

    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    NODE_SKIPPED = "node.skipped"

    AI_REQUEST_STARTED = "ai.request.started"
    AI_REQUEST_COMPLETED = "ai.request.completed"
    AI_REQUEST_FAILED = "ai.request.failed"

    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"

    CHECKPOINT_CREATED = "checkpoint.created"
    CHECKPOINT_RESTORED = "checkpoint.restored"


class EventSeverity(str, Enum):
    """Event severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BaseEventPayload(BaseModel):
    """Base event payload"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    severity: EventSeverity = EventSeverity.INFO
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str = "workflow-automation"
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None


class WorkflowEventPayload(BaseEventPayload):
    """Workflow-related events"""
    workflow_id: str
    workflow_name: str
    workflow_version: Optional[int] = None
    workflow_definition: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionEventPayload(BaseEventPayload):
    """Execution-related events"""
    execution_id: str
    workflow_id: str
    execution_status: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    current_step: Optional[str] = None
    steps_completed: int = 0
    total_steps: Optional[int] = None
    tokens_used: int = 0
    execution_time_ms: Optional[int] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeEventPayload(BaseEventPayload):
    """Node execution events"""
    execution_id: str
    workflow_id: str
    node_id: str
    node_type: str
    node_name: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AIRequestEventPayload(BaseEventPayload):
    """AI request events"""
    execution_id: str
    workflow_id: str
    ai_provider: str
    ai_model: str
    request_id: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    response_time_ms: Optional[int] = None
    cost_estimate: Optional[float] = None
    error_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolCallEventPayload(BaseEventPayload):
    """Tool call events"""
    execution_id: str
    workflow_id: str
    tool_name: str
    tool_id: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    execution_time_ms: Optional[int] = None
    error_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CheckpointEventPayload(BaseEventPayload):
    """Checkpoint events"""
    execution_id: str
    workflow_id: str
    checkpoint_id: str
    checkpoint_data: Optional[Dict[str, Any]] = None
    state_size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Union type for all event payloads
EventPayload = Union[
    WorkflowEventPayload,
    ExecutionEventPayload,
    NodeEventPayload,
    AIRequestEventPayload,
    ToolCallEventPayload,
    CheckpointEventPayload
]


class KafkaMessage(BaseModel):
    """Kafka message wrapper"""
    partition: Optional[int] = None
    key: Optional[str] = None
    value: EventPayload
    headers: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EventBatch(BaseModel):
    """Batch of events for bulk processing"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    events: List[EventPayload]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Event factory functions
def create_workflow_event(
        event_type: EventType,
        workflow_id: str,
        workflow_name: str,
        **kwargs
) -> WorkflowEventPayload:
    """Create workflow event"""
    return WorkflowEventPayload(
        event_type=event_type,
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        **kwargs
    )


def create_execution_event(
        event_type: EventType,
        execution_id: str,
        workflow_id: str,
        execution_status: str,
        **kwargs
) -> ExecutionEventPayload:
    """Create execution event"""
    return ExecutionEventPayload(
        event_type=event_type,
        execution_id=execution_id,
        workflow_id=workflow_id,
        execution_status=execution_status,
        **kwargs
    )


def create_node_event(
        event_type: EventType,
        execution_id: str,
        workflow_id: str,
        node_id: str,
        node_type: str,
        **kwargs
) -> NodeEventPayload:
    """Create node event"""
    return NodeEventPayload(
        event_type=event_type,
        execution_id=execution_id,
        workflow_id=workflow_id,
        node_id=node_id,
        node_type=node_type,
        **kwargs
    )


def create_ai_request_event(
        event_type: EventType,
        execution_id: str,
        workflow_id: str,
        ai_provider: str,
        ai_model: str,
        **kwargs
) -> AIRequestEventPayload:
    """Create AI request event"""
    return AIRequestEventPayload(
        event_type=event_type,
        execution_id=execution_id,
        workflow_id=workflow_id,
        ai_provider=ai_provider,
        ai_model=ai_model,
        **kwargs
    )


def create_tool_call_event(
        event_type: EventType,
        execution_id: str,
        workflow_id: str,
        tool_name: str,
        **kwargs
) -> ToolCallEventPayload:
    """Create tool call event"""
    return ToolCallEventPayload(
        event_type=event_type,
        execution_id=execution_id,
        workflow_id=workflow_id,
        tool_name=tool_name,
        **kwargs
    )


def create_checkpoint_event(
        event_type: EventType,
        execution_id: str,
        workflow_id: str,
        checkpoint_id: str,
        **kwargs
) -> CheckpointEventPayload:
    """Create checkpoint event"""
    return CheckpointEventPayload(
        event_type=event_type,
        execution_id=execution_id,
        workflow_id=workflow_id,
        checkpoint_id=checkpoint_id,
        **kwargs
    )