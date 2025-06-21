from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class EventType(str, Enum):
    """Types of events in the workflow system"""
    # Execution events
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_CANCELLED = "execution_cancelled"

    # Node events
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_RETRY = "node_retry"

    # AI events
    AI_REQUEST_STARTED = "ai_request_started"
    AI_REQUEST_COMPLETED = "ai_request_completed"
    AI_REQUEST_FAILED = "ai_request_failed"

    # Tool events
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"

    # System events
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"


class BaseEvent(BaseModel):
    """Base event structure"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    service: str = "fastapi"
    execution_id: Optional[str] = None
    workflow_id: Optional[str] = None


class ExecutionEvent(BaseEvent):
    """Execution lifecycle events"""
    execution_status: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    steps_completed: Optional[int] = None
    tokens_used: Optional[int] = None
    execution_time_ms: Optional[int] = None


class NodeEvent(BaseEvent):
    """Node execution events"""
    node_id: str
    node_type: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None
    retry_count: Optional[int] = None


class AIRequestEvent(BaseEvent):
    """AI request events"""
    ai_provider: str
    ai_model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    response_time_ms: Optional[int] = None
    error_data: Optional[Dict[str, Any]] = None


class ToolCallEvent(BaseEvent):
    """Tool call events"""
    tool_name: str
    tool_id: str
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    execution_time_ms: Optional[int] = None
    error_data: Optional[Dict[str, Any]] = None


# Event creation helper functions
def create_execution_event(
        event_type: EventType,
        execution_id: str,
        workflow_id: str,
        execution_status: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        steps_completed: Optional[int] = None,
        tokens_used: Optional[int] = None,
        execution_time_ms: Optional[int] = None
) -> ExecutionEvent:
    """Create an execution event"""
    return ExecutionEvent(
        event_type=event_type,
        execution_id=execution_id,
        workflow_id=workflow_id,
        execution_status=execution_status,
        input_data=input_data,
        output_data=output_data,
        error_data=error_data,
        user_id=user_id,
        steps_completed=steps_completed,
        tokens_used=tokens_used,
        execution_time_ms=execution_time_ms
    )


def create_node_event(
        event_type: EventType,
        execution_id: str,
        workflow_id: str,
        node_id: str,
        node_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_data: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[int] = None,
        retry_count: Optional[int] = None
) -> NodeEvent:
    """Create a node event"""
    return NodeEvent(
        event_type=event_type,
        execution_id=execution_id,
        workflow_id=workflow_id,
        node_id=node_id,
        node_type=node_type,
        input_data=input_data,
        output_data=output_data,
        error_data=error_data,
        execution_time_ms=execution_time_ms,
        retry_count=retry_count
    )


def create_ai_request_event(
        event_type: EventType,
        execution_id: str,
        workflow_id: str,
        ai_provider: str,
        ai_model: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        error_data: Optional[Dict[str, Any]] = None
) -> AIRequestEvent:
    """Create an AI request event"""
    return AIRequestEvent(
        event_type=event_type,
        execution_id=execution_id,
        workflow_id=workflow_id,
        ai_provider=ai_provider,
        ai_model=ai_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        response_time_ms=response_time_ms,
        error_data=error_data
    )


def create_tool_call_event(
        event_type: EventType,
        execution_id: str,
        workflow_id: str,
        tool_name: str,
        tool_id: str,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Any] = None,
        execution_time_ms: Optional[int] = None,
        error_data: Optional[Dict[str, Any]] = None
) -> ToolCallEvent:
    """Create a tool call event"""
    return ToolCallEvent(
        event_type=event_type,
        execution_id=execution_id,
        workflow_id=workflow_id,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_args=tool_args,
        tool_result=tool_result,
        execution_time_ms=execution_time_ms,
        error_data=error_data
    )