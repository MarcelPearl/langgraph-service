from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid


class WorkflowBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
    description: Optional[str] = Field(None, max_length=1000, description="Workflow description")


class WorkflowCreate(WorkflowBase):
    definition: Dict[str, Any] = Field(..., description="Workflow definition in JSON format")
    ai_config: Optional[Dict[str, Any]] = Field(default={}, description="AI configuration settings")


class WorkflowUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    definition: Optional[Dict[str, Any]] = None
    ai_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class WorkflowResponse(WorkflowBase):
    id: uuid.UUID
    version: int
    is_active: bool
    definition: Dict[str, Any]
    ai_config: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime]
    created_by: Optional[uuid.UUID]

    class Config:
        from_attributes = True


class WorkflowExecutionCreate(BaseModel):
    input_data: Dict[str, Any] = Field(..., description="Input data for workflow execution")


class WorkflowExecutionResponse(BaseModel):
    id: uuid.UUID
    workflow_id: uuid.UUID
    status: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error_data: Optional[Dict[str, Any]]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    current_step: Optional[str]
    steps_completed: int
    total_steps: Optional[int]
    ai_tokens_used: int
    ai_cost_estimate: Optional[str]

    class Config:
        from_attributes = True


class AIToolSchema(BaseModel):
    name: str
    description: str
    tool_type: str
    config: Dict[str, Any]
    schema_definition: Dict[str, Any]
    is_active: bool = True


class ExecutionStatus(BaseModel):
    execution_id: uuid.UUID
    status: str
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
