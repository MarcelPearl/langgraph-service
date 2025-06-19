from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    definition = Column(JSONB, nullable=False)  # LangGraph workflow definition
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Fix: Use default instead of server_default for better compatibility
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=datetime.utcnow, nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    ai_config = Column(JSONB, default={}, nullable=False)  # Model preferences, etc.

    executions = relationship("WorkflowExecution", back_populates="workflow")


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False)

    status = Column(String(50), default="pending", nullable=False, index=True)
    # Status: pending, running, completed, failed, cancelled

    input_data = Column(JSONB, default={}, nullable=False)
    output_data = Column(JSONB, default={}, nullable=True)
    error_data = Column(JSONB, default={}, nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    current_step = Column(String(255), nullable=True)
    steps_completed = Column(Integer, default=0, nullable=False)
    total_steps = Column(Integer, nullable=True)

    ai_tokens_used = Column(Integer, default=0, nullable=False)
    ai_cost_estimate = Column(String(50), nullable=True)  # In USD cents

    workflow = relationship("Workflow", back_populates="executions")
    checkpoints = relationship("ExecutionCheckpoint", back_populates="execution")


class ExecutionCheckpoint(Base):
    __tablename__ = "execution_checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id = Column(UUID(as_uuid=True), ForeignKey("workflow_executions.id"), nullable=False)

    checkpoint_id = Column(String(255), nullable=False, index=True)
    state_data = Column(JSONB, nullable=False)
    checkpoint_metadata = Column(JSONB, default={}, nullable=False)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    execution = relationship("WorkflowExecution", back_populates="checkpoints")


class AITool(Base):
    __tablename__ = "ai_tools"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    tool_type = Column(String(100), nullable=False)

    config = Column(JSONB, default={}, nullable=False)
    schema_definition = Column(JSONB, nullable=False)

    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=datetime.utcnow, nullable=True)