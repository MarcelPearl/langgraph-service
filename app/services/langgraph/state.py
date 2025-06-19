from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
import uuid
from datetime import datetime


class WorkflowState(TypedDict):
    """Core state for LangGraph workflows"""
    execution_id: str
    workflow_id: str
    user_id: Optional[str]

    messages: Annotated[List[BaseMessage], add_messages]

    current_step: Optional[str]
    step_count: int
    max_steps: int

    input_data: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    final_output: Optional[Dict[str, Any]]

    error_state: Optional[Dict[str, Any]]
    retry_count: int
    max_retries: int

    ai_model: str
    ai_provider: str
    ai_config: Dict[str, Any]
    tokens_used: int

    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    started_at: str
    last_updated: str
    execution_metadata: Dict[str, Any]


class StateManager:
    """Manages workflow state persistence and updates"""

    def __init__(self, db_session):
        self.db_session = db_session

    def create_initial_state(
            self,
            execution_id: str,
            workflow_id: str,
            input_data: Dict[str, Any],
            ai_config: Dict[str, Any],
            user_id: Optional[str] = None
    ) -> WorkflowState:
        """Create initial workflow state"""

        now = datetime.utcnow().isoformat()

        system_msg = SystemMessage(
            content=f"""You are an AI assistant executing workflow {workflow_id}.
            Current execution ID: {execution_id}
            Input data: {input_data}

            Follow the workflow steps carefully and provide detailed responses.
            Use available tools when needed and maintain conversation context."""
        )

        return WorkflowState(
            execution_id=execution_id,
            workflow_id=workflow_id,
            user_id=user_id,
            messages=[system_msg],
            current_step=None,
            step_count=0,
            max_steps=ai_config.get("max_steps", 50),
            input_data=input_data,
            intermediate_results={},
            final_output=None,
            error_state=None,
            retry_count=0,
            max_retries=ai_config.get("max_retries", 3),
            ai_model=ai_config.get("model", "gpt-4"),
            ai_provider=ai_config.get("provider", "openai"),
            ai_config=ai_config,
            tokens_used=0,
            tool_calls=[],
            tool_results=[],
            started_at=now,
            last_updated=now,
            execution_metadata={}
        )

    def update_state(
            self,
            state: WorkflowState,
            updates: Dict[str, Any]
    ) -> WorkflowState:
        """Update workflow state with new data"""

        updates["last_updated"] = datetime.utcnow().isoformat()

        if "current_step" in updates and updates["current_step"] != state["current_step"]:
            updates["step_count"] = state["step_count"] + 1

        for key, value in updates.items():
            if key in state:
                state[key] = value

        return state

    def add_message(
            self,
            state: WorkflowState,
            message: BaseMessage
    ) -> WorkflowState:
        """Add message to state"""
        state["messages"].append(message)
        state["last_updated"] = datetime.utcnow().isoformat()
        return state

    def handle_error(
            self,
            state: WorkflowState,
            error: Exception,
            step: str
    ) -> WorkflowState:
        """Handle error in workflow execution"""

        error_data = {
            "step": step,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "retry_count": state["retry_count"]
        }

        state["error_state"] = error_data
        state["retry_count"] += 1
        state["last_updated"] = datetime.utcnow().isoformat()

        return state

    def is_max_retries_reached(self, state: WorkflowState) -> bool:
        """Check if maximum retries reached"""
        return state["retry_count"] >= state["max_retries"]

    def is_max_steps_reached(self, state: WorkflowState) -> bool:
        """Check if maximum steps reached"""
        return state["step_count"] >= state["max_steps"]