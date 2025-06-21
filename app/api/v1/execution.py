# app/api/v1/execution.py
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uuid
import logging
from datetime import datetime

from app.services.execution.node_engine import node_execution_engine
from app.services.context.redis_service import redis_context_service
from app.services.messaging.kafka_service import kafka_service, NodeExecutionMessage
from app.services.validation.workflow_validator import workflow_validator
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/execution", tags=["execution"])


class NodeExecutionRequest(BaseModel):
    """Request to execute a single node"""
    node_id: str = Field(..., description="Node identifier")
    node_type: str = Field(..., description="Type of node to execute")
    node_data: Dict[str, Any] = Field(..., description="Node configuration")
    context: Dict[str, Any] = Field(default={}, description="Execution context")
    execution_id: Optional[str] = Field(None, description="Execution ID (auto-generated if not provided)")
    workflow_id: Optional[str] = Field(None, description="Workflow ID")


class WorkflowExecutionRequest(BaseModel):
    """Request to execute a complete workflow"""
    workflow_definition: Dict[str, Any] = Field(..., description="Complete workflow definition")
    input_data: Dict[str, Any] = Field(default={}, description="Input data for workflow")
    ai_config: Dict[str, Any] = Field(default={}, description="AI configuration")
    execution_id: Optional[str] = Field(None, description="Execution ID (auto-generated if not provided)")


class ExecutionStatusResponse(BaseModel):
    """Response with execution status"""
    execution_id: str
    status: str
    context: Optional[Dict[str, Any]] = None
    processing_nodes: list = []
    completed_nodes: list = []
    workflow_status: Optional[Dict[str, Any]] = None
    timestamp: str


@router.post("/node/execute")
async def execute_single_node(request: NodeExecutionRequest):
    """Execute a single AI node"""
    try:
        execution_id = request.execution_id or str(uuid.uuid4())
        workflow_id = request.workflow_id or str(uuid.uuid4())

        logger.info(f"Executing single node {request.node_id} in execution {execution_id}")

        # Validate node configuration
        from app.services.validation.workflow_validator import NodeType
        try:
            node_type = NodeType(request.node_type)
            validation_result = workflow_validator.validate_node_config(node_type, request.node_data)

            if not validation_result.is_valid:
                return {
                    "execution_id": execution_id,
                    "status": "validation_failed",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported node type: {request.node_type}"
            )

        # Create execution message
        message = NodeExecutionMessage(
            execution_id=execution_id,
            workflow_id=workflow_id,
            node_id=request.node_id,
            node_type=request.node_type,
            node_data=request.node_data,
            context=request.context,
            dependencies=[],
            timestamp=datetime.utcnow().isoformat(),
            priority="high"  # Single node execution gets high priority
        )

        # Set initial context
        await redis_context_service.set_execution_context(
            execution_id=execution_id,
            context={
                "workflow_id": workflow_id,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "node_count": 1,
                "input_data": request.context
            }
        )

        # Execute directly (for single nodes, we bypass Kafka)
        await node_execution_engine.handle_execution_message(message)

        # Get results
        result = await redis_context_service.get_node_result(execution_id, request.node_id)

        return {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "node_id": request.node_id,
            "status": "completed" if result and not result.get("error") else "failed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Node execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/execute")
async def execute_workflow(request: WorkflowExecutionRequest, background_tasks: BackgroundTasks):
    """Execute a complete workflow"""
    try:
        execution_id = request.execution_id or str(uuid.uuid4())

        logger.info(f"Starting workflow execution {execution_id}")

        # Validate workflow
        validation_result = workflow_validator.validate_workflow(request.workflow_definition)

        if not validation_result.is_valid:
            return {
                "execution_id": execution_id,
                "status": "validation_failed",
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "complexity": validation_result.estimated_complexity
            }

        # Set up execution context
        workflow_context = {
            "workflow_id": request.workflow_definition.get("id", str(uuid.uuid4())),
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "input_data": request.input_data,
            "ai_config": request.ai_config,
            "node_count": validation_result.node_count,
            "estimated_complexity": validation_result.estimated_complexity
        }

        await redis_context_service.set_execution_context(execution_id, workflow_context)

        # For now, execute workflow using LangGraph for AI-heavy workflows
        if validation_result.estimated_complexity > 50:
            # Use LangGraph for complex workflows
            background_tasks.add_task(
                execute_langgraph_workflow,
                execution_id,
                request.workflow_definition,
                request.input_data,
                request.ai_config
            )
        else:
            # Use simple node-by-node execution for simple workflows
            background_tasks.add_task(
                execute_simple_workflow,
                execution_id,
                request.workflow_definition,
                request.input_data
            )

        return {
            "execution_id": execution_id,
            "status": "started",
            "workflow_id": workflow_context["workflow_id"],
            "estimated_complexity": validation_result.estimated_complexity,
            "validation_warnings": validation_result.warnings,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{execution_id}", response_model=ExecutionStatusResponse)
async def get_execution_status(execution_id: str):
    """Get execution status"""
    try:
        status = await node_execution_engine.get_execution_status(execution_id)

        return ExecutionStatusResponse(
            execution_id=execution_id,
            status=status.get("workflow_status", {}).get("status", "unknown"),
            context=status.get("context"),
            processing_nodes=status.get("processing_nodes", []),
            completed_nodes=status.get("completed_nodes", []),
            workflow_status=status.get("workflow_status"),
            timestamp=status.get("timestamp", datetime.utcnow().isoformat())
        )

    except Exception as e:
        logger.error(f"Failed to get execution status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{execution_id}")
async def get_execution_results(execution_id: str):
    """Get all results from an execution"""
    try:
        context = await redis_context_service.get_execution_context(execution_id)
        results = await redis_context_service.get_all_node_results(execution_id)

        if not context:
            raise HTTPException(status_code=404, detail="Execution not found")

        return {
            "execution_id": execution_id,
            "context": context,
            "node_results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup/{execution_id}")
async def cleanup_execution(execution_id: str):
    """Clean up execution data"""
    try:
        await redis_context_service.cleanup_execution(execution_id)

        return {
            "execution_id": execution_id,
            "status": "cleaned_up",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to cleanup execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def execute_langgraph_workflow(
        execution_id: str,
        workflow_definition: Dict[str, Any],
        input_data: Dict[str, Any],
        ai_config: Dict[str, Any]
):
    """Execute workflow using LangGraph"""
    try:
        from app.services.langgraph.workflow import BasicWorkflowEngine

        # Create workflow engine (with minimal DB session for now)
        workflow_engine = BasicWorkflowEngine(db_session=None)

        result = await workflow_engine.execute_workflow(
            workflow_id=workflow_definition.get("id", str(uuid.uuid4())),
            execution_id=execution_id,
            input_data=input_data,
            ai_config=ai_config or {"provider": "huggingface", "model": "microsoft/Phi-3-mini-4k-instruct"}
        )

        # Update final status
        await redis_context_service.set_workflow_status(
            execution_id=execution_id,
            status=result["status"],
            metadata=result.get("metadata", {})
        )

        logger.info(f"LangGraph workflow {execution_id} completed with status: {result['status']}")

    except Exception as e:
        logger.error(f"LangGraph workflow execution failed: {e}")

        await redis_context_service.set_workflow_status(
            execution_id=execution_id,
            status="failed",
            metadata={"error": str(e), "timestamp": datetime.utcnow().isoformat()}
        )


async def execute_simple_workflow(
        execution_id: str,
        workflow_definition: Dict[str, Any],
        input_data: Dict[str, Any]
):
    """Execute simple workflow node by node"""
    try:
        nodes = workflow_definition.get("nodes", [])
        ai_nodes = [node for node in nodes if node.get("type", "").startswith("ai_")]

        # For simple workflows, execute AI nodes sequentially
        for node in ai_nodes:
            message = NodeExecutionMessage(
                execution_id=execution_id,
                workflow_id=workflow_definition.get("id", str(uuid.uuid4())),
                node_id=node["id"],
                node_type=node["type"],
                node_data=node.get("config", {}),
                context=input_data,
                dependencies=[],
                timestamp=datetime.utcnow().isoformat()
            )

            await node_execution_engine.handle_execution_message(message)

        # Update final status
        await redis_context_service.set_workflow_status(
            execution_id=execution_id,
            status="completed",
            metadata={"nodes_executed": len(ai_nodes), "timestamp": datetime.utcnow().isoformat()}
        )

        logger.info(f"Simple workflow {execution_id} completed")

    except Exception as e:
        logger.error(f"Simple workflow execution failed: {e}")

        await redis_context_service.set_workflow_status(
            execution_id=execution_id,
            status="failed",
            metadata={"error": str(e), "timestamp": datetime.utcnow().isoformat()}
        )


@router.get("/test/ai-capabilities")
async def test_ai_capabilities():
    """Test AI capabilities and model availability"""
    try:
        from app.services.ai.model_factory import model_factory
        from app.services.ai.huggingface_service import hf_api

        # Test model factory
        providers = model_factory.get_available_providers()
        models = model_factory.get_popular_huggingface_models()

        # Test a simple AI generation
        test_result = await hf_api.generate_text(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            prompt="Say hello and confirm you're working correctly.",
            parameters={"max_tokens": 50, "temperature": 0.7}
        )

        return {
            "status": "success",
            "available_providers": providers,
            "available_models": list(models.keys()),
            "test_generation": test_result,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"AI capabilities test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }