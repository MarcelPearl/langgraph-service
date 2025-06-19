import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.workflow import Workflow, WorkflowExecution
from app.schemas.workflow import WorkflowResponse, WorkflowExecutionResponse, WorkflowExecutionCreate
from app.services.ai.model_factory import model_factory
from app.services.ai.huggingface_service import hf_api
from app.schemas.ai import (
    AIModelRequest, AIModelResponse,
    HuggingFaceModelInfo, AvailableProvidersResponse
)
import logging

from app.services.langgraph.workflow import BasicWorkflowEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])


@router.get("/providers", response_model=AvailableProvidersResponse)
async def get_available_providers():
    """Get a list of available AI providers"""
    try:
        providers = model_factory.get_available_providers()
        hf_models = model_factory.get_popular_huggingface_models()

        return {
            "providers": providers,
            "huggingface_models": hf_models,
            "total_providers": len(providers)
        }
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=AIModelResponse)
async def generate_ai_response(request: AIModelRequest):
    """Generate AI response using specified model"""
    try:
        if request.provider.lower() == "huggingface":
            result = await hf_api.generate_text(
                model_name=request.model,
                prompt=request.prompt,
                parameters={
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "top_p": request.top_p,
                    **request.additional_params
                }
            )
            return AIModelResponse(
                response=result["text"],
                model=request.model,
                provider="huggingface",
                metadata={"usage": result}
            )
        else:
            model = model_factory.get_model(
                provider=request.provider,
                model_name=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                **request.additional_params
            )

            response = await model.ainvoke(request.prompt)

            return AIModelResponse(
                response=response.content,
                model=request.model,
                provider=request.provider,
                metadata={"usage": getattr(response, "usage_metadata", {})}
            )

    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/huggingface/models", response_model=List[Dict[str, Any]])
async def list_huggingface_models(
        task: Optional[str] = None,
        limit: int = 20
):
    """List available Hugging Face models"""
    try:
        models = await hf_api.list_available_models(task=task)
        return models[:limit]
    except Exception as e:
        logger.error(f"Error listing HF models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/huggingface/models/{model_name}/info", response_model=HuggingFaceModelInfo)
async def get_huggingface_model_info(model_name: str):
    """Get information about a specific Hugging Face model"""
    try:
        model_id = model_name.replace("--", "/")
        info = await hf_api.get_model_info(model_id)

        return HuggingFaceModelInfo(
            model_id=model_id,
            info=info,
            available=not info.get("error"),
            error=info.get("error")
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-free-models")
async def test_free_models():
    """Test various free models to check availability"""
    try:
        results = {}
        test_prompt = "Hello, how are you today?"

        free_models = [
            "gpt2",
            "distilgpt2",
            "microsoft/DialoGPT-medium",
            "google/flan-t5-base"
        ]

        for model_name in free_models:
            try:
                result = await hf_api.generate_text(
                    model_name=model_name,
                    prompt=test_prompt,
                    parameters={"max_tokens": 50}
                )
                results[model_name] = {
                    "status": "success",
                    "response": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                }
            except Exception as model_error:
                results[model_name] = {
                    "status": "error",
                    "error": str(model_error)
                }

            await hf_api.rate_limiter.wait_if_needed()

        return results

    except Exception as e:
        logger.error(f"Error testing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(
        skip: int = 0,
        limit: int = 100,
        db: AsyncSession = Depends(get_db)
):
    """List all workflows"""
    try:
        result = await db.execute(
            select(Workflow)
            .where(Workflow.is_active == True)
            .offset(skip)
            .limit(limit)
            .order_by(Workflow.created_at.desc())
        )
        workflows = result.scalars().all()
        return workflows

    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
        workflow_id: str,
        db: AsyncSession = Depends(get_db)
):
    """Get workflow by ID"""
    try:
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return workflow

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
        workflow_id: str,
        execution_data: WorkflowExecutionCreate,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db)
):
    """Execute a workflow"""
    try:
        # Get workflow
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            input_data=execution_data.input_data,
            status="pending"
        )

        db.add(execution)
        await db.commit()
        await db.refresh(execution)

        # Start background execution
        background_tasks.add_task(
            execute_workflow_background,
            str(execution.id),
            workflow_id,
            execution_data.input_data,
            workflow.ai_config,
            db
        )

        logger.info(f"Started workflow execution: {execution.id}")
        return execution

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error starting workflow execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_workflow_background(
        execution_id: str,
        workflow_id: str,
        input_data: Dict[str, Any],
        ai_config: Dict[str, Any],
        db: AsyncSession
):
    """Background task for workflow execution"""
    try:
        # Update execution status
        await db.execute(
            update(WorkflowExecution)
            .where(WorkflowExecution.id == execution_id)
            .values(status="running", started_at=datetime.utcnow())
        )
        await db.commit()

        # Execute workflow
        engine = BasicWorkflowEngine(db)
        result = await engine.execute_workflow(
            workflow_id=workflow_id,
            execution_id=execution_id,
            input_data=input_data,
            ai_config=ai_config
        )

        # Update execution with results
        update_data = {
            "status": result["status"],
            "completed_at": datetime.utcnow(),
            "output_data": result.get("output"),
            "error_data": result.get("error")
        }

        if result.get("metadata"):
            update_data.update({
                "ai_tokens_used": result["metadata"].get("tokens_used", 0),
                "steps_completed": result["metadata"].get("steps_completed", 0)
            })

        await db.execute(
            update(WorkflowExecution)
            .where(WorkflowExecution.id == execution_id)
            .values(**update_data)
        )
        await db.commit()

        logger.info(f"Workflow execution completed: {execution_id}")

    except Exception as e:
        logger.error(f"Background execution failed for {execution_id}: {e}")

        # Update execution with error
        await db.execute(
            update(WorkflowExecution)
            .where(WorkflowExecution.id == execution_id)
            .values(
                status="failed",
                completed_at=datetime.utcnow(),
                error_data={
                    "type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        )
        await db.commit()


@router.get("/{workflow_id}/executions", response_model=List[WorkflowExecutionResponse])
async def list_workflow_executions(
        workflow_id: str,
        skip: int = 0,
        limit: int = 100,
        db: AsyncSession = Depends(get_db)
):
    """List workflow executions"""
    try:
        result = await db.execute(
            select(WorkflowExecution)
            .where(WorkflowExecution.workflow_id == workflow_id)
            .offset(skip)
            .limit(limit)
            .order_by(WorkflowExecution.created_at.desc())
        )
        executions = result.scalars().all()
        return executions

    except Exception as e:
        logger.error(f"Error listing executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_execution(
        execution_id: str,
        db: AsyncSession = Depends(get_db)
):
    """Get execution by ID"""
    try:
        result = await db.execute(
            select(WorkflowExecution)
            .where(WorkflowExecution.id == execution_id)
        )
        execution = result.scalar_one_or_none()

        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")

        return execution

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))