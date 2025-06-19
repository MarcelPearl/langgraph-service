
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.workflow import Workflow
from app.schemas.workflow import (
    WorkflowCreate, WorkflowResponse
)

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(
        workflow_data: WorkflowCreate,
        db: AsyncSession = Depends(get_db)
):
    """Create a new workflow"""
    try:
        workflow = Workflow(
            name=workflow_data.name,
            description=workflow_data.description,
            definition=workflow_data.definition,
            ai_config=workflow_data.ai_config or {}
        )

        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)

        logger.info(f"Created workflow: {workflow.id}")
        return workflow

    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))
