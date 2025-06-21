# app/api/v1/ai.py
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from app.services.ai.model_factory import model_factory
from app.services.ai.huggingface_service import hf_api
from app.schemas.ai import (
    AIModelRequest, AIModelResponse,
    HuggingFaceModelInfo, AvailableProvidersResponse
)
import logging

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
                response_metadata={"usage": result}
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
            "microsoft/Phi-3-mini-4k-instruct",
            "microsoft/DialoGPT-medium",
            "deepset/roberta-base-squad2"
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


@router.get("/capabilities")
async def get_ai_capabilities():
    """Get AI engine capabilities and supported operations"""
    try:
        from app.core.config import settings

        capabilities = {
            "node_types": [
                {
                    "type": "ai_decision",
                    "description": "AI-powered decision making with multiple options",
                    "required_config": ["prompt", "options"]
                },
                {
                    "type": "ai_text_generator",
                    "description": "Generate text based on prompts and templates",
                    "required_config": ["prompt_template"]
                },
                {
                    "type": "ai_data_processor",
                    "description": "Process and analyze data using AI",
                    "required_config": ["operation"]
                }
            ],
            "providers": model_factory.get_available_providers(),
            "models": {
                "huggingface": list(model_factory.get_popular_huggingface_models().keys()),
                "openai": [settings.DEFAULT_OPENAI_MODEL] if getattr(settings, 'OPENAI_API_KEY', None) else [],
                "anthropic": [settings.DEFAULT_ANTHROPIC_MODEL] if getattr(settings, 'ANTHROPIC_API_KEY', None) else []
            },
            "features": [
                "Multi-provider AI support",
                "Free model execution",
                "Async node processing",
                "Context-aware execution",
                "Rate limiting",
                "Error handling and retries"
            ]
        }

        return capabilities

    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))