from typing import Dict, Any, Optional, Union, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models import BaseChatModel
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class AIModelFactory:
    """Factory for creating AI model instances"""

    _models: Dict[str, BaseChatModel] = {}

    @classmethod
    def get_model(
            cls,
            provider: str = "huggingface",
            model_name: Optional[str] = None,
            **kwargs
    ) -> BaseChatModel:
        """Get AI model instance with caching"""

        # Set default model names
        if model_name is None:
            model_name = cls._get_default_model(provider)

        cache_key = f"{provider}:{model_name}:{hash(str(sorted(kwargs.items())))}"

        if cache_key in cls._models:
            return cls._models[cache_key]

        try:
            model = cls._create_model(provider, model_name, **kwargs)
            cls._models[cache_key] = model
            logger.info(f"Created new AI model: {provider}:{model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to create AI model {provider}:{model_name}: {e}")
            raise

    @classmethod
    @classmethod
    def _get_default_model(cls, provider: str) -> str:
        """Get the default model name for provider"""
        defaults = {
            "openai": settings.DEFAULT_OPENAI_MODEL,
            "anthropic": settings.DEFAULT_ANTHROPIC_MODEL,
            "huggingface": settings.DEFAULT_HUGGINGFACE_MODEL
        }
        return defaults.get(provider, "gpt-3.5-turbo")

    @classmethod
    def _create_model(
            cls,
            provider: str,
            model_name: str,
            **kwargs
    ) -> ChatOpenAI | ChatAnthropic | HuggingFaceEndpoint:
        """Create new model instance"""

        common_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "request_timeout": kwargs.get("timeout", settings.AI_REQUEST_TIMEOUT),
            "max_retries": kwargs.get("max_retries", settings.AI_MAX_RETRIES),
        }

        if provider.lower() == "openai":
            return cls._create_openai_model(model_name, common_config, **kwargs)
        elif provider.lower() == "anthropic":
            return cls._create_anthropic_model(model_name, common_config, **kwargs)
        elif provider.lower() == "huggingface":
            return cls._create_huggingface_model(model_name, common_config, **kwargs)
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")

    @classmethod
    def _create_openai_model(
            cls,
            model_name: str,
            common_config: Dict[str, Any],
            **kwargs
    ) -> ChatOpenAI:
        """Create OpenAI model instance"""

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")

        config = {
            "model": model_name,
            "api_key": settings.OPENAI_API_KEY,
            **common_config
        }

        if "max_tokens" in kwargs:
            config["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            config["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            config["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            config["presence_penalty"] = kwargs["presence_penalty"]

        return ChatOpenAI(**config)

    @classmethod
    def _create_anthropic_model(
            cls,
            model_name: str,
            common_config: Dict[str, Any],
            **kwargs
    ) -> ChatAnthropic:
        """Create Anthropic model instance"""

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")

        config = {
            "model": model_name,
            "api_key": settings.ANTHROPIC_API_KEY,
            **common_config
        }

        if "max_tokens" in kwargs:
            config["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            config["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            config["top_k"] = kwargs["top_k"]

        return ChatAnthropic(**config)

    @classmethod
    def _create_huggingface_model(
            cls,
            model_name: str,
            common_config: Dict[str, Any],
            **kwargs
    ) -> HuggingFaceEndpoint:
        """Create Hugging Face model instance"""

        repo_id = model_name
        endpoint_url = f"{settings.HUGGINGFACE_INFERENCE_URL}/{repo_id}"

        config = {
            "endpoint_url": endpoint_url,
            "task": kwargs.get("task", "text-generation"),
            "temperature": common_config.get("temperature", 0.7),
            "max_new_tokens": kwargs.get("max_tokens", settings.HUGGINGFACE_MAX_NEW_TOKENS),
            "return_full_text": kwargs.get("return_full_text", False),
            "do_sample": kwargs.get("do_sample", True)
        }

        if settings.HUGGINGFACE_API_KEY:
            config["huggingfacehub_api_token"] = settings.HUGGINGFACE_API_KEY

        # Optional generation parameters (must also be flattened)
        for optional_param in ["top_p", "top_k", "repetition_penalty"]:
            if optional_param in kwargs:
                config[optional_param] = kwargs[optional_param]

        return HuggingFaceEndpoint(**config)

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get a list of available AI providers"""
        providers = ["huggingface"]  # Always available (free)

        if settings.OPENAI_API_KEY:
            providers.append("openai")
        if settings.ANTHROPIC_API_KEY:
            providers.append("anthropic")

        return providers

    @classmethod
    def get_popular_huggingface_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get a list of popular free Hugging Face models"""
        return {
            "microsoft/Phi-3-mini-4k-instruct": {
                "description": "Small but powerful instruction-following model",
                "type": "instruction",
                "task": "text-generation",
                "free": True,
                "reliable": True,
                "endpoint_type": "standard"
            },
            "deepseek/deepseek-v3-0324": {
                "description": "DeepSeek V3 model via Novita router",
                "type": "chat",
                "task": "text-generation",
                "free": True,
                "reliable": True,
                "endpoint_type": "openai_compatible",
                "custom_endpoint": "https://router.huggingface.co/novita/v3/openai/chat/completions"
            },
            "deepset/roberta-base-squad2": {
                "description": "Question answering model based on RoBERTa",
                "type": "qa",
                "task": "question-answering",
                "free": True,
                "reliable": True,
                "endpoint_type": "standard"
            },
            "microsoft/DialoGPT-medium": {
                "description": "Conversational AI model (legacy)",
                "type": "chat",
                "task": "text-generation",
                "free": True,
                "reliable": True,
                "endpoint_type": "standard"
            }
        }

    @classmethod
    def clear_cache(cls):
        """Clear model cache"""
        cls._models.clear()
        logger.info("AI model cache cleared")


model_factory = AIModelFactory()
