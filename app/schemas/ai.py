from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid

class AIModelRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt for the AI model")
    provider: str = Field(..., description="AI provider (openai, anthropic, huggingface)")
    model: str = Field(..., description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=500, ge=1, le=4000, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    additional_params: Dict[str, Any] = Field(default={}, description="Additional model parameters")

class AIModelResponse(BaseModel):
    response: str = Field(..., description="Generated response from the AI model")
    model: str = Field(..., description="Model that generated the response")
    provider: str = Field(..., description="AI provider used")
    response_metadata: Dict[str, Any] = Field(default={}, description="Additional response metadata")

class HuggingFaceModelInfo(BaseModel):
    model_id: str = Field(..., description="Hugging Face model ID")
    info: Dict[str, Any] = Field(..., description="Model information from HF API")
    available: bool = Field(..., description="Whether the model is available for inference")
    error: Optional[str] = Field(None, description="Error message if model not available")

class AvailableProvidersResponse(BaseModel):
    providers: List[str] = Field(..., description="List of available AI providers")
    huggingface_models: Dict[str, Dict[str, Any]] = Field(..., description="Popular HF models")
    total_providers: int = Field(..., description="Total number of available providers")

class ModelTestResult(BaseModel):
    model_name: str
    status: str
    response: Optional[str] = None
    error: Optional[str] = None
    response_time: Optional[float] = None

class FreeModelConfiguration(BaseModel):
    """Configuration for free AI models"""
    prefer_free: bool = Field(default=True, description="Prefer free models when available")
    fallback_to_paid: bool = Field(default=False, description="Fallback to paid models if free fails")
    max_wait_time: int = Field(default=60, description="Max wait time for model loading")
    rate_limit_delay: float = Field(default=2.0, description="Delay between free API calls")