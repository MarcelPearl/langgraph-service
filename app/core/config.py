from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "AI Workflow Automation"
    DEBUG: bool = False
    API_VERSION: str = "v1"

    DATABASE_URL: str
    REDIS_URL: Optional[str] = None
    SECRET_KEY: str = "my-super-secret-jwt-key-1234567890"

    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None

    DEFAULT_OPENAI_MODEL: str = "gpt-4"
    DEFAULT_ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
    DEFAULT_HUGGINGFACE_MODEL: str = "microsoft/DialoGPT-medium"

    HUGGINGFACE_INFERENCE_URL: str = "https://api-inference.huggingface.co/models"
    HUGGINGFACE_FREE_TIER_DELAY: float = 2.0
    HUGGINGFACE_MAX_NEW_TOKENS: int = 1000

    AI_REQUEST_TIMEOUT: int = 120
    AI_MAX_RETRIES: int = 3
    AI_RETRY_DELAY: float = 1.0

    LANGGRAPH_CHECKPOINT_DB: Optional[str] = None
    WORKFLOW_MAX_EXECUTION_TIME: int = 3600
    MAX_WORKFLOW_STEPS: int = 100

    AI_CALLS_PER_MINUTE: int = 60
    AI_CALLS_PER_HOUR: int = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.LANGGRAPH_CHECKPOINT_DB:
            self.LANGGRAPH_CHECKPOINT_DB = self.DATABASE_URL

    class Config:
        env_file = ".env"


settings = Settings()
