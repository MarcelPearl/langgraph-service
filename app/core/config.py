from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
   
    APP_NAME: str = "AI Workflow Automation - FastAPI Engine"
    DEBUG: bool = False
    API_VERSION: str = "v1"

    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_SOCKET_TIMEOUT: int = 30
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 30

    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_GROUP_ID: str = "fastapi-ai-workers"
    KAFKA_AUTO_OFFSET_RESET: str = "latest"
    KAFKA_ENABLE_AUTO_COMMIT: bool = True
    KAFKA_AUTO_COMMIT_INTERVAL_MS: int = 5000
    KAFKA_REQUEST_TIMEOUT_MS: int = 30000
    KAFKA_RETRY_BACKOFF_MS: int = 1000
    KAFKA_MAX_REQUEST_SIZE: int = 1048576  # 1MB

    KAFKA_TOPIC_COORDINATION: str = "workflow-coordination"
    KAFKA_TOPIC_FASTAPI_QUEUE: str = "fastapi-execution-queue"
    KAFKA_TOPIC_SPRING_QUEUE: str = "spring-execution-queue"
    KAFKA_TOPIC_COMPLETION: str = "node-completion-events"
    KAFKA_TOPIC_STATE_UPDATES: str = "workflow-state-updates"

    SECRET_KEY: str = "my-super-secret-jwt-key-1234567890"

    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None

    DEFAULT_OPENAI_MODEL: str = "gpt-4"
    DEFAULT_ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
    DEFAULT_HUGGINGFACE_MODEL: str = "microsoft/Phi-3-mini-4k-instruct"

    HUGGINGFACE_INFERENCE_URL: str = "https://api-inference.huggingface.co/models"
    HUGGINGFACE_FREE_TIER_DELAY: float = 2.0
    HUGGINGFACE_MAX_NEW_TOKENS: int = 1000

    AI_REQUEST_TIMEOUT: int = 120
    AI_MAX_RETRIES: int = 3
    AI_RETRY_DELAY: float = 1.0

    WORKFLOW_MAX_EXECUTION_TIME: int = 3600  
    MAX_WORKFLOW_STEPS: int = 100
    NODE_EXECUTION_TIMEOUT: int = 300 

    AI_CALLS_PER_MINUTE: int = 60
    AI_CALLS_PER_HOUR: int = 1000
    NODE_CONCURRENT_LIMIT: int = 10

    LOCAL_MODEL_CACHE_DIR: str = "./models"
    USE_LOCAL_MODELS: bool = True
    PREFERRED_LOCAL_MODELS: List[str] = [
        "microsoft/Phi-3-mini-4k-instruct",
        "deepset/roberta-base-squad2"
    ]

    EXECUTION_CONTEXT_TTL: int = 86400 
    RESULT_CACHE_TTL: int = 43200 
    CLEANUP_INTERVAL_HOURS: int = 24

    NODE_RETRY_ATTEMPTS: int = 3
    NODE_RETRY_DELAY: float = 5.0
    NODE_TIMEOUT_SECONDS: int = 300

    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    ENABLE_DISTRIBUTED_TRACING: bool = False
    JAEGER_ENDPOINT: Optional[str] = None

    MAX_CONCURRENT_EXECUTIONS: int = 100
    THREAD_POOL_SIZE: int = 20
    ASYNC_TIMEOUT: int = 30

    ENABLE_ERROR_DETAILS: bool = True
    ERROR_NOTIFICATION_WEBHOOK: Optional[str] = None


    SERVICE_REGISTRY_URL: Optional[str] = None
    CONSUL_HOST: Optional[str] = None
    CONSUL_PORT: int = 8500

    ENVIRONMENT: str = "development"  

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.ENVIRONMENT == "production":
            self.DEBUG = False
            self.LOG_LEVEL = "WARNING"
            self.ENABLE_ERROR_DETAILS = False
        elif self.ENVIRONMENT == "development":
            self.DEBUG = True
            self.LOG_LEVEL = "DEBUG"
            self.ENABLE_ERROR_DETAILS = True

    def get_kafka_config(self) -> dict:
        """Get Kafka configuration dictionary"""
        return {
            "bootstrap_servers": self.KAFKA_BOOTSTRAP_SERVERS,
            "group_id": self.KAFKA_GROUP_ID,
            "auto_offset_reset": self.KAFKA_AUTO_OFFSET_RESET,
            "enable_auto_commit": self.KAFKA_ENABLE_AUTO_COMMIT,
            "auto_commit_interval_ms": self.KAFKA_AUTO_COMMIT_INTERVAL_MS,
            "request_timeout_ms": self.KAFKA_REQUEST_TIMEOUT_MS,
            "retry_backoff_ms": self.KAFKA_RETRY_BACKOFF_MS,
            "max_request_size": self.KAFKA_MAX_REQUEST_SIZE
        }

    def get_redis_config(self) -> dict:
        """Get Redis configuration dictionary"""
        return {
            "url": self.REDIS_URL,
            "max_connections": self.REDIS_MAX_CONNECTIONS,
            "socket_timeout": self.REDIS_SOCKET_TIMEOUT,
            "socket_connect_timeout": self.REDIS_SOCKET_CONNECT_TIMEOUT
        }

    def get_ai_provider_config(self) -> dict:
        """Get AI provider configuration"""
        providers = {}

        if self.OPENAI_API_KEY:
            providers["openai"] = {
                "api_key": self.OPENAI_API_KEY,
                "default_model": self.DEFAULT_OPENAI_MODEL,
                "timeout": self.AI_REQUEST_TIMEOUT,
                "max_retries": self.AI_MAX_RETRIES
            }

        if self.ANTHROPIC_API_KEY:
            providers["anthropic"] = {
                "api_key": self.ANTHROPIC_API_KEY,
                "default_model": self.DEFAULT_ANTHROPIC_MODEL,
                "timeout": self.AI_REQUEST_TIMEOUT,
                "max_retries": self.AI_MAX_RETRIES
            }

        if self.HUGGINGFACE_API_KEY:
            providers["huggingface"] = {
                "api_key": self.HUGGINGFACE_API_KEY,
                "default_model": self.DEFAULT_HUGGINGFACE_MODEL,
                "inference_url": self.HUGGINGFACE_INFERENCE_URL,
                "free_tier_delay": self.HUGGINGFACE_FREE_TIER_DELAY,
                "max_new_tokens": self.HUGGINGFACE_MAX_NEW_TOKENS
            }
        else:
            # HuggingFace can work without API key (free tier)
            providers["huggingface"] = {
                "api_key": None,
                "default_model": self.DEFAULT_HUGGINGFACE_MODEL,
                "inference_url": self.HUGGINGFACE_INFERENCE_URL,
                "free_tier_delay": self.HUGGINGFACE_FREE_TIER_DELAY,
                "max_new_tokens": self.HUGGINGFACE_MAX_NEW_TOKENS
            }

        return providers

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()