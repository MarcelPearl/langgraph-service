from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "AI Workflow Automation"
    DEBUG: bool = False
    API_VERSION: str = "v1"
    
    DATABASE_URL: str
    
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    REDIS_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"

settings = Settings()