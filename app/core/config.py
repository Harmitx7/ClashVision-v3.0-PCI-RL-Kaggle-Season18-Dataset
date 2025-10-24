from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Clash Royale API
    CLASH_ROYALE_API_KEY: str
    CLASH_ROYALE_BASE_URL: str = "https://api.clashroyale.com/v1"
    
    # Database
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379"
    
    # JWT Configuration
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Rate Limiting
    API_RATE_LIMIT_PER_MINUTE: int = 100
    CACHE_TTL_SECONDS: int = 300
    
    # Model Configuration
    MODEL_UPDATE_INTERVAL: int = 3600  # seconds
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.7
    BATTLE_UPDATE_INTERVAL: int = 10  # seconds
    
    # File Paths
    MODEL_PATH: str = "app/ml/models/"
    DATA_PATH: str = "data/"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
