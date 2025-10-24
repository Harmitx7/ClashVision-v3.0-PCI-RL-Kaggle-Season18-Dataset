from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize database components as None
engine = None
SessionLocal = None
Base = declarative_base()
redis_client = None

# Try to initialize database if URL is provided
try:
    if settings.DATABASE_URL:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=settings.DEBUG
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Database connection initialized")
    else:
        logger.warning("DATABASE_URL not provided, running without database")
except Exception as e:
    logger.warning(f"Database initialization failed: {e}")
    engine = None
    SessionLocal = None

# Try to initialize Redis
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("Redis connection initialized")
except Exception as e:
    logger.warning(f"Redis initialization failed: {e}")
    redis_client = None

def get_db():
    """Dependency to get database session"""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Please configure DATABASE_URL.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_redis():
    """Dependency to get Redis client"""
    if redis_client is None:
        raise RuntimeError("Redis not initialized. Please configure REDIS_URL.")
    return redis_client

def is_database_available() -> bool:
    """Check if database is available"""
    return SessionLocal is not None

def is_redis_available() -> bool:
    """Check if Redis is available"""
    return redis_client is not None
