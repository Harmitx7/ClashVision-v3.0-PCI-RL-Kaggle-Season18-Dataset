from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import logging
import asyncio
from datetime import datetime

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.router import api_router
from app.services.websocket_manager import WebSocketManager
from app.services.clash_royale_service import ClashRoyaleService
from app.ml.predictor import WinPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Trigger reload for new API token

# Global instances
websocket_manager = WebSocketManager()
clash_service = ClashRoyaleService()
win_predictor = WinPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Clash Royale Win Predictor API...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Initialize ML model
    await win_predictor.initialize()
    
    # Start background tasks
    await clash_service.start_background_tasks()
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await clash_service.stop_background_tasks()
    logger.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Clash Royale Win Predictor API",
    description="Real-time AI-powered win prediction for Clash Royale matches",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "v3.0-PCI-RL"
    }

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Clash Royale Win Predictor API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected",
        "ml_model": "loaded" if win_predictor.is_ready else "loading"
    }

@app.websocket("/ws/predictions/{player_tag}")
async def websocket_predictions(websocket: WebSocket, player_tag: str):
    """WebSocket endpoint for real-time predictions"""
    await websocket_manager.connect(websocket, player_tag)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Process prediction request
            prediction = await win_predictor.predict_live(player_tag, data)
            
            # Send prediction to all connected clients for this player
            await websocket_manager.send_prediction(player_tag, prediction)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, player_tag)
    except Exception as e:
        logger.error(f"WebSocket error for {player_tag}: {e}")
        websocket_manager.disconnect(websocket, player_tag)

@app.websocket("/ws/battles/{player_tag}")
async def websocket_battles(websocket: WebSocket, player_tag: str):
    """WebSocket endpoint for live battle monitoring"""
    await websocket_manager.connect_battle(websocket, player_tag)
    try:
        while True:
            # Monitor battle state
            battle_data = await clash_service.get_live_battle_data(player_tag)
            
            if battle_data:
                # Send battle updates
                await websocket_manager.send_battle_update(player_tag, battle_data)
            
            # Wait before next update
            await asyncio.sleep(settings.BATTLE_UPDATE_INTERVAL)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect_battle(websocket, player_tag)
    except Exception as e:
        logger.error(f"Battle WebSocket error for {player_tag}: {e}")
        websocket_manager.disconnect_battle(websocket, player_tag)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
