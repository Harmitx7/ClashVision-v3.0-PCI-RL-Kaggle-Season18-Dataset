from fastapi import APIRouter
from app.api.v1.endpoints import players_minimal, predictions, battles, clans, match_predictor

api_router = APIRouter()

api_router.include_router(players_minimal.router, prefix="/players", tags=["players"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(battles.router, prefix="/battles", tags=["battles"])
api_router.include_router(clans.router, prefix="/clans", tags=["clans"])
api_router.include_router(match_predictor.router, prefix="/match", tags=["match-prediction"])
