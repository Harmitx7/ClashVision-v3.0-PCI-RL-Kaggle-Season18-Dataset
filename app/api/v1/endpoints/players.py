from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from app.core.database import get_db
from app.services.clash_royale_service import ClashRoyaleService
from app.models.player import Player, PlayerStats
from app.schemas.player import PlayerResponse, PlayerStatsResponse

router = APIRouter()
logger = logging.getLogger(__name__)
clash_service = ClashRoyaleService()

@router.get("/{player_tag}", response_model=PlayerResponse)
async def get_player(
    player_tag: str,
    db: Session = Depends(get_db),
    refresh: bool = Query(False, description="Force refresh from API")
):
    """Get player information by tag"""
    try:
        # Clean player tag
        clean_tag = player_tag.replace("#", "").upper()
        
        if refresh:
            # Fetch fresh data from API
            player_data = await clash_service.get_player(clean_tag)
            if not player_data:
                raise HTTPException(status_code=404, detail="Player not found")
            
            # Update database
            await clash_service.update_player_in_db(db, player_data)
        
        # Get from database
        player = db.query(Player).filter(Player.tag == clean_tag).first()
        if not player:
            # Try to fetch from API if not in database
            player_data = await clash_service.get_player(clean_tag)
            if not player_data:
                raise HTTPException(status_code=404, detail="Player not found")
            
            player = await clash_service.update_player_in_db(db, player_data)
        
        return PlayerResponse.from_orm(player)
        
    except Exception as e:
        logger.error(f"Error getting player {player_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{player_tag}/stats", response_model=PlayerStatsResponse)
async def get_player_stats(
    player_tag: str,
    db: Session = Depends(get_db)
):
    """Get calculated player statistics"""
    try:
        clean_tag = player_tag.replace("#", "").upper()
        
        stats = db.query(PlayerStats).filter(PlayerStats.player_tag == clean_tag).first()
        if not stats:
            # Calculate stats if not available
            stats = await clash_service.calculate_player_stats(db, clean_tag)
            if not stats:
                raise HTTPException(status_code=404, detail="Player stats not available")
        
        return PlayerStatsResponse.from_orm(stats)
        
    except Exception as e:
        logger.error(f"Error getting player stats {player_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{player_tag}/battles")
async def get_player_battles(
    player_tag: str,
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get player battle history"""
    try:
        clean_tag = player_tag.replace("#", "").upper()
        
        # Get battles from API
        battles = await clash_service.get_player_battles(clean_tag, limit)
        
        return {
            "player_tag": clean_tag,
            "battles": battles,
            "count": len(battles)
        }
        
    except Exception as e:
        logger.error(f"Error getting player battles {player_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{player_tag}/upcoming-chests")
async def get_upcoming_chests(
    player_tag: str
):
    """Get player's upcoming chest cycle"""
    try:
        clean_tag = player_tag.replace("#", "").upper()
        
        chests = await clash_service.get_upcoming_chests(clean_tag)
        
        return {
            "player_tag": clean_tag,
            "upcoming_chests": chests
        }
        
    except Exception as e:
        logger.error(f"Error getting upcoming chests {player_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{player_tag}/refresh")
async def refresh_player_data(
    player_tag: str,
    db: Session = Depends(get_db)
):
    """Force refresh all player data from API"""
    try:
        clean_tag = player_tag.replace("#", "").upper()
        
        # Refresh player data
        player_data = await clash_service.get_player(clean_tag)
        if not player_data:
            raise HTTPException(status_code=404, detail="Player not found")
        
        # Update in database
        player = await clash_service.update_player_in_db(db, player_data)
        
        # Refresh battles
        battles = await clash_service.get_player_battles(clean_tag, 25)
        await clash_service.update_battles_in_db(db, battles, clean_tag)
        
        # Recalculate stats
        stats = await clash_service.calculate_player_stats(db, clean_tag)
        
        return {
            "message": "Player data refreshed successfully",
            "player_tag": clean_tag,
            "updated_at": player.updated_at.isoformat() if player.updated_at else None
        }
        
    except Exception as e:
        logger.error(f"Error refreshing player data {player_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
