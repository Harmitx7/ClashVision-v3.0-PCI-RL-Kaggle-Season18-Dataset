from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from app.core.database import get_db
from app.services.clash_royale_service import ClashRoyaleService
from app.models.battle import Battle
from app.schemas.battle import BattleResponse, BattleAnalysisResponse

router = APIRouter()
logger = logging.getLogger(__name__)
clash_service = ClashRoyaleService()

@router.post("/analyze")
async def analyze_battle(
    battle_data: dict,
    db: Session = Depends(get_db)
):
    """Analyze a battle and provide insights"""
    try:
        # Extract battle information
        player_tag = battle_data.get("player_tag", "").replace("#", "").upper()
        
        if not player_tag:
            raise HTTPException(status_code=400, detail="Player tag is required")
        
        # Analyze battle performance
        analysis = {
            "battle_id": battle_data.get("battle_id"),
            "player_tag": player_tag,
            "result": battle_data.get("result"),
            "performance_metrics": {
                "elixir_efficiency": _calculate_elixir_efficiency(battle_data),
                "deck_synergy": _calculate_deck_synergy(battle_data),
                "defensive_rating": _calculate_defensive_rating(battle_data),
                "offensive_rating": _calculate_offensive_rating(battle_data)
            },
            "recommendations": _generate_battle_recommendations(battle_data)
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing battle: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/recent")
async def get_recent_battles(
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get recent battles from all players"""
    try:
        battles = db.query(Battle)\
            .order_by(Battle.created_at.desc())\
            .limit(limit)\
            .all()
        
        return {
            "battles": [BattleResponse.from_orm(battle) for battle in battles],
            "count": len(battles)
        }
        
    except Exception as e:
        logger.error(f"Error getting recent battles: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats")
async def get_battle_stats(
    db: Session = Depends(get_db)
):
    """Get overall battle statistics"""
    try:
        total_battles = db.query(Battle).count()
        
        # Win rate by arena
        arena_stats = db.query(
            Battle.arena_id,
            Battle.arena_name
        ).distinct().all()
        
        # Game mode distribution
        game_mode_stats = db.query(
            Battle.game_mode_name
        ).distinct().all()
        
        return {
            "total_battles": total_battles,
            "arenas": len(arena_stats),
            "game_modes": len(game_mode_stats),
            "arena_list": [{"id": stat[0], "name": stat[1]} for stat in arena_stats],
            "game_mode_list": [stat[0] for stat in game_mode_stats if stat[0]]
        }
        
    except Exception as e:
        logger.error(f"Error getting battle stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def _calculate_elixir_efficiency(battle_data: dict) -> float:
    """Calculate elixir efficiency score"""
    try:
        player_deck = battle_data.get("player_deck", [])
        if not player_deck:
            return 0.0
        
        # Simple efficiency calculation based on deck cost and performance
        avg_cost = sum(card.get("elixirCost", 0) for card in player_deck) / len(player_deck)
        
        # Lower cost generally means higher efficiency potential
        efficiency = max(0.0, min(1.0, (5.0 - avg_cost) / 2.0))
        
        return efficiency
        
    except Exception as e:
        logger.error(f"Error calculating elixir efficiency: {e}")
        return 0.0

def _calculate_deck_synergy(battle_data: dict) -> float:
    """Calculate deck synergy score"""
    try:
        player_deck = battle_data.get("player_deck", [])
        if len(player_deck) < 2:
            return 0.0
        
        # Simple synergy calculation
        card_names = [card.get("name", "") for card in player_deck]
        
        # Known synergy combinations
        synergies = [
            ("Giant", "Wizard"),
            ("Hog Rider", "Freeze"),
            ("Golem", "Night Witch"),
            ("Miner", "Poison")
        ]
        
        synergy_score = 0.0
        for card1, card2 in synergies:
            if card1 in card_names and card2 in card_names:
                synergy_score += 0.25
        
        return min(1.0, synergy_score)
        
    except Exception as e:
        logger.error(f"Error calculating deck synergy: {e}")
        return 0.0

def _calculate_defensive_rating(battle_data: dict) -> float:
    """Calculate defensive performance rating"""
    try:
        player_crowns = battle_data.get("player_crowns", 0)
        opponent_crowns = battle_data.get("opponent_crowns", 0)
        
        # Better defense = fewer crowns lost
        if opponent_crowns == 0:
            return 1.0
        elif opponent_crowns == 1:
            return 0.7
        elif opponent_crowns == 2:
            return 0.4
        else:
            return 0.1
        
    except Exception as e:
        logger.error(f"Error calculating defensive rating: {e}")
        return 0.5

def _calculate_offensive_rating(battle_data: dict) -> float:
    """Calculate offensive performance rating"""
    try:
        player_crowns = battle_data.get("player_crowns", 0)
        
        # Better offense = more crowns taken
        if player_crowns == 3:
            return 1.0
        elif player_crowns == 2:
            return 0.7
        elif player_crowns == 1:
            return 0.4
        else:
            return 0.1
        
    except Exception as e:
        logger.error(f"Error calculating offensive rating: {e}")
        return 0.5

def _generate_battle_recommendations(battle_data: dict) -> List[str]:
    """Generate recommendations based on battle performance"""
    try:
        recommendations = []
        
        result = battle_data.get("result", "loss")
        player_crowns = battle_data.get("player_crowns", 0)
        opponent_crowns = battle_data.get("opponent_crowns", 0)
        
        if result == "loss":
            if opponent_crowns == 3:
                recommendations.append("Focus on early defense to prevent three-crown defeats")
            elif opponent_crowns == 2:
                recommendations.append("Improve late-game defense")
            else:
                recommendations.append("Work on converting defensive stops into counter-attacks")
        
        elif result == "win":
            if player_crowns == 3:
                recommendations.append("Excellent offensive execution - maintain this aggression")
            else:
                recommendations.append("Good win - look for opportunities to secure more crowns")
        
        # Deck-specific recommendations
        player_deck = battle_data.get("player_deck", [])
        if player_deck:
            avg_cost = sum(card.get("elixirCost", 0) for card in player_deck) / len(player_deck)
            
            if avg_cost > 4.5:
                recommendations.append("Heavy deck - focus on building strong pushes")
            elif avg_cost < 3.0:
                recommendations.append("Fast cycle - maintain constant pressure")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []
