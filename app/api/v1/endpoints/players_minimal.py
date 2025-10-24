from fastapi import APIRouter, HTTPException, Query
import httpx
import os
import logging
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/test")
async def test_api():
    """Test endpoint to verify API configuration"""
    api_key = os.getenv('CLASH_ROYALE_API_KEY')
    base_url = os.getenv('CLASH_ROYALE_BASE_URL', 'https://api.clashroyale.com/v1')
    
    return {
        "api_configured": bool(api_key),
        "api_key_length": len(api_key) if api_key else 0,
        "base_url": base_url,
        "status": "API configuration test"
    }


@router.get("/{player_tag}")
async def get_player(player_tag: str, refresh: bool = Query(False)):
    """Get player information by tag (minimal version)"""
    try:
        # Clean player tag
        clean_tag = player_tag.replace("#", "").upper()
        
        # Get API credentials
        api_key = os.getenv('CLASH_ROYALE_API_KEY')
        base_url = os.getenv('CLASH_ROYALE_BASE_URL', 'https://api.clashroyale.com/v1')
        
        if not api_key:
            logger.error("Clash Royale API key not configured")
            raise HTTPException(status_code=500, detail="API key not configured")
        
        # Make direct API call
        url = f"{base_url}/players/%23{clean_tag}"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }
        
        logger.info(f"Fetching player data for tag: {clean_tag}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)
            
            logger.info(f"API response status: {response.status_code}")
            
            if response.status_code == 200:
                player_data = response.json()
                
                # Return simplified player data with current deck
                current_deck = []
                if "currentDeck" in player_data:
                    current_deck = [
                        {
                            "name": card.get("name", "Unknown"),
                            "elixirCost": card.get("elixirCost", 0),
                            "level": card.get("level", 1)
                        }
                        for card in player_data["currentDeck"]
                    ]
                
                return {
                    "tag": clean_tag,
                    "name": player_data.get("name", "Unknown"),
                    "trophies": player_data.get("trophies", 0),
                    "best_trophies": player_data.get("bestTrophies", 0),
                    "wins": player_data.get("wins", 0),
                    "losses": player_data.get("losses", 0),
                    "battle_count": player_data.get("battleCount", 0),
                    "three_crown_wins": player_data.get("threeCrownWins", 0),
                    "exp_level": player_data.get("expLevel", 1),
                    "arena_name": player_data.get("arena", {}).get("name", "Unknown"),
                    "clan_name": player_data.get("clan", {}).get("name", "No Clan"),
                    "clan_tag": player_data.get("clan", {}).get("tag", "").replace("#", ""),
                    "currentDeck": current_deck
                }
            elif response.status_code == 404:
                logger.warning(f"Player not found: {clean_tag}")
                raise HTTPException(status_code=404, detail=f"Player with tag #{clean_tag} not found")
            elif response.status_code == 403:
                logger.error(f"API access forbidden - check API key")
                raise HTTPException(status_code=500, detail="API access denied - invalid API key")
            else:
                logger.error(f"Clash Royale API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=500, detail=f"Clash Royale API error: {response.status_code}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error fetching player {clean_tag}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/{player_tag}/battles")
async def get_player_battles(player_tag: str):
    """Get player battles (minimal version)"""
    try:
        clean_tag = player_tag.replace("#", "").upper()
        
        # Get API credentials
        api_key = os.getenv('CLASH_ROYALE_API_KEY')
        base_url = os.getenv('CLASH_ROYALE_BASE_URL', 'https://api.clashroyale.com/v1')
        
        # Make direct API call
        url = f"{base_url}/players/%23{clean_tag}/battlelog"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)
            
            if response.status_code == 200:
                battles_data = response.json()
                return {
                    "player_tag": clean_tag,
                    "battles": battles_data,
                    "count": len(battles_data)
                }
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="Player not found")
            else:
                raise HTTPException(status_code=500, detail=f"API error: {response.status_code}")
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/{player_tag}/stats")
async def get_player_stats(player_tag: str):
    """Get player stats (minimal version)"""
    try:
        # Get player data first
        player_data = await get_player(player_tag)
        
        # Calculate simple stats
        wins = player_data.get("wins", 0)
        losses = player_data.get("losses", 0)
        total_battles = wins + losses
        win_rate = wins / max(1, total_battles)
        
        return {
            "player_tag": player_data["tag"],
            "win_rate": win_rate,
            "total_battles": total_battles,
            "skill_rating": min(1.0, player_data.get("trophies", 0) / 6000),
            "consistency_score": 0.7,  # Placeholder
            "deck_mastery": 0.8,  # Placeholder
            "average_elixir_cost": 3.8  # Placeholder
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
