import httpx
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import json

from app.core.config import settings
from app.core.database import get_redis
from app.models.player import Player, PlayerStats, PlayerCard
from app.models.battle import Battle, Card
from app.models.clan import Clan

logger = logging.getLogger(__name__)

class ClashRoyaleService:
    def __init__(self):
        self.base_url = settings.CLASH_ROYALE_BASE_URL
        self.api_key = settings.CLASH_ROYALE_API_KEY
        self.redis_client = get_redis()
        self.rate_limit_delay = 60 / settings.API_RATE_LIMIT_PER_MINUTE
        self.last_request_time = 0
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
    
    async def _make_request(self, endpoint: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Make rate-limited request to Clash Royale API with retry logic"""
        cache_key = f"cr_api:{endpoint}"
        
        # Check cache first
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    logger.info(f"Cache hit for endpoint: {endpoint}")
                    return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                # Rate limiting
                current_time = asyncio.get_event_loop().time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.rate_limit_delay:
                    await asyncio.sleep(self.rate_limit_delay - time_since_last)
                
                self.last_request_time = asyncio.get_event_loop().time()
                
                # Make API request
                full_url = f"{self.base_url}{endpoint}"
                logger.info(f"Making API request (attempt {attempt + 1}/{max_retries}): {full_url}")
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        full_url,
                        headers=self.headers,
                        timeout=30.0
                    )
                    
                    logger.info(f"API response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"API response data keys: {list(data.keys()) if data else 'None'}")
                        
                        # Cache the response
                        try:
                            if self.redis_client:
                                self.redis_client.setex(
                                    cache_key,
                                    600,  # 10 minutes cache
                                    json.dumps(data)
                                )
                        except Exception as cache_error:
                            logger.warning(f"Cache write error: {cache_error}")
                        
                        return data
                    elif response.status_code == 404:
                        logger.warning(f"Resource not found: {endpoint}")
                        return None
                    elif response.status_code == 429:
                        # Rate limited, wait longer
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"API request failed: {response.status_code} - {response.text}")
                        if attempt == max_retries - 1:
                            return None
                        continue
                        
            except Exception as e:
                logger.error(f"Error making API request to {endpoint} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                
                # Exponential backoff
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        
        return None
    
    async def get_player(self, player_tag: str) -> Optional[Dict[str, Any]]:
        """Get player data from API with validation"""
        clean_tag = player_tag.replace("#", "").upper()
        data = await self._make_request(f"/players/%23{clean_tag}")
        
        if data:
            # Validate and sanitize player data
            data = self._validate_player_data(data)
        
        return data
    
    def _validate_player_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize player data"""
        try:
            # Ensure required fields exist with defaults
            validated_data = {
                'tag': data.get('tag', ''),
                'name': data.get('name', 'Unknown'),
                'trophies': max(0, data.get('trophies', 0)),
                'bestTrophies': max(0, data.get('bestTrophies', 0)),
                'wins': max(0, data.get('wins', 0)),
                'losses': max(0, data.get('losses', 0)),
                'battleCount': max(0, data.get('battleCount', 0)),
                'threeCrownWins': max(0, data.get('threeCrownWins', 0)),
                'challengeCardsWon': max(0, data.get('challengeCardsWon', 0)),
                'challengeMaxWins': max(0, data.get('challengeMaxWins', 0)),
                'tournamentCardsWon': max(0, data.get('tournamentCardsWon', 0)),
                'tournamentBattleCount': max(0, data.get('tournamentBattleCount', 0)),
                'role': data.get('role', 'member'),
                'donations': max(0, data.get('donations', 0)),
                'donationsReceived': max(0, data.get('donationsReceived', 0)),
                'totalDonations': max(0, data.get('totalDonations', 0)),
                'warDayWins': max(0, data.get('warDayWins', 0)),
                'clanCardsCollected': max(0, data.get('clanCardsCollected', 0)),
                'arena': data.get('arena', {}),
                'leagueStatistics': data.get('leagueStatistics', {}),
                'badges': data.get('badges', []),
                'achievements': data.get('achievements', []),
                'cards': data.get('cards', []),
                'currentDeck': data.get('currentDeck', []),
                'currentFavouriteCard': data.get('currentFavouriteCard', {}),
                'starPoints': max(0, data.get('starPoints', 0)),
                'expLevel': max(1, data.get('expLevel', 1)),
                'expPoints': max(0, data.get('expPoints', 0)),
                'clan': data.get('clan', {})
            }
            
            # Validate current deck
            if validated_data['currentDeck']:
                validated_data['currentDeck'] = self._validate_deck_data(validated_data['currentDeck'])
            
            logger.debug(f"Player data validated for {validated_data['tag']}")
            return validated_data
            
        except Exception as e:
            logger.error(f"Error validating player data: {e}")
            return data  # Return original data if validation fails
    
    def _validate_deck_data(self, deck: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and sanitize deck data"""
        validated_deck = []
        
        for card in deck:
            if isinstance(card, dict):
                validated_card = {
                    'name': card.get('name', 'Unknown'),
                    'id': card.get('id', 0),
                    'level': max(1, min(14, card.get('level', 1))),
                    'maxLevel': max(1, min(14, card.get('maxLevel', 1))),
                    'count': max(0, card.get('count', 0)),
                    'iconUrls': card.get('iconUrls', {}),
                    'elixirCost': max(0, min(10, card.get('elixirCost', 0)))
                }
                validated_deck.append(validated_card)
        
        return validated_deck
    
    async def get_player_battles(self, player_tag: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Get player battle log from API"""
        clean_tag = player_tag.replace("#", "").upper()
        data = await self._make_request(f"/players/%23{clean_tag}/battlelog")
        return data if data else []
    
    async def get_upcoming_chests(self, player_tag: str) -> List[Dict[str, Any]]:
        """Get player's upcoming chests"""
        clean_tag = player_tag.replace("#", "").upper()
        data = await self._make_request(f"/players/%23{clean_tag}/upcomingchests")
        return data.get("items", []) if data else []
    
    async def get_clan(self, clan_tag: str) -> Optional[Dict[str, Any]]:
        """Get clan data from API"""
        clean_tag = clan_tag.replace("#", "").upper()
        return await self._make_request(f"/clans/%23{clean_tag}")
    
    async def get_cards(self) -> List[Dict[str, Any]]:
        """Get all cards data"""
        data = await self._make_request("/cards")
        return data.get("items", []) if data else []
    
    async def update_player_in_db(self, db: Session, player_data: Dict[str, Any]) -> Player:
        """Update or create player in database"""
        try:
            player_tag = player_data["tag"].replace("#", "")
            
            # Check if player exists
            player = db.query(Player).filter(Player.tag == player_tag).first()
            
            if player:
                # Update existing player
                for key, value in player_data.items():
                    if hasattr(player, key) and key != "tag":
                        setattr(player, key, value)
                player.last_seen = datetime.utcnow()
            else:
                # Create new player
                player = Player(
                    tag=player_tag,
                    name=player_data.get("name", ""),
                    trophies=player_data.get("trophies", 0),
                    best_trophies=player_data.get("bestTrophies", 0),
                    wins=player_data.get("wins", 0),
                    losses=player_data.get("losses", 0),
                    battle_count=player_data.get("battleCount", 0),
                    three_crown_wins=player_data.get("threeCrownWins", 0),
                    cards_found=player_data.get("cardsFound", 0),
                    favorite_card=player_data.get("currentFavouriteCard", {}).get("name"),
                    total_donations=player_data.get("totalDonations", 0),
                    clan_tag=player_data.get("clan", {}).get("tag", "").replace("#", ""),
                    clan_role=player_data.get("role"),
                    arena_id=player_data.get("arena", {}).get("id"),
                    arena_name=player_data.get("arena", {}).get("name"),
                    exp_level=player_data.get("expLevel", 1),
                    exp_points=player_data.get("expPoints", 0),
                    star_points=player_data.get("starPoints", 0),
                    last_seen=datetime.utcnow()
                )
            
            db.add(player)
            db.commit()
            db.refresh(player)
            
            # Update player cards
            if "cards" in player_data:
                await self._update_player_cards(db, player_tag, player_data["cards"])
            
            return player
            
        except Exception as e:
            logger.error(f"Error updating player in database: {e}")
            db.rollback()
            raise
    
    async def _update_player_cards(self, db: Session, player_tag: str, cards_data: List[Dict]):
        """Update player's cards in database"""
        try:
            # Delete existing cards
            db.query(PlayerCard).filter(PlayerCard.player_tag == player_tag).delete()
            
            # Add new cards
            for card_data in cards_data:
                player_card = PlayerCard(
                    player_tag=player_tag,
                    card_name=card_data.get("name", ""),
                    level=card_data.get("level", 1),
                    max_level=card_data.get("maxLevel", 1),
                    count=card_data.get("count", 0),
                    star_level=card_data.get("starLevel", 0)
                )
                db.add(player_card)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error updating player cards: {e}")
            db.rollback()
    
    async def update_battles_in_db(self, db: Session, battles_data: List[Dict], player_tag: str):
        """Update battles in database"""
        try:
            for battle_data in battles_data:
                # Check if battle already exists
                battle_time = battle_data.get("battleTime", "")
                existing_battle = db.query(Battle).filter(
                    Battle.player_tag == player_tag,
                    Battle.battle_time == battle_time
                ).first()
                
                if existing_battle:
                    continue  # Skip if already exists
                
                # Create new battle record
                battle = Battle(
                    battle_time=battle_time,
                    type=battle_data.get("type", ""),
                    is_ladder_tournament=battle_data.get("isLadderTournament", False),
                    arena_id=battle_data.get("arena", {}).get("id"),
                    arena_name=battle_data.get("arena", {}).get("name"),
                    game_mode_id=battle_data.get("gameMode", {}).get("id"),
                    game_mode_name=battle_data.get("gameMode", {}).get("name"),
                    deck_selection=battle_data.get("deckSelection"),
                    player_tag=player_tag,
                    player_name=battle_data.get("team", [{}])[0].get("name", ""),
                    player_trophies=battle_data.get("team", [{}])[0].get("trophies", 0),
                    player_starting_trophies=battle_data.get("team", [{}])[0].get("startingTrophies", 0),
                    player_trophy_change=battle_data.get("team", [{}])[0].get("trophyChange", 0),
                    player_crowns=battle_data.get("team", [{}])[0].get("crowns", 0),
                    player_deck=battle_data.get("team", [{}])[0].get("cards", []),
                    opponent_tag=battle_data.get("opponent", [{}])[0].get("tag", "").replace("#", ""),
                    opponent_name=battle_data.get("opponent", [{}])[0].get("name", ""),
                    opponent_trophies=battle_data.get("opponent", [{}])[0].get("trophies", 0),
                    opponent_starting_trophies=battle_data.get("opponent", [{}])[0].get("startingTrophies", 0),
                    opponent_trophy_change=battle_data.get("opponent", [{}])[0].get("trophyChange", 0),
                    opponent_crowns=battle_data.get("opponent", [{}])[0].get("crowns", 0),
                    opponent_deck=battle_data.get("opponent", [{}])[0].get("cards", [])
                )
                
                # Determine result
                player_crowns = battle.player_crowns or 0
                opponent_crowns = battle.opponent_crowns or 0
                
                if player_crowns > opponent_crowns:
                    battle.result = "win"
                elif player_crowns < opponent_crowns:
                    battle.result = "loss"
                else:
                    battle.result = "draw"
                
                db.add(battle)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error updating battles in database: {e}")
            db.rollback()
    
    async def calculate_player_stats(self, db: Session, player_tag: str) -> Optional[PlayerStats]:
        """Calculate and update player statistics"""
        try:
            # Get recent battles
            battles = db.query(Battle).filter(
                Battle.player_tag == player_tag
            ).order_by(Battle.battle_time.desc()).limit(25).all()
            
            if not battles:
                return None
            
            # Calculate win rate
            wins = sum(1 for battle in battles if battle.result == "win")
            win_rate = wins / len(battles) if battles else 0.0
            
            # Calculate average elixir cost
            total_elixir = 0
            deck_count = 0
            
            for battle in battles:
                if battle.player_deck:
                    deck_elixir = sum(card.get("elixirCost", 0) for card in battle.player_deck)
                    if deck_elixir > 0:
                        total_elixir += deck_elixir / len(battle.player_deck)
                        deck_count += 1
            
            avg_elixir = total_elixir / deck_count if deck_count > 0 else 0.0
            
            # Get or create player stats
            stats = db.query(PlayerStats).filter(PlayerStats.player_tag == player_tag).first()
            
            if stats:
                stats.win_rate = win_rate
                stats.average_elixir_cost = avg_elixir
                stats.recent_performance = {
                    "battles": len(battles),
                    "wins": wins,
                    "losses": len(battles) - wins,
                    "win_streak": self._calculate_win_streak(battles)
                }
                stats.updated_at = datetime.utcnow()
            else:
                stats = PlayerStats(
                    player_tag=player_tag,
                    win_rate=win_rate,
                    average_elixir_cost=avg_elixir,
                    recent_performance={
                        "battles": len(battles),
                        "wins": wins,
                        "losses": len(battles) - wins,
                        "win_streak": self._calculate_win_streak(battles)
                    },
                    skill_rating=self._calculate_skill_rating(battles),
                    consistency_score=self._calculate_consistency_score(battles),
                    deck_mastery=self._calculate_deck_mastery(battles)
                )
            
            db.add(stats)
            db.commit()
            db.refresh(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating player stats: {e}")
            return None
    
    def _calculate_win_streak(self, battles: List[Battle]) -> int:
        """Calculate current win streak"""
        streak = 0
        for battle in battles:
            if battle.result == "win":
                streak += 1
            else:
                break
        return streak
    
    def _calculate_skill_rating(self, battles: List[Battle]) -> float:
        """Calculate skill rating based on battle performance"""
        if not battles:
            return 0.0
        
        # Simple skill rating based on trophy changes and win rate
        total_trophy_change = sum(battle.player_trophy_change or 0 for battle in battles)
        wins = sum(1 for battle in battles if battle.result == "win")
        win_rate = wins / len(battles)
        
        # Normalize and combine factors
        skill_rating = (win_rate * 0.7) + (min(total_trophy_change / 100, 1.0) * 0.3)
        return max(0.0, min(1.0, skill_rating))
    
    def _calculate_consistency_score(self, battles: List[Battle]) -> float:
        """Calculate consistency score based on performance variance"""
        if len(battles) < 5:
            return 0.5
        
        # Calculate variance in trophy changes
        trophy_changes = [battle.player_trophy_change or 0 for battle in battles]
        mean_change = sum(trophy_changes) / len(trophy_changes)
        variance = sum((x - mean_change) ** 2 for x in trophy_changes) / len(trophy_changes)
        
        # Lower variance = higher consistency
        consistency = max(0.0, min(1.0, 1.0 - (variance / 1000)))
        return consistency
    
    def _calculate_deck_mastery(self, battles: List[Battle]) -> float:
        """Calculate deck mastery based on deck usage patterns"""
        if not battles:
            return 0.0
        
        # Count deck variations
        deck_signatures = set()
        for battle in battles:
            if battle.player_deck:
                deck_sig = tuple(sorted(card.get("name", "") for card in battle.player_deck))
                deck_signatures.add(deck_sig)
        
        # Fewer deck variations = higher mastery
        mastery = max(0.0, min(1.0, 1.0 - (len(deck_signatures) / len(battles))))
        return mastery
    
    async def get_live_battle_data(self, player_tag: str) -> Optional[Dict[str, Any]]:
        """Get live battle data (simulated for now)"""
        # In a real implementation, this would connect to live battle data
        # For now, return None to indicate no active battle
        return None
    
    async def get_clan_current_war(self, clan_tag: str) -> Optional[Dict[str, Any]]:
        """Get current clan war"""
        clean_tag = clan_tag.replace("#", "").upper()
        return await self._make_request(f"/clans/%23{clean_tag}/currentwar")
    
    async def get_clan_current_river_race(self, clan_tag: str) -> Optional[Dict[str, Any]]:
        """Get current river race"""
        clean_tag = clan_tag.replace("#", "").upper()
        return await self._make_request(f"/clans/%23{clean_tag}/currentriverrace")
    
    async def update_clan_in_db(self, db: Session, clan_data: Dict[str, Any]) -> Clan:
        """Update or create clan in database"""
        try:
            clan_tag = clan_data["tag"].replace("#", "")
            
            # Check if clan exists
            clan = db.query(Clan).filter(Clan.tag == clan_tag).first()
            
            if clan:
                # Update existing clan
                for key, value in clan_data.items():
                    if hasattr(clan, key) and key != "tag":
                        setattr(clan, key, value)
            else:
                # Create new clan
                clan = Clan(
                    tag=clan_tag,
                    name=clan_data.get("name", ""),
                    description=clan_data.get("description", ""),
                    type=clan_data.get("type", ""),
                    score=clan_data.get("clanScore", 0),
                    required_trophies=clan_data.get("requiredTrophies", 0),
                    donations_per_week=clan_data.get("donationsPerWeek", 0),
                    member_count=clan_data.get("members", 0),
                    location_id=clan_data.get("location", {}).get("id"),
                    location_name=clan_data.get("location", {}).get("name"),
                    location_country_code=clan_data.get("location", {}).get("countryCode"),
                    badge_id=clan_data.get("badgeId"),
                    badge_name=clan_data.get("badge", {}).get("name"),
                    clan_war_trophies=clan_data.get("clanWarTrophies", 0),
                    clan_war_trophy_change=clan_data.get("clanWarTrophyChange", 0)
                )
            
            db.add(clan)
            db.commit()
            db.refresh(clan)
            
            return clan
            
        except Exception as e:
            logger.error(f"Error updating clan in database: {e}")
            db.rollback()
            raise
    
    async def start_background_tasks(self):
        """Start background tasks for data updates"""
        logger.info("Starting background tasks...")
        # Implement background tasks for periodic data updates
    
    async def stop_background_tasks(self):
        """Stop background tasks"""
        logger.info("Stopping background tasks...")
