from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class PlayerBase(BaseModel):
    tag: str
    name: str
    trophies: int = 0
    best_trophies: int = 0
    wins: int = 0
    losses: int = 0
    battle_count: int = 0
    three_crown_wins: int = 0
    cards_found: int = 0
    favorite_card: Optional[str] = None
    total_donations: int = 0
    clan_tag: Optional[str] = None
    clan_role: Optional[str] = None
    arena_id: Optional[int] = None
    arena_name: Optional[str] = None
    league_id: Optional[int] = None
    league_name: Optional[str] = None
    exp_level: int = 1
    exp_points: int = 0
    star_points: int = 0

class PlayerResponse(PlayerBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class PlayerStatsResponse(BaseModel):
    id: int
    player_tag: str
    win_rate: float
    average_elixir_cost: float
    favorite_deck: Optional[Dict[str, Any]] = None
    recent_performance: Optional[Dict[str, Any]] = None
    skill_rating: float
    consistency_score: float
    deck_mastery: float
    calculated_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class PlayerCardResponse(BaseModel):
    card_name: str
    level: int
    max_level: int
    count: int = 0
    star_level: int = 0
    
    class Config:
        from_attributes = True
