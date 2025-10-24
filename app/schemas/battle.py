from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class BattleBase(BaseModel):
    battle_time: str
    type: str
    is_ladder_tournament: bool = False
    arena_id: Optional[int] = None
    arena_name: Optional[str] = None
    game_mode_id: Optional[int] = None
    game_mode_name: Optional[str] = None
    deck_selection: Optional[str] = None

class BattleResponse(BattleBase):
    id: int
    player_tag: str
    player_name: Optional[str] = None
    player_trophies: Optional[int] = None
    player_starting_trophies: Optional[int] = None
    player_trophy_change: Optional[int] = None
    player_crowns: Optional[int] = None
    player_deck: Optional[List[Dict[str, Any]]] = None
    
    opponent_tag: Optional[str] = None
    opponent_name: Optional[str] = None
    opponent_trophies: Optional[int] = None
    opponent_starting_trophies: Optional[int] = None
    opponent_trophy_change: Optional[int] = None
    opponent_crowns: Optional[int] = None
    opponent_deck: Optional[List[Dict[str, Any]]] = None
    
    result: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class BattleAnalysisResponse(BaseModel):
    battle_id: Optional[str] = None
    player_tag: str
    result: Optional[str] = None
    performance_metrics: Dict[str, float]
    recommendations: List[str]

class CardResponse(BaseModel):
    id: int
    name: str
    card_id: Optional[int] = None
    max_level: Optional[int] = None
    max_evolution_level: Optional[int] = None
    elixir_cost: Optional[int] = None
    type: Optional[str] = None
    rarity: Optional[str] = None
    arena: Optional[int] = None
    description: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True
