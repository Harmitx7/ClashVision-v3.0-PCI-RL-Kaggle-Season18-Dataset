from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ClanBase(BaseModel):
    tag: str
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    score: int = 0
    required_trophies: int = 0
    donations_per_week: int = 0
    member_count: int = 0
    location_id: Optional[int] = None
    location_name: Optional[str] = None
    location_country_code: Optional[str] = None
    badge_id: Optional[int] = None
    badge_name: Optional[str] = None
    clan_war_trophies: int = 0
    clan_war_trophy_change: int = 0

class ClanResponse(ClanBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class ClanWarResponse(BaseModel):
    id: int
    clan_tag: str
    season_id: Optional[int] = None
    created_date: Optional[str] = None
    state: Optional[str] = None
    war_end_time: Optional[str] = None
    participants: Optional[Dict[str, Any]] = None
    standings: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class ClanMemberResponse(BaseModel):
    tag: str
    name: str
    role: Optional[str] = None
    last_seen: Optional[str] = None
    exp_level: int = 1
    trophies: int = 0
    arena_id: Optional[int] = None
    arena_name: Optional[str] = None
    clan_rank: Optional[int] = None
    previous_clan_rank: Optional[int] = None
    donations: int = 0
    donations_received: int = 0
    clan_chest_points: Optional[int] = None

class ClanPerformanceResponse(BaseModel):
    clan_tag: str
    clan_name: str
    member_count: int
    total_trophies: int
    average_trophies: float
    total_donations: int
    average_donations: float
    clan_war_trophies: int
    required_trophies: int
    member_distribution: Dict[str, int]
    activity_score: float
