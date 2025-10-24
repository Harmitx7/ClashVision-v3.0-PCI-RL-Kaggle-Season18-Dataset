from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Clan(Base):
    __tablename__ = "clans"
    
    id = Column(Integer, primary_key=True, index=True)
    tag = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    type = Column(String)  # open, inviteOnly, closed
    score = Column(Integer, default=0)
    required_trophies = Column(Integer, default=0)
    donations_per_week = Column(Integer, default=0)
    member_count = Column(Integer, default=0)
    location_id = Column(Integer)
    location_name = Column(String)
    location_country_code = Column(String)
    badge_id = Column(Integer)
    badge_name = Column(String)
    
    # War stats
    clan_war_trophies = Column(Integer, default=0)
    clan_war_trophy_change = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    members = relationship("Player", back_populates="clan")
    wars = relationship("ClanWar", back_populates="clan")

class ClanWar(Base):
    __tablename__ = "clan_wars"
    
    id = Column(Integer, primary_key=True, index=True)
    clan_tag = Column(String, nullable=False)
    season_id = Column(Integer)
    created_date = Column(String)
    
    # War state
    state = Column(String)  # notInWar, matchmaking, preparation, battle, warDay
    war_end_time = Column(String)
    
    # Participants and results
    participants = Column(JSON)
    standings = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    clan = relationship("Clan", back_populates="wars")
