from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Player(Base):
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)
    tag = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    trophies = Column(Integer, default=0)
    best_trophies = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    battle_count = Column(Integer, default=0)
    three_crown_wins = Column(Integer, default=0)
    cards_found = Column(Integer, default=0)
    favorite_card = Column(String)
    total_donations = Column(Integer, default=0)
    clan_tag = Column(String, ForeignKey("clans.tag"))
    clan_role = Column(String)
    arena_id = Column(Integer)
    arena_name = Column(String)
    league_id = Column(Integer)
    league_name = Column(String)
    exp_level = Column(Integer, default=1)
    exp_points = Column(Integer, default=0)
    star_points = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_seen = Column(DateTime(timezone=True))
    
    # Relationships
    clan = relationship("Clan", back_populates="members")
    battles = relationship("Battle", back_populates="player")
    predictions = relationship("Prediction", back_populates="player")

class PlayerCard(Base):
    __tablename__ = "player_cards"
    
    id = Column(Integer, primary_key=True, index=True)
    player_tag = Column(String, ForeignKey("players.tag"), nullable=False)
    card_name = Column(String, nullable=False)
    level = Column(Integer, nullable=False)
    max_level = Column(Integer, nullable=False)
    count = Column(Integer, default=0)
    star_level = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class PlayerStats(Base):
    __tablename__ = "player_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    player_tag = Column(String, ForeignKey("players.tag"), nullable=False)
    
    # Performance metrics
    win_rate = Column(Float, default=0.0)
    average_elixir_cost = Column(Float, default=0.0)
    favorite_deck = Column(JSON)
    recent_performance = Column(JSON)  # Last 25 battles
    
    # Calculated stats
    skill_rating = Column(Float, default=0.0)
    consistency_score = Column(Float, default=0.0)
    deck_mastery = Column(Float, default=0.0)
    
    # Metadata
    calculated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
