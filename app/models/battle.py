from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Battle(Base):
    __tablename__ = "battles"
    
    id = Column(Integer, primary_key=True, index=True)
    battle_time = Column(String, nullable=False)
    type = Column(String, nullable=False)  # PvP, tournament, etc.
    is_ladder_tournament = Column(Boolean, default=False)
    arena_id = Column(Integer)
    arena_name = Column(String)
    game_mode_id = Column(Integer)
    game_mode_name = Column(String)
    deck_selection = Column(String)
    
    # Player data
    player_tag = Column(String, ForeignKey("players.tag"), nullable=False)
    player_name = Column(String)
    player_trophies = Column(Integer)
    player_starting_trophies = Column(Integer)
    player_trophy_change = Column(Integer)
    player_crowns = Column(Integer)
    player_king_tower_hp = Column(Integer)
    player_princess_towers_hp = Column(JSON)
    player_deck = Column(JSON)  # Array of cards with levels
    
    # Opponent data
    opponent_tag = Column(String)
    opponent_name = Column(String)
    opponent_trophies = Column(Integer)
    opponent_starting_trophies = Column(Integer)
    opponent_trophy_change = Column(Integer)
    opponent_crowns = Column(Integer)
    opponent_king_tower_hp = Column(Integer)
    opponent_princess_towers_hp = Column(JSON)
    opponent_deck = Column(JSON)  # Array of cards with levels
    
    # Battle result
    result = Column(String)  # win, loss, draw
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="battles")
    predictions = relationship("Prediction", back_populates="battle")

class BattleTimeline(Base):
    __tablename__ = "battle_timelines"
    
    id = Column(Integer, primary_key=True, index=True)
    battle_id = Column(Integer, ForeignKey("battles.id"), nullable=False)
    
    # Timeline data for ML training
    elixir_usage = Column(JSON)  # Time-series elixir data
    card_deployment = Column(JSON)  # Card deployment timeline
    tower_damage = Column(JSON)  # Tower damage over time
    troop_interactions = Column(JSON)  # Troop vs troop interactions
    
    # Calculated metrics
    elixir_efficiency = Column(Float)
    deck_synergy_score = Column(Float)
    defensive_rating = Column(Float)
    offensive_rating = Column(Float)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Card(Base):
    __tablename__ = "cards"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    card_id = Column(Integer, unique=True)
    max_level = Column(Integer)
    max_evolution_level = Column(Integer)
    elixir_cost = Column(Integer)
    type = Column(String)  # troop, spell, building
    rarity = Column(String)  # common, rare, epic, legendary, champion
    arena = Column(Integer)
    description = Column(String)
    
    # Card stats (can vary by level)
    stats = Column(JSON)  # HP, damage, etc. by level
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
