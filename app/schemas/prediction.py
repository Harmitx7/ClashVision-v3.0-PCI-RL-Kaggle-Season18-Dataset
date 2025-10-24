from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class PredictionRequest(BaseModel):
    # Player data
    player_trophies: Optional[int] = None
    player_level: Optional[int] = None
    player_wins: Optional[int] = None
    player_losses: Optional[int] = None
    current_deck: Optional[List[Dict[str, Any]]] = None
    
    # Opponent data
    opponent_trophies: Optional[int] = None
    opponent_level: Optional[int] = None
    
    # Battle context
    game_mode: Optional[str] = None
    arena_id: Optional[int] = None
    
    # Legacy fields for backward compatibility
    opponent_data: Optional[Dict[str, Any]] = None
    battle_context: Optional[Dict[str, Any]] = None
    deck_data: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    id: int
    player_tag: str
    battle_id: Optional[int] = None
    win_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    
    # Influencing factors
    deck_synergy_impact: Optional[float] = None
    elixir_efficiency_impact: Optional[float] = None
    opponent_counter_impact: Optional[float] = None
    player_skill_impact: Optional[float] = None
    recent_performance_impact: Optional[float] = None
    
    prediction_type: str
    actual_result: Optional[str] = None
    prediction_accuracy: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class LivePredictionResponse(BaseModel):
    player_tag: str
    win_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    
    influencing_factors: Dict[str, float]
    battle_state: Dict[str, Any]
    recommendations: Optional[List[str]] = None
    
    timestamp: datetime

class ModelMetricsResponse(BaseModel):
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    total_predictions: int
    correct_predictions: int
    feature_importance: Dict[str, float]
    last_updated: datetime
