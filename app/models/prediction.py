from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    player_tag = Column(String, ForeignKey("players.tag"), nullable=False)
    battle_id = Column(Integer, ForeignKey("battles.id"), nullable=True)
    
    # Prediction data
    win_probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    
    # Input features used for prediction
    input_features = Column(JSON)
    
    # Influencing factors
    deck_synergy_impact = Column(Float)
    elixir_efficiency_impact = Column(Float)
    opponent_counter_impact = Column(Float)
    player_skill_impact = Column(Float)
    recent_performance_impact = Column(Float)
    
    # Prediction context
    prediction_type = Column(String)  # pre_battle, live_battle, post_battle
    battle_state = Column(JSON)  # Current state when prediction was made
    
    # Actual outcome (filled after battle)
    actual_result = Column(String)  # win, loss, draw
    prediction_accuracy = Column(Float)  # How close was the prediction
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="predictions")
    battle = relationship("Battle", back_populates="predictions")

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String, nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    
    # Prediction distribution
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    
    # Feature importance
    feature_importance = Column(JSON)
    
    # Training data
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    test_samples = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
class PredictionFeedback(Base):
    __tablename__ = "prediction_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    
    # User feedback
    user_rating = Column(Integer)  # 1-5 stars
    user_comment = Column(String)
    was_helpful = Column(Boolean)
    
    # System feedback
    prediction_error = Column(Float)
    confidence_calibration = Column(Float)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
