from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from app.core.database import get_db
from app.core.data_validator import DataValidator
from app.services.clash_royale_service import ClashRoyaleService
from app.ml.predictor import WinPredictor
from app.models.prediction import Prediction
from app.schemas.prediction import PredictionResponse, PredictionRequest

router = APIRouter()
logger = logging.getLogger(__name__)
clash_service = ClashRoyaleService()
data_validator = DataValidator()

# Global predictor instance
_predictor_instance = None

async def get_predictor() -> WinPredictor:
    """Get or create predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        try:
            _predictor_instance = WinPredictor()
            await _predictor_instance.initialize()
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            # Create a minimal fallback predictor
            _predictor_instance = WinPredictor()
            _predictor_instance.is_ready = True
            _predictor_instance.model_version = "v3.0-PCI-RL-Fallback"
    return _predictor_instance

@router.post("/predict")
async def predict_win_probability(
    request: PredictionRequest,
    predictor: WinPredictor = Depends(get_predictor),
    db: Session = Depends(get_db)
):
    """
    Predict win probability for a player using Kaggle-trained models
    
    This endpoint uses advanced ML models trained on 37.9M real matches
    to predict the likelihood of a player winning their next battle based 
    on their stats, current deck, Player Consistency Index (PCI), and battle context.
    """
    try:
        # Extract player data from request
        player_data = {
            "trophies": request.player_trophies,
            "expLevel": request.player_level,
            "wins": request.player_wins or 0,
            "losses": request.player_losses or 0,
            "currentDeck": request.current_deck if request.current_deck else []
        }
        
        logger.info(f"Player data for prediction: {player_data}")
        logger.info(f"Current deck: {player_data['currentDeck']}")
        
        # Validate and sanitize player data
        validated_player_data, is_valid = data_validator.validate_data(player_data, 'player_data')
        if not is_valid:
            logger.warning("Player data validation failed, using validated data with fixes")
        player_data = validated_player_data
        
        # Add opponent data if provided
        opponent_data = None
        if request.opponent_trophies or request.opponent_level:
            opponent_data = {
                "trophies": request.opponent_trophies or 0,
                "expLevel": request.opponent_level or 1
            }
        
        # Add battle context if provided
        battle_context = None
        if request.game_mode or request.arena_id:
            battle_context = {
                "gameMode": request.game_mode,
                "arena": {"id": request.arena_id} if request.arena_id else None
            }
        
        # Make prediction with enhanced model
        result = await predictor.predict(
            player_data=player_data,
            opponent_data=opponent_data,
            battle_context=battle_context
        )
        
        logger.info(f"Prediction result keys: {list(result.keys())}")
        logger.info(f"Strategic analysis fields present: {[k for k in result.keys() if 'tactic' in k or 'card' in k or 'strategy' in k or 'meta' in k or 'pci' in k]}")
        
        # Enhanced response with PCI, strategic analysis, and card suggestions
        response_data = {
            "win_probability": result["win_probability"],
            "confidence": result["confidence"],
            "model_version": result["model_version"],
            "factors": result.get("influencing_factors", {}),
            "recommendations": result.get("recommendations", [])
        }
        
        # Add enhanced features if available
        if "pci_value" in result:
            response_data["pci_value"] = result["pci_value"]
            response_data["pci_interpretation"] = result.get("pci_interpretation", {})
        
        if "enhanced_analysis" in result:
            response_data["enhanced_analysis"] = result["enhanced_analysis"]
        
        # Add strategic analysis and card suggestions
        if "strategic_analysis" in result:
            response_data["strategic_analysis"] = result["strategic_analysis"]
        
        if "battle_tactics" in result:
            response_data["battle_tactics"] = result["battle_tactics"]
        
        if "card_suggestions" in result:
            response_data["card_suggestions"] = result["card_suggestions"]
        
        if "detailed_card_suggestions" in result:
            response_data["detailed_card_suggestions"] = result["detailed_card_suggestions"]
        
        if "counter_strategies" in result:
            response_data["counter_strategies"] = result["counter_strategies"]
        
        if "meta_insights" in result:
            response_data["meta_insights"] = result["meta_insights"]
        
        logger.info(f"Final response data keys: {list(response_data.keys())}")
        logger.info(f"Response data strategic fields: {[k for k in response_data.keys() if 'tactic' in k or 'card' in k or 'strategy' in k or 'meta' in k or 'pci' in k]}")
        
        # Validate prediction result
        validated_response, is_valid = data_validator.validate_data(response_data, 'prediction_data')
        if not is_valid:
            logger.warning("Prediction result validation failed, using validated data with fixes")
        
        # Log validation statistics
        validation_stats = data_validator.get_validation_stats()
        logger.info(f"Validation stats: {validation_stats}")
        
        # Return the validated response
        return validated_response
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{player_tag}/predictions", response_model=List[PredictionResponse])
async def get_player_predictions(
    player_tag: str,
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get prediction history for a player"""
    try:
        clean_tag = player_tag.replace("#", "").upper()
        
        predictions = db.query(Prediction)\
            .filter(Prediction.player_tag == clean_tag)\
            .order_by(Prediction.created_at.desc())\
            .limit(limit)\
            .all()
        
        return [PredictionResponse.from_orm(pred) for pred in predictions]
        
    except Exception as e:
        logger.error(f"Error getting predictions for player {player_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{player_tag}/live-prediction")
async def get_live_prediction(
    player_tag: str,
    db: Session = Depends(get_db)
):
    """Get real-time prediction for ongoing battle"""
    try:
        clean_tag = player_tag.replace("#", "").upper()
        
        # Check if player is in an active battle
        battle_data = await clash_service.get_live_battle_data(clean_tag)
        if not battle_data:
            return {
                "message": "Player is not currently in a battle",
                "in_battle": False
            }
        
        # Make live prediction
        prediction = await win_predictor.predict_live(clean_tag, battle_data)
        
        return {
            "player_tag": clean_tag,
            "in_battle": True,
            "prediction": prediction,
            "battle_state": battle_data
        }
        
    except Exception as e:
        logger.error(f"Error getting live prediction for player {player_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{prediction_id}/feedback")
async def submit_prediction_feedback(
    prediction_id: int,
    rating: int = Query(..., ge=1, le=5),
    comment: Optional[str] = None,
    helpful: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Submit feedback for a prediction"""
    try:
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        # Create feedback record
        from app.models.prediction import PredictionFeedback
        
        feedback = PredictionFeedback(
            prediction_id=prediction_id,
            user_rating=rating,
            user_comment=comment,
            was_helpful=helpful
        )
        
        db.add(feedback)
        db.commit()
        
        return {
            "message": "Feedback submitted successfully",
            "prediction_id": prediction_id
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback for prediction {prediction_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/model/metrics")
async def get_model_metrics(db: Session = Depends(get_db)):
    """Get current model performance metrics"""
    try:
        from app.models.prediction import ModelMetrics
        
        latest_metrics = db.query(ModelMetrics)\
            .order_by(ModelMetrics.created_at.desc())\
            .first()
        
        if not latest_metrics:
            return {"message": "No model metrics available"}
        
        return {
            "model_version": latest_metrics.model_version,
            "accuracy": latest_metrics.accuracy,
            "precision": latest_metrics.precision,
            "recall": latest_metrics.recall,
            "f1_score": latest_metrics.f1_score,
            "auc_roc": latest_metrics.auc_roc,
            "total_predictions": latest_metrics.total_predictions,
            "correct_predictions": latest_metrics.correct_predictions,
            "feature_importance": latest_metrics.feature_importance,
            "last_updated": latest_metrics.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
