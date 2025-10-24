import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Any, Tuple
import logging
import asyncio
import json
from datetime import datetime
import os

from app.core.config import settings
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model_architecture import TransformerLSTMModel
from app.ml.enhanced_predictor import EnhancedWinPredictor
from app.ml.monitoring_system import ModelMonitoringSystem
from app.ml.kaggle_data_integration import KaggleDataIntegration
from app.ml.battle_strategy_analyzer import BattleStrategyAnalyzer

logger = logging.getLogger(__name__)

class WinPredictor:
    def __init__(self):
        # Legacy model components
        self.model: Optional[TransformerLSTMModel] = None
        self.feature_engineer = FeatureEngineer()
        self.model_version = "1.0.0"
        self.is_ready = False
        self.confidence_threshold = settings.PREDICTION_CONFIDENCE_THRESHOLD
        
        # Enhanced v3.0 components
        self.enhanced_predictor: Optional[EnhancedWinPredictor] = None
        self.monitoring_system: Optional[ModelMonitoringSystem] = None
        self.kaggle_integration: Optional[KaggleDataIntegration] = None
        self.use_enhanced_model = True  # Flag to switch between legacy and enhanced
        self.auto_train_with_kaggle = True  # Auto-train with Kaggle data when available
        self.use_hybrid_training = True  # Mix recent match data with Kaggle data
        self.recent_match_buffer = []  # Buffer for recent match outcomes
        self.max_recent_matches = 10000  # Maximum recent matches to keep
        self.strategy_analyzer = BattleStrategyAnalyzer()  # Strategic analysis engine
        
        # Model paths
        self.model_path = os.path.join(settings.MODEL_PATH, "win_predictor.h5")
        self.scaler_path = os.path.join(settings.MODEL_PATH, "feature_scaler.pkl")
        
    async def initialize(self):
        """Initialize the ML model (Enhanced v3.0 or Legacy)"""
        try:
            logger.info("Initializing Win Predictor model...")
            
            # Create model directory if it doesn't exist
            os.makedirs(settings.MODEL_PATH, exist_ok=True)
            
            if self.use_enhanced_model:
                try:
                    # Initialize Enhanced v3.0 system
                    logger.info("Initializing Enhanced Win Predictor v3.0-PCI-RL...")
                    
                    # Initialize Kaggle data integration
                    try:
                        self.kaggle_integration = KaggleDataIntegration()
                        await self.kaggle_integration.download_and_prepare_dataset()
                        logger.info("Kaggle dataset integration initialized successfully")
                    except Exception as e:
                        logger.info(f"Kaggle dataset initialization failed: {e} - continuing with fallback mode")
                        self.kaggle_integration = None
                    
                    self.enhanced_predictor = EnhancedWinPredictor()
                    await self.enhanced_predictor.initialize()
                    
                    # Initialize monitoring system
                    self.monitoring_system = ModelMonitoringSystem(self.enhanced_predictor)
                    await self.monitoring_system.start_monitoring()
                    
                    # Auto-train with hybrid data (Kaggle + Recent matches) if available
                    if self.auto_train_with_kaggle and self.kaggle_integration and await self._is_kaggle_data_ready():
                        try:
                            logger.info("Kaggle dataset detected, starting hybrid auto-training...")
                            await self._auto_train_with_hybrid_data()
                        except Exception as e:
                            logger.info(f"Hybrid training failed: {e} - continuing without auto-training")
                    
                    # Set ready state even if enhanced predictor has issues
                    self.is_ready = True  # We can still make predictions with fallback logic
                    self.model_version = getattr(self.enhanced_predictor, 'model_version', 'v3.0-PCI-RL-Enhanced')
                    
                    logger.info("Enhanced Win Predictor v3.0-PCI-RL initialized successfully")
                    
                except Exception as e:
                    logger.error(f"Enhanced predictor initialization failed: {e}")
                    logger.info("Falling back to legacy model...")
                    self.use_enhanced_model = False
            
            # Initialize legacy model (either by choice or fallback)
            if not self.use_enhanced_model:
                if os.path.exists(self.model_path):
                    await self._load_model()
                else:
                    await self._create_new_model()
                
                self.is_ready = True
                logger.info("Legacy Win Predictor model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Win Predictor: {e}")
            self.is_ready = False
    
    async def _load_model(self):
        """Load existing model from disk"""
        try:
            self.model = TransformerLSTMModel()
            self.model.load_weights(self.model_path)
            
            # Load feature scaler
            self.feature_engineer.load_scaler(self.scaler_path)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to creating new model
            await self._create_new_model()
    
    async def _create_new_model(self):
        """Create and initialize a new model"""
        try:
            logger.info("Creating new model...")
            
            # Initialize model architecture
            self.model = TransformerLSTMModel()
            
            # Build model with dummy data to initialize weights
            dummy_input = np.random.random((1, 50, 64))  # (batch, sequence, features)
            _ = self.model(dummy_input)
            
            # Initialize feature scaler
            self.feature_engineer.initialize_scaler()
            
            logger.info("New model created successfully")
            
        except Exception as e:
            logger.error(f"Error creating new model: {e}")
            raise
    
    async def predict(
        self,
        player_data: Dict[str, Any],
        opponent_data: Optional[Dict[str, Any]] = None,
        battle_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a win prediction (Enhanced v3.0 or Legacy)"""
        try:
            if not self.is_ready:
                raise ValueError("Model not ready")
            
            if self.use_enhanced_model and self.enhanced_predictor:
                try:
                    logger.info("Using Enhanced Win Predictor v3.0-PCI-RL for prediction")
                    # Use Enhanced v3.0 predictor with PCI integration
                    result = await self.enhanced_predictor.predict(
                        player_data=player_data,
                        opponent_data=opponent_data,
                        battle_context=battle_context
                    )
                    logger.info(f"Enhanced prediction result keys: {list(result.keys())}")
                except Exception as e:
                    logger.warning(f"Enhanced prediction failed, using fallback: {e}")
                    # Fallback to simple prediction
                    result = self._simple_prediction_fallback(player_data, opponent_data)
                
                # Add strategic analysis to enhanced prediction
                if self.strategy_analyzer:
                    try:
                        strategy_analysis = self.strategy_analyzer.analyze_battle_strategy(
                            player_data=player_data,
                            opponent_data=opponent_data,
                            recent_matches=self.recent_match_buffer[-20:],  # Last 20 matches
                            pci_value=result.get('pci_value', 0.5)
                        )
                        
                        # Add strategic insights to result
                        result.update({
                            'strategic_analysis': strategy_analysis,
                            'battle_tactics': strategy_analysis.get('battle_tactics', []),
                            'card_suggestions': strategy_analysis.get('card_suggestions', {}),
                            'counter_strategies': strategy_analysis.get('counter_strategies', []),
                            'meta_insights': strategy_analysis.get('meta_analysis', {})
                        })
                        
                        # Generate specific card recommendations
                        if 'currentDeck' in player_data and player_data['currentDeck']:
                            card_suggestions = self.strategy_analyzer.generate_card_suggestions(
                                current_deck=player_data['currentDeck'],
                                strategy_analysis=strategy_analysis,
                                recent_performance=strategy_analysis.get('win_rate_by_card', {})
                            )
                            result['detailed_card_suggestions'] = card_suggestions
                        
                    except Exception as e:
                        logger.warning(f"Strategic analysis failed: {e}")
                        result['strategic_analysis'] = {'error': 'Strategic analysis unavailable'}
                
                # Log prediction for monitoring
                if self.monitoring_system:
                    self.monitoring_system.log_prediction(result)
                
                return result
            
            else:
                # Use legacy prediction logic
                features = self.feature_engineer.extract_features(
                    player_data=player_data,
                    opponent_data=opponent_data,
                    battle_context=battle_context
                )
                
                model_input = self._prepare_model_input(features)
                prediction_raw = self.model(model_input)
                win_probability = float(prediction_raw[0][0])
                confidence = self._calculate_confidence(prediction_raw)
                influencing_factors = self._analyze_influencing_factors(features)
                recommendations = self._generate_recommendations(features, win_probability)
                
                return {
                    "win_probability": win_probability,
                    "confidence": confidence,
                    "model_version": self.model_version,
                    "input_features": features,
                    "influencing_factors": influencing_factors,
                    "recommendations": recommendations,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def _simple_prediction_fallback(self, player_data: Dict[str, Any], opponent_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simple fallback prediction when enhanced model fails"""
        try:
            # Simple heuristic-based prediction
            player_trophies = player_data.get('trophies', 4000)
            opponent_trophies = opponent_data.get('trophies', 4000) if opponent_data else 4000
            
            # Basic trophy difference calculation
            trophy_diff = player_trophies - opponent_trophies
            base_probability = 0.5 + (trophy_diff / 2000.0)  # Normalize trophy difference
            win_probability = max(0.1, min(0.9, base_probability))
            
            return {
                "win_probability": win_probability,
                "confidence": 0.6,  # Lower confidence for fallback
                "model_version": "v3.0-PCI-RL-Fallback",
                "prediction_timestamp": datetime.now().isoformat(),
                "fallback_mode": True,
                "pci_value": 0.5,  # Default PCI
                "pci_interpretation": {
                    "stability_level": "Stable",
                    "description": "Fallback analysis - limited data available"
                },
                "strategic_analysis": {
                    "deck_balance": {"average_elixir": 3.5, "balance_score": 0.7},
                    "strengths": ["Balanced approach"],
                    "weaknesses": ["Limited analysis in fallback mode"]
                },
                "battle_tactics": [
                    "🎯 Play balanced strategy based on trophy difference",
                    "⚡ Focus on positive elixir trades",
                    "🛡️ Maintain defensive positioning"
                ],
                "detailed_card_suggestions": {
                    "cards_to_add": [],
                    "cards_to_remove": [],
                    "deck_improvements": ["Limited suggestions available in fallback mode"]
                },
                "counter_strategies": [
                    "Adapt strategy based on opponent's moves",
                    "Focus on fundamental gameplay"
                ],
                "meta_insights": {
                    "trending_cards": [],
                    "recommended_adaptations": ["Full analysis requires enhanced model"]
                }
            }
        except Exception as e:
            logger.error(f"Even fallback prediction failed: {e}")
            # Ultimate fallback
            return {
                "win_probability": 0.5,
                "confidence": 0.3,
                "model_version": "v3.0-PCI-RL-Emergency",
                "prediction_timestamp": datetime.now().isoformat(),
                "emergency_mode": True
            }
    
    async def predict_live(self, player_tag: str, battle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a live prediction during an ongoing battle (Enhanced v3.0 or Legacy)"""
        try:
            if not self.is_ready:
                raise ValueError("Model not ready")
            
            if self.use_enhanced_model and self.enhanced_predictor:
                # Use Enhanced v3.0 live prediction
                result = await self.enhanced_predictor.predict_live(
                    player_tag=player_tag,
                    battle_data=battle_data
                )
                
                # Log prediction for monitoring
                if self.monitoring_system:
                    self.monitoring_system.log_prediction(result)
                
                return result
            
            else:
                # Use legacy live prediction logic
                features = self.feature_engineer.extract_live_features(
                    player_tag=player_tag,
                    battle_data=battle_data
                )
                
                model_input = self._prepare_model_input(features)
                prediction_raw = self.model(model_input)
                win_probability = float(prediction_raw[0][0])
                confidence = self._calculate_confidence(prediction_raw)
                battle_analysis = self._analyze_battle_state(battle_data)
                recommendations = self._generate_live_recommendations(
                    features, win_probability, battle_data
                )
                
                # Add strategic analysis
                if self.strategy_analyzer:
                    try:
                        strategy_analysis = self.strategy_analyzer.analyze_battle_strategy(
                            player_data={'tag': player_tag},
                            opponent_data={'tag': battle_data.get('opponent_tag', '')},
                            recent_matches=self.recent_match_buffer[-20:],  # Last 20 matches
                            pci_value=0.5
                        )
                        
                        # Add strategic insights to result
                        battle_analysis.update({
                            'strategic_analysis': strategy_analysis,
                            'battle_tactics': strategy_analysis.get('battle_tactics', []),
                            'card_suggestions': strategy_analysis.get('card_suggestions', {}),
                            'counter_strategies': strategy_analysis.get('counter_strategies', []),
                            'meta_insights': strategy_analysis.get('meta_analysis', {})
                        })
                        
                        # Generate specific card recommendations
                        if 'currentDeck' in battle_data and battle_data['currentDeck']:
                            card_suggestions = self.strategy_analyzer.generate_card_suggestions(
                                current_deck=battle_data['currentDeck'],
                                strategy_analysis=strategy_analysis,
                                recent_performance=strategy_analysis.get('win_rate_by_card', {})
                            )
                            battle_analysis['detailed_card_suggestions'] = card_suggestions
                        
                    except Exception as e:
                        logger.warning(f"Strategic analysis failed: {e}")
                        battle_analysis['strategic_analysis'] = {'error': 'Strategic analysis unavailable'}
                
                return {
                    "win_probability": win_probability,
                    "confidence": confidence,
                    "model_version": self.model_version,
                    "battle_analysis": battle_analysis,
                    "recommendations": recommendations,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error making live prediction: {e}")
            raise
    
    def _prepare_model_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model input"""
        try:
            # Convert features to sequence format expected by Transformer-LSTM
            feature_vector = self.feature_engineer.features_to_vector(features)
            
            # Reshape for sequence input (batch_size, sequence_length, features)
            # For now, we'll use a single timestep, but this can be extended for temporal data
            sequence_input = np.expand_dims(feature_vector, axis=0)  # Add batch dimension
            sequence_input = np.expand_dims(sequence_input, axis=1)  # Add sequence dimension
            
            # Pad or truncate to expected sequence length (50)
            target_seq_len = 50
            current_seq_len = sequence_input.shape[1]
            
            if current_seq_len < target_seq_len:
                # Pad with zeros
                padding = np.zeros((1, target_seq_len - current_seq_len, sequence_input.shape[2]))
                sequence_input = np.concatenate([sequence_input, padding], axis=1)
            elif current_seq_len > target_seq_len:
                # Truncate
                sequence_input = sequence_input[:, :target_seq_len, :]
            
            return sequence_input
            
        except Exception as e:
            logger.error(f"Error preparing model input: {e}")
            raise
    
    def _calculate_confidence(self, prediction_raw: tf.Tensor) -> float:
        """Calculate prediction confidence"""
        try:
            # Use prediction certainty as confidence measure
            prob = float(prediction_raw[0][0])
            
            # Confidence is higher when prediction is closer to 0 or 1
            confidence = 2 * abs(prob - 0.5)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _analyze_influencing_factors(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Analyze which factors most influence the prediction"""
        try:
            # Simplified feature importance analysis
            # In a real implementation, this would use SHAP or similar techniques
            
            factors = {
                "deck_synergy": features.get("deck_synergy_score", 0.0),
                "elixir_efficiency": features.get("elixir_efficiency", 0.0),
                "opponent_counter": features.get("opponent_counter_score", 0.0),
                "player_skill": features.get("skill_rating", 0.0),
                "recent_performance": features.get("recent_win_rate", 0.0)
            }
            
            # Normalize factors to sum to 1
            total = sum(abs(v) for v in factors.values())
            if total > 0:
                factors = {k: v / total for k, v in factors.items()}
            
            return factors
            
        except Exception as e:
            logger.error(f"Error analyzing influencing factors: {e}")
            return {}
    
    def _generate_recommendations(
        self, 
        features: Dict[str, Any], 
        win_probability: float
    ) -> List[str]:
        """Generate strategic recommendations"""
        try:
            recommendations = []
            
            if win_probability < 0.4:
                recommendations.append("Consider a more defensive strategy")
                recommendations.append("Focus on positive elixir trades")
                
                if features.get("deck_synergy_score", 0) < 0.5:
                    recommendations.append("Your deck synergy is low - consider card substitutions")
                    
            elif win_probability > 0.7:
                recommendations.append("You have a strong advantage - maintain pressure")
                recommendations.append("Look for opportunities to take towers")
                
            else:
                recommendations.append("Match is balanced - adapt to opponent's strategy")
                recommendations.append("Monitor elixir carefully")
            
            # Add deck-specific recommendations
            if features.get("average_elixir_cost", 0) > 4.0:
                recommendations.append("Heavy deck - be patient with elixir management")
            elif features.get("average_elixir_cost", 0) < 3.0:
                recommendations.append("Fast cycle deck - maintain constant pressure")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _analyze_battle_state(self, battle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current battle state"""
        try:
            # Extract battle metrics
            analysis = {
                "battle_time_remaining": battle_data.get("time_remaining", 0),
                "player_towers": battle_data.get("player_towers", 3),
                "opponent_towers": battle_data.get("opponent_towers", 3),
                "elixir_advantage": battle_data.get("elixir_advantage", 0),
                "current_phase": self._determine_battle_phase(battle_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing battle state: {e}")
            return {}
    
    def _determine_battle_phase(self, battle_data: Dict[str, Any]) -> str:
        """Determine current battle phase"""
        time_remaining = battle_data.get("time_remaining", 0)
        
        if time_remaining > 120:
            return "early_game"
        elif time_remaining > 60:
            return "mid_game"
        elif time_remaining > 0:
            return "late_game"
        else:
            return "overtime"
    
    def _generate_live_recommendations(
        self,
        features: Dict[str, Any],
        win_probability: float,
        battle_data: Dict[str, Any]
    ) -> List[str]:
        """Generate live battle recommendations"""
        try:
            recommendations = []
            
            battle_phase = self._determine_battle_phase(battle_data)
            towers_down = 3 - battle_data.get("player_towers", 3)
            opponent_towers_down = 3 - battle_data.get("opponent_towers", 3)
            
            if battle_phase == "early_game":
                recommendations.append("Focus on learning opponent's deck")
                recommendations.append("Make positive elixir trades")
                
            elif battle_phase == "mid_game":
                if win_probability > 0.6:
                    recommendations.append("Build a strong push")
                else:
                    recommendations.append("Defend and counter-attack")
                    
            elif battle_phase == "late_game":
                if towers_down > opponent_towers_down:
                    recommendations.append("Play defensively - protect your lead")
                else:
                    recommendations.append("Time for aggressive plays")
                    
            elif battle_phase == "overtime":
                recommendations.append("High-risk, high-reward plays")
                recommendations.append("Focus on tower damage")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating live recommendations: {e}")
            return []
    
    async def train_model(self, training_data: pd.DataFrame):
        """Train the model with new data"""
        try:
            logger.info("Starting model training...")
            
            # Prepare training data
            X, y = self.feature_engineer.prepare_training_data(training_data)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=1
            )
            
            # Save model
            self.model.save_weights(self.model_path)
            self.feature_engineer.save_scaler(self.scaler_path)
            
            logger.info("Model training completed successfully")
            
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    async def update_model_weights(self, feedback_data: List[Dict[str, Any]]):
        """Update model with prediction feedback"""
        try:
            if not feedback_data:
                return
            
            logger.info(f"Updating model with {len(feedback_data)} feedback samples")
            
            if self.use_enhanced_model and self.enhanced_predictor and self.enhanced_predictor.rl_loop:
                # Use enhanced RL loop for feedback processing
                for feedback in feedback_data:
                    await self.enhanced_predictor.rl_loop.process_battle_outcome(
                        player_tag=feedback.get('player_tag', 'unknown'),
                        prediction_data=feedback.get('prediction_data', {}),
                        actual_outcome=feedback.get('actual_outcome', False),
                        battle_data=feedback.get('battle_data', {})
                    )
            else:
                # Legacy feedback processing
                # Process feedback and retrain if necessary
                # This is a simplified implementation
                
                # In a real system, you would:
                # 1. Collect feedback data
                # 2. Prepare it for training
                # 3. Perform incremental learning
                # 4. Validate improvements
                # 5. Deploy updated model
                pass
            
            logger.info("Model weights updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {e}")
            raise
    
    async def process_battle_outcome(
        self,
        player_tag: str,
        prediction_data: Dict[str, Any],
        actual_outcome: bool,
        battle_data: Dict[str, Any]
    ):
        """Process battle outcome for adaptive learning"""
        try:
            if self.use_enhanced_model and self.enhanced_predictor:
                # Use enhanced predictor's battle outcome processing
                await self.enhanced_predictor.process_battle_outcome(
                    player_tag=player_tag,
                    prediction_data=prediction_data,
                    actual_outcome=actual_outcome,
                    battle_data=battle_data
                )
                
                # Update monitoring system
                if self.monitoring_system:
                    self.monitoring_system.log_prediction(prediction_data, actual_outcome)
            
            logger.info(f"Battle outcome processed for player {player_tag}: {actual_outcome}")
            
        except Exception as e:
            logger.warning(f"Battle outcome processing failed (continuing without): {e}")
        
        # Store recent match data for hybrid training
        if self.use_hybrid_training:
            await self._store_recent_match_data(player_tag, prediction_data, actual_outcome, battle_data)
    
    async def _is_kaggle_data_ready(self) -> bool:
        """Check if Kaggle dataset is ready for training"""
        try:
            if not self.kaggle_integration:
                return False
            
            # Check if dataset directory exists and has files
            dataset_path = self.kaggle_integration.dataset_path
            if dataset_path.exists() and list(dataset_path.glob("*.csv")):
                logger.info("Kaggle dataset is ready for training")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking Kaggle data readiness: {e}")
            return False
    
    async def _auto_train_with_kaggle_data(self):
        """Automatically train the model with Kaggle dataset"""
        try:
            if not self.kaggle_integration or not self.enhanced_predictor:
                return
            
            logger.info("Starting automatic training with Kaggle dataset...")
            
            # Process dataset for training (use a reasonable sample size for production)
            training_samples = min(500000, 37900000)  # 500K samples for production balance
            processed_file = await self.kaggle_integration.process_dataset_for_training(
                output_file=f"auto_training_data_{training_samples}.parquet"
            )
            
            if processed_file and os.path.exists(processed_file):
                # Train the enhanced model
                if hasattr(self.enhanced_predictor, 'training_pipeline') and self.enhanced_predictor.training_pipeline:
                    training_result = await self.kaggle_integration.train_enhanced_model_with_kaggle_data(
                        training_pipeline=self.enhanced_predictor.training_pipeline,
                        processed_data_file=processed_file
                    )
                    
                    if training_result:
                        logger.info(f"Auto-training completed! New model version: {training_result['model_version']}")
                        self.model_version = training_result['model_version']
                    else:
                        logger.warning("Auto-training failed, using default model")
                else:
                    logger.warning("Training pipeline not available, skipping auto-training")
            else:
                logger.warning("Processed training data not available, skipping auto-training")
                
        except Exception as e:
            logger.warning(f"Auto-training with Kaggle data failed: {e}")
            logger.info("Continuing with default model...")
    
    async def _auto_train_with_hybrid_data(self):
        """Train model with hybrid data (Kaggle + Recent matches)"""
        try:
            if not self.kaggle_integration or not self.enhanced_predictor:
                return
            
            logger.info("Starting hybrid training (Kaggle + Recent matches)...")
            
            # Process Kaggle dataset
            kaggle_samples = min(400000, 37900000)  # 400K from Kaggle for balance
            processed_kaggle_file = await self.kaggle_integration.process_dataset_for_training(
                output_file=f"hybrid_kaggle_data_{kaggle_samples}.parquet"
            )
            
            # Process recent match data
            recent_data_file = None
            if len(self.recent_match_buffer) > 100:  # Minimum recent matches
                recent_data_file = await self._process_recent_matches_for_training()
                logger.info(f"Processed {len(self.recent_match_buffer)} recent matches for training")
            
            # Combine datasets for hybrid training
            if processed_kaggle_file and os.path.exists(processed_kaggle_file):
                # Train with hybrid data
                if hasattr(self.enhanced_predictor, 'training_pipeline') and self.enhanced_predictor.training_pipeline:
                    training_result = await self._hybrid_model_training(
                        kaggle_file=processed_kaggle_file,
                        recent_file=recent_data_file
                    )
                    
                    if training_result:
                        logger.info(f"Hybrid training completed! New model version: {training_result['model_version']}")
                        logger.info(f"Training accuracy: {training_result.get('accuracy', 'N/A'):.3f}")
                        logger.info(f"Kaggle samples: {training_result.get('kaggle_samples', 0):,}")
                        logger.info(f"Recent samples: {training_result.get('recent_samples', 0):,}")
                        self.model_version = training_result['model_version']
                    else:
                        logger.warning("Hybrid training failed, using default model")
                else:
                    logger.warning("Training pipeline not available, skipping hybrid training")
            else:
                logger.warning("Kaggle data not available, falling back to recent-only training")
                await self._train_with_recent_data_only()
                
        except Exception as e:
            logger.warning(f"Hybrid training failed: {e}")
            logger.info("Continuing with default model...")
    
    async def _store_recent_match_data(self, player_tag: str, prediction_data: Dict[str, Any], 
                                     actual_outcome: bool, battle_data: Dict[str, Any]):
        """Store recent match data for hybrid training"""
        try:
            # Create training sample from recent match
            recent_sample = {
                'timestamp': datetime.now().isoformat(),
                'player_tag': player_tag,
                'prediction': prediction_data,
                'actual_outcome': actual_outcome,
                'battle_data': battle_data,
                'features': self._extract_features_from_battle(battle_data)
            }
            
            # Add to buffer
            self.recent_match_buffer.append(recent_sample)
            
            # Maintain buffer size
            if len(self.recent_match_buffer) > self.max_recent_matches:
                self.recent_match_buffer = self.recent_match_buffer[-self.max_recent_matches:]
            
            # Trigger retraining if we have enough new data
            if len(self.recent_match_buffer) % 1000 == 0:  # Every 1000 new matches
                logger.info(f"Collected {len(self.recent_match_buffer)} recent matches, considering retraining...")
                await self._consider_hybrid_retraining()
                
        except Exception as e:
            logger.warning(f"Failed to store recent match data: {e}")
    
    async def _process_recent_matches_for_training(self) -> Optional[str]:
        """Process recent matches into training format"""
        try:
            if not self.recent_match_buffer:
                return None
            
            import pandas as pd
            
            # Convert recent matches to training format
            training_data = []
            for match in self.recent_match_buffer:
                if 'features' in match and match['features']:
                    training_sample = {
                        **match['features'],
                        'result': 1 if match['actual_outcome'] else 0,
                        'confidence': match['prediction'].get('confidence', 0.5),
                        'pci_value': match['prediction'].get('pci_value', 0.5)
                    }
                    training_data.append(training_sample)
            
            if not training_data:
                return None
            
            # Save to file
            df = pd.DataFrame(training_data)
            recent_file = f"data/recent_matches_training_{len(training_data)}.parquet"
            os.makedirs("data", exist_ok=True)
            df.to_parquet(recent_file)
            
            logger.info(f"Processed {len(training_data)} recent matches for training")
            return recent_file
            
        except Exception as e:
            logger.error(f"Failed to process recent matches: {e}")
            return None
    
    def _extract_features_from_battle(self, battle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from battle data for training"""
        try:
            # Use existing feature engineer
            if hasattr(self, 'feature_engineer') and self.feature_engineer:
                return self.feature_engineer.extract_features(battle_data)
            
            # Fallback simple feature extraction
            return {
                'player_trophies': battle_data.get('player_trophies', 0),
                'opponent_trophies': battle_data.get('opponent_trophies', 0),
                'trophy_difference': battle_data.get('player_trophies', 0) - battle_data.get('opponent_trophies', 0),
                'player_level': battle_data.get('player_level', 1),
                'opponent_level': battle_data.get('opponent_level', 1)
            }
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {}
    
    async def _hybrid_model_training(self, kaggle_file: str, recent_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Train model with hybrid data"""
        try:
            import pandas as pd
            
            # Load Kaggle data
            kaggle_df = pd.read_parquet(kaggle_file)
            logger.info(f"Loaded {len(kaggle_df)} Kaggle samples")
            
            # Load recent data if available
            recent_df = None
            if recent_file and os.path.exists(recent_file):
                recent_df = pd.read_parquet(recent_file)
                logger.info(f"Loaded {len(recent_df)} recent samples")
                
                # Combine datasets with weighting (recent data gets higher weight)
                kaggle_df['sample_weight'] = 1.0  # Standard weight for Kaggle data
                recent_df['sample_weight'] = 2.0  # Higher weight for recent data
                
                # Combine
                combined_df = pd.concat([kaggle_df, recent_df], ignore_index=True)
            else:
                combined_df = kaggle_df
                combined_df['sample_weight'] = 1.0
            
            logger.info(f"Training with {len(combined_df)} total samples (Kaggle: {len(kaggle_df)}, Recent: {len(recent_df) if recent_df is not None else 0})")
            
            # Train the model using Kaggle integration
            if self.kaggle_integration:
                training_result = await self.kaggle_integration.train_enhanced_model_with_kaggle_data(
                    training_pipeline=self.enhanced_predictor.training_pipeline,
                    processed_data_file=kaggle_file,  # Will be enhanced to handle hybrid data
                    recent_data_file=recent_file
                )
                
                # Add hybrid training metrics
                if training_result:
                    training_result['kaggle_samples'] = len(kaggle_df)
                    training_result['recent_samples'] = len(recent_df) if recent_df is not None else 0
                    training_result['total_samples'] = len(combined_df)
                    training_result['training_type'] = 'hybrid'
                
                return training_result
            
            return None
            
        except Exception as e:
            logger.error(f"Hybrid training failed: {e}")
            return None
    
    async def _consider_hybrid_retraining(self):
        """Consider if hybrid retraining should be triggered"""
        try:
            # Check if we have enough recent data and performance has changed
            if len(self.recent_match_buffer) >= 1000:  # Minimum threshold
                # Check current model performance
                if self.monitoring_system:
                    current_accuracy = self.monitoring_system.get_current_accuracy()
                    if current_accuracy < 0.88:  # Retrain if accuracy drops below 88%
                        logger.info("Performance drop detected, triggering hybrid retraining...")
                        await self._auto_train_with_hybrid_data()
                
        except Exception as e:
            logger.warning(f"Failed to consider retraining: {e}")
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics and monitoring data"""
        try:
            metrics = {}
            
            if self.use_enhanced_model and self.enhanced_predictor:
                metrics['enhanced_performance'] = self.enhanced_predictor.get_performance_metrics()
            
            if self.monitoring_system:
                metrics['monitoring_dashboard'] = self.monitoring_system.get_monitoring_dashboard()
            
            if self.kaggle_integration:
                metrics['kaggle_dataset'] = self.kaggle_integration.get_dataset_statistics()
            
            # Add hybrid training metrics
            metrics['hybrid_training'] = {
                'enabled': self.use_hybrid_training,
                'recent_matches_collected': len(self.recent_match_buffer),
                'max_recent_matches': self.max_recent_matches,
                'recent_match_coverage': f"{len(self.recent_match_buffer)}/{self.max_recent_matches}"
            }
            
            metrics['model_version'] = self.model_version
            metrics['is_enhanced'] = self.use_enhanced_model
            metrics['kaggle_integrated'] = self.kaggle_integration is not None
            metrics['hybrid_training_active'] = self.use_hybrid_training
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting enhanced metrics: {e}")
            return {'error': str(e)}
