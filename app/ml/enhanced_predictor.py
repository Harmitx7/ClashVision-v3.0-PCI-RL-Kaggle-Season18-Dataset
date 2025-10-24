import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json
from datetime import datetime
import os
import time

from app.core.config import settings
from app.core.structured_logger import structured_logger, log_error, log_info, log_warning, LayerType, ErrorType
from app.ml.enhanced_model_architecture import EnhancedTransformerLSTMModel, create_enhanced_model
from app.ml.player_consistency_index import PlayerConsistencyIndex
from app.ml.feature_engineering import FeatureEngineer
from app.ml.reinforcement_learning_loop import ReinforcementLearningLoop
from app.ml.adaptive_training_pipeline import AdaptiveTrainingPipeline
from app.ml.kaggle_data_integration import KaggleDataIntegration

logger = logging.getLogger(__name__)

class EnhancedWinPredictor:
    """
    Enhanced Win Predictor for ClashVision v3.0-PCI-RL
    
    Features:
    - PCI-conditioned predictions with confidence modulation
    - Adaptive self-learning with RL loop
    - Specialized model routing for extreme PCI values
    - Real-time prediction accuracy monitoring
    - Animated confidence visualization
    """
    
    def __init__(self):
        self.model: Optional[EnhancedTransformerLSTMModel] = None
        self.feature_engineer = FeatureEngineer()
        self.pci_calculator = PlayerConsistencyIndex()
        self.rl_loop: Optional[ReinforcementLearningLoop] = None
        self.training_pipeline: Optional[AdaptiveTrainingPipeline] = None
        self.kaggle_integration: Optional[KaggleDataIntegration] = None
        
        self.model_version = "v3.0-PCI-RL"
        self.is_ready = False
        self.confidence_threshold = settings.PREDICTION_CONFIDENCE_THRESHOLD
        
        # Model paths
        self.model_path = os.path.join(settings.MODEL_PATH, "enhanced_win_predictor.h5")
        self.scaler_path = os.path.join(settings.MODEL_PATH, "enhanced_feature_scaler.pkl")
        
        # Specialized models for extreme PCI values
        self.tilt_model = None
        self.elite_model = None
        
        # Performance monitoring
        self.prediction_count = 0
        self.accuracy_rolling = []
        self.confidence_history = []
        
    async def initialize(self):
        """Initialize the enhanced ML system"""
        try:
            logger.info("Initializing Enhanced Win Predictor v3.0-PCI-RL...")
            
            # Create model directory
            os.makedirs(settings.MODEL_PATH, exist_ok=True)
            
            # Initialize training pipeline
            self.training_pipeline = AdaptiveTrainingPipeline()
            await self.training_pipeline.initialize()
            
            # Load or create main model
            if os.path.exists(self.model_path):
                await self._load_model()
            else:
                await self._create_new_model()
            
            # Initialize Kaggle dataset integration
            try:
                logger.info("Initializing Kaggle dataset integration...")
                self.kaggle_integration = KaggleDataIntegration()
                await self.kaggle_integration.download_and_prepare_dataset()
                logger.info("Kaggle dataset integration completed")
            except Exception as e:
                logger.info(f"Kaggle dataset integration failed: {e} - using fallback mode")
                self.kaggle_integration = None
            
            # Initialize RL loop with Kaggle data
            self.rl_loop = ReinforcementLearningLoop(
                model=self.model,
                feature_engineer=self.feature_engineer,
                pci_calculator=self.pci_calculator
            )
            
            # Load specialized models
            await self._load_specialized_models()
            
            self.is_ready = True
            logger.info("Enhanced Win Predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Enhanced Win Predictor: {e}")
            self.is_ready = False
    
    async def _load_model(self):
        """Load existing enhanced model with robust error handling"""
        try:
            logger.info("Loading enhanced model with stability checks...")
            self.model = create_enhanced_model()
            
            # Build model with dummy data and validate
            dummy_features = np.random.random((1, 50, 64))
            
            # Check for NaN/Inf values in dummy data
            if np.any(np.isnan(dummy_features)) or np.any(np.isinf(dummy_features)):
                logger.warning("NaN/Inf detected in dummy features, regenerating...")
                dummy_features = np.random.uniform(0.1, 0.9, (1, 50, 64))
            dummy_pci = np.random.random((1, 1))
            _ = self.model([dummy_features, dummy_pci])
            
            # Load weights with validation
            if os.path.exists(self.model_path):
                try:
                    self.model.load_weights(self.model_path)
                    logger.info("Model weights loaded successfully")
                    
                    # Validate model with test prediction
                    test_output = self.model([dummy_features, dummy_pci])
                    if np.any(np.isnan(test_output)) or np.any(np.isinf(test_output)):
                        raise ValueError("Model produces NaN/Inf outputs")
                    
                except Exception as e:
                    logger.warning(f"Failed to load model weights: {e}, reinitializing...")
                    await self._create_new_model()
                    return
            else:
                logger.info("No existing model weights found, creating new model")
                await self._create_new_model()
                return
            
            # Load feature scaler with validation
            if os.path.exists(self.scaler_path):
                try:
                    self.feature_engineer.load_scaler(self.scaler_path)
                    logger.info("Feature scaler loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load feature scaler: {e}, using default scaling")
            
            logger.info("Enhanced model loaded and validated successfully")
            
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")
            await self._create_new_model()
    
    async def _create_new_model(self):
        """Create new enhanced model"""
        try:
            logger.info("Creating new enhanced model...")
            
            self.model = create_enhanced_model()
            
            # Build model with dummy data
            dummy_features = np.random.random((1, 50, 64))
            dummy_pci = np.random.random((1, 1))
            _ = self.model([dummy_features, dummy_pci])
            
            # Initialize feature scaler
            self.feature_engineer.initialize_scaler()
            
            logger.info("New enhanced model created successfully")
            
        except Exception as e:
            logger.error(f"Error creating new enhanced model: {e}")
            raise
    
    async def _load_specialized_models(self):
        """Load specialized models for extreme PCI values"""
        try:
            # For now, use the same model architecture
            # In production, these would be separately trained models
            self.tilt_model = self.model
            self.elite_model = self.model
            
            logger.info("Specialized models loaded")
            
        except Exception as e:
            logger.error(f"Error loading specialized models: {e}")
    
    async def predict(
        self,
        player_data: Dict[str, Any],
        opponent_data: Optional[Dict[str, Any]] = None,
        battle_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make enhanced prediction with PCI integration"""
        start_time = time.time()
        player_tag = player_data.get('tag', 'unknown')
        
        try:
            if not self.is_ready:
                log_error("Enhanced model not ready for prediction", LayerType.MODEL, ErrorType.MODEL_ERROR,
                         additional_data={"player_tag": player_tag})
                raise ValueError("Enhanced model not ready")
            
            # Calculate PCI using enhanced dataset analysis
            player_tag = player_data.get('tag', 'unknown')
            battle_history = player_data.get('battlelog', [])
            player_stats = player_data.get('stats', {})
            
            # Enhanced PCI calculation with Kaggle dataset benchmarking
            pci_value = await self._calculate_enhanced_pci(
                player_tag=player_tag,
                battle_history=battle_history,
                player_stats=player_stats,
                current_deck=player_data.get('currentDeck', []),
                trophies=player_data.get('trophies', 5000)
            )
            
            # Get PCI interpretation
            pci_interpretation = self.pci_calculator.get_pci_interpretation(pci_value)
            
            # Extract features with validation
            features = self.feature_engineer.extract_features(
                player_data=player_data,
                opponent_data=opponent_data,
                battle_context=battle_context
            )
            
            # Validate and sanitize features
            features = self._validate_and_sanitize_features(features)
            if features is None:
                raise ValueError("Feature extraction failed - invalid input data")
            
            # Prepare model input
            model_input = self._prepare_model_input(features)
            pci_input = np.array([[pci_value]])
            
            # Select appropriate model based on PCI
            should_use_specialized, model_type = self.pci_calculator.should_use_specialized_model(pci_value)
            
            if should_use_specialized:
                if model_type == "tilt_model" and self.tilt_model:
                    selected_model = self.tilt_model
                elif model_type == "elite_model" and self.elite_model:
                    selected_model = self.elite_model
                else:
                    selected_model = self.model
            else:
                selected_model = self.model
            
            # Make prediction
            prediction_output = selected_model([model_input, pci_input])
            
            win_probability = float(prediction_output['prediction'][0][0])
            base_confidence = float(prediction_output['confidence'][0][0])
            
            # Apply PCI-based confidence modulation
            modulated_confidence = self.pci_calculator.calculate_confidence_modulation(
                base_confidence, pci_value
            )
            
            # Generate enhanced analysis
            analysis = self._generate_enhanced_analysis(features, pci_value, win_probability)
            
            # Generate strategic analysis
            logger.info(f"Generating strategic analysis with Kaggle dataset integration: {bool(self.kaggle_integration)}")
            strategic_analysis = await self._generate_strategic_analysis(player_data, pci_value, win_probability)
            
            # Create prediction result
            result = {
                "win_probability": win_probability,
                "confidence": modulated_confidence,
                "base_confidence": base_confidence,
                "pci_value": pci_value,
                "pci_interpretation": pci_interpretation,
                "model_version": self.model_version,
                "model_used": model_type if should_use_specialized else "standard_model",
                "input_features": features,
                "enhanced_analysis": analysis,
                "visual_output": self._generate_visual_output(win_probability, modulated_confidence, pci_value),
                "timestamp": datetime.utcnow().isoformat(),
                # Add strategic analysis fields that frontend expects
                "battle_tactics": strategic_analysis.get("battle_tactics", []),
                "detailed_card_suggestions": strategic_analysis.get("detailed_card_suggestions", {}),
                "counter_strategies": strategic_analysis.get("counter_strategies", []),
                "meta_insights": strategic_analysis.get("meta_insights", {})
            }
            
            # Update monitoring
            self.prediction_count += 1
            self.confidence_history.append(modulated_confidence)
            
            # Log successful prediction with structured logging
            processing_time = time.time() - start_time
            structured_logger.log_model_prediction(
                player_tag=player_tag,
                prediction_result=result,
                confidence_before=base_confidence,
                confidence_after=modulated_confidence,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            log_error(f"Error making enhanced prediction: {e}", LayerType.MODEL, ErrorType.MODEL_ERROR,
                     additional_data={
                         "player_tag": player_tag,
                         "processing_time_ms": processing_time * 1000,
                         "error_details": str(e)
                     })
            raise
    
    async def predict_live(self, player_tag: str, battle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make live prediction with real-time PCI updates"""
        try:
            if not self.is_ready:
                raise ValueError("Enhanced model not ready")
            
            # Extract live features
            features = self.feature_engineer.extract_live_features(
                player_tag=player_tag,
                battle_data=battle_data
            )
            
            # Calculate real-time PCI (simplified for live prediction)
            pci_value = 0.5  # Default for live prediction, would be updated with recent battle data
            
            # Prepare model input
            model_input = self._prepare_model_input(features)
            pci_input = np.array([[pci_value]])
            
            # Make prediction
            prediction_output = self.model([model_input, pci_input])
            
            win_probability = float(prediction_output['prediction'][0][0])
            confidence = float(prediction_output['confidence'][0][0])
            
            # Generate live recommendations
            live_analysis = self._generate_live_analysis(battle_data, win_probability, pci_value)
            
            return {
                "win_probability": win_probability,
                "confidence": confidence,
                "pci_value": pci_value,
                "model_version": self.model_version,
                "live_analysis": live_analysis,
                "visual_output": self._generate_visual_output(win_probability, confidence, pci_value),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making live prediction: {e}")
            raise
    
    def _prepare_model_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for enhanced model input"""
        try:
            feature_vector = self.feature_engineer.features_to_vector(features)
            
            # Create sequence input
            sequence_input = np.expand_dims(feature_vector, axis=0)
            sequence_input = np.expand_dims(sequence_input, axis=1)
            
            # Pad to expected sequence length
            target_seq_len = 50
            current_seq_len = sequence_input.shape[1]
            
            if current_seq_len < target_seq_len:
                padding = np.zeros((1, target_seq_len - current_seq_len, sequence_input.shape[2]))
                sequence_input = np.concatenate([sequence_input, padding], axis=1)
            elif current_seq_len > target_seq_len:
                sequence_input = sequence_input[:, :target_seq_len, :]
            
            return sequence_input
            
        except Exception as e:
            logger.error(f"Error preparing model input: {e}")
            raise
    
    def _generate_enhanced_analysis(
        self, 
        features: Dict[str, Any], 
        pci_value: float, 
        win_probability: float
    ) -> Dict[str, Any]:
        """Generate enhanced analysis with PCI insights"""
        try:
            analysis = {
                "key_factors": {
                    "player_consistency": {
                        "pci_score": pci_value,
                        "stability_level": "High" if pci_value > 0.7 else "Medium" if pci_value > 0.4 else "Low",
                        "impact": "Positive" if pci_value > 0.5 else "Negative"
                    },
                    "deck_synergy": features.get("deck_synergy_score", 0.0),
                    "opponent_matchup": features.get("counter_score", 0.0),
                    "recent_form": features.get("recent_win_rate", 0.0)
                },
                "risk_assessment": {
                    "tilt_probability": max(0.0, 1.0 - pci_value * 2),
                    "consistency_risk": "Low" if pci_value > 0.6 else "High",
                    "prediction_reliability": "High" if pci_value > 0.5 else "Moderate"
                },
                "strategic_insights": self._generate_strategic_insights(features, pci_value, win_probability)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating enhanced analysis: {e}")
            return {}
    
    def _generate_strategic_insights(
        self, 
        features: Dict[str, Any], 
        pci_value: float, 
        win_probability: float
    ) -> List[str]:
        """Generate strategic insights based on PCI and features"""
        insights = []
        
        if pci_value < 0.3:
            insights.append("Player showing signs of tilt - consider defensive strategy")
            insights.append("High variance expected - prepare for unpredictable plays")
        elif pci_value > 0.8:
            insights.append("Highly consistent player - expect optimal decision making")
            insights.append("Low variance expected - standard counter-strategies effective")
        
        if win_probability > 0.7:
            insights.append("Strong advantage - maintain pressure and avoid risky plays")
        elif win_probability < 0.3:
            insights.append("Challenging matchup - focus on elixir efficiency and patience")
        
        return insights
    
    async def _generate_strategic_analysis(
        self, 
        player_data: Dict[str, Any], 
        pci_value: float, 
        win_probability: float
    ) -> Dict[str, Any]:
        """Generate comprehensive strategic analysis for frontend"""
        try:
            current_deck = player_data.get('currentDeck', [])
            trophies = player_data.get('trophies', 5000)
            
            # Generate battle tactics based on PCI and win probability
            battle_tactics = []
            if pci_value > 0.7:
                battle_tactics.extend([
                    "🎯 Play aggressively - your consistency allows for complex strategies",
                    "⚡ Execute advanced combos with confidence",
                    "🛡️ Trust your instincts on risky plays"
                ])
            elif pci_value > 0.4:
                battle_tactics.extend([
                    "🎯 Play balanced strategy - mix offense and defense",
                    "⚡ Focus on positive elixir trades",
                    "🛡️ Avoid overly risky plays"
                ])
            else:
                battle_tactics.extend([
                    "🎯 Play defensively - build confidence with safe plays",
                    "⚡ Stick to familiar card combinations",
                    "🛡️ Avoid complex strategies until consistency improves"
                ])
            
            if win_probability > 0.6:
                battle_tactics.append("🔥 You have the advantage - maintain pressure")
            else:
                battle_tactics.append("🛡️ Focus on defense and counter-attacks")
            
            # Generate card suggestions based on current deck and meta
            card_suggestions = {
                "cards_to_add": [],
                "cards_to_remove": [],
                "deck_improvements": []
            }
            
            # Always add some basic suggestions
            card_suggestions["cards_to_add"].append({
                "card": "Musketeer",
                "reason": "Versatile ranged unit with good damage output",
                "priority": "medium",
                "synergy_score": 0.75
            })
            
            card_suggestions["cards_to_remove"].append({
                "card": "Wizard",
                "reason": "High elixir cost with limited defensive value",
                "priority": "low",
                "alternative_suggestions": ["Musketeer", "Archers", "Baby Dragon"]
            })
            
            card_suggestions["deck_improvements"].append("Consider balancing offensive and defensive cards")
            
            if current_deck:
                # Analyze deck composition
                avg_elixir = sum(card.get('elixirCost', 0) for card in current_deck) / len(current_deck)
                
                if avg_elixir > 4.2:
                    card_suggestions["cards_to_remove"].append({
                        "card": "High-cost card",
                        "reason": f"Average elixir cost too high: {avg_elixir:.1f}",
                        "priority": "medium",
                        "alternative_suggestions": ["Lower cost alternatives"]
                    })
                
                # Check for spell coverage
                spells = [card for card in current_deck if card.get('name', '') in ['Fireball', 'Zap', 'Lightning', 'Arrows', 'Poison']]
                if len(spells) < 2:
                    card_suggestions["cards_to_add"].append({
                        "card": "Fireball",
                        "reason": "Excellent spell for crowd control and finishing towers",
                        "priority": "high",
                        "synergy_score": 0.85
                    })
                
                # Check for win conditions
                win_conditions = [card for card in current_deck if card.get('name', '') in ['Hog Rider', 'Giant', 'Golem', 'Balloon', 'Miner']]
                if len(win_conditions) == 0:
                    card_suggestions["cards_to_add"].append({
                        "card": "Hog Rider",
                        "reason": "Fast win condition with high versatility",
                        "priority": "high",
                        "synergy_score": 0.78
                    })
                
                # Check for defensive buildings
                buildings = [card for card in current_deck if card.get('name', '') in ['Tesla', 'Cannon', 'Inferno Tower', 'X-Bow']]
                if len(buildings) == 0 and trophies > 4000:
                    card_suggestions["cards_to_add"].append({
                        "card": "Tesla",
                        "reason": "Versatile defensive building effective against most pushes",
                        "priority": "medium",
                        "synergy_score": 0.72
                    })
                
                # Suggest removing high-cost cards if deck is too expensive
                expensive_cards = [card for card in current_deck if card.get('elixirCost', 0) >= 6]
                if expensive_cards and avg_elixir > 4.5:
                    card_suggestions["cards_to_remove"].append({
                        "card": expensive_cards[0].get('name', 'Expensive card'),
                        "reason": f"High elixir cost ({expensive_cards[0].get('elixirCost', 6)}) makes deck too slow",
                        "priority": "medium",
                        "alternative_suggestions": ["Musketeer", "Mini P.E.K.K.A", "Valkyrie"]
                    })
                
                # General improvements
                if pci_value < 0.5:
                    card_suggestions["deck_improvements"].append("Consider using more consistent, reliable cards")
                else:
                    card_suggestions["deck_improvements"].append("Deck composition allows for advanced strategies")
            
            # Generate counter strategies
            counter_strategies = [
                "Adapt strategy based on opponent's opening moves",
                "Counter heavy pushes with defensive positioning",
                "Use spell combinations for maximum efficiency"
            ]
            
            if trophies > 5500:
                counter_strategies.append("Expect skilled opponents - prepare for meta strategies")
            
            # Generate meta insights from Kaggle dataset
            meta_insights = await self._get_kaggle_meta_insights(trophies)
            if not meta_insights:
                # Fallback meta insights
                meta_insights = {
                    "trending_cards": [
                        {
                            "card": "Musketeer",
                            "usage_rate": 0.58,
                            "win_rate": 0.68,
                            "trend_strength": "high"
                        },
                        {
                            "card": "Fireball",
                            "usage_rate": 0.52,
                            "win_rate": 0.64,
                            "trend_strength": "medium"
                        }
                    ],
                    "recommended_adaptations": [
                        "Current meta favors balanced decks",
                        "Air defense becoming increasingly important",
                        "Cycle decks showing strong performance"
                    ]
                }
            
            return {
                "battle_tactics": battle_tactics,
                "detailed_card_suggestions": card_suggestions,
                "counter_strategies": counter_strategies,
                "meta_insights": meta_insights
            }
            
        except Exception as e:
            logger.error(f"Error generating strategic analysis: {e}")
            return {
                "battle_tactics": ["Strategic analysis unavailable"],
                "detailed_card_suggestions": {"cards_to_add": [], "cards_to_remove": [], "deck_improvements": []},
                "counter_strategies": ["Basic counter strategies recommended"],
                "meta_insights": {"trending_cards": [], "recommended_adaptations": []}
            }
    
    async def _get_kaggle_meta_insights(self, trophies: int) -> Optional[Dict[str, Any]]:
        """Get meta insights from Kaggle dataset based on trophy range"""
        try:
            if not self.kaggle_integration:
                return None
            
            # Get card statistics from Kaggle dataset for trophy range
            card_stats = await self.kaggle_integration.get_card_statistics_by_trophy_range(
                min_trophies=max(4000, trophies - 500),
                max_trophies=trophies + 500
            )
            
            if not card_stats:
                return None
            
            # Convert to frontend format
            trending_cards = []
            for card_name, stats in card_stats.items():
                if stats.get('usage_count', 0) > 100:  # Minimum usage threshold
                    trending_cards.append({
                        "card": card_name,
                        "usage_rate": stats.get('usage_rate', 0.0),
                        "win_rate": stats.get('win_rate', 0.0),
                        "trend_strength": "high" if stats.get('win_rate', 0.0) > 0.6 else "medium"
                    })
            
            # Sort by win rate and take top 5
            trending_cards.sort(key=lambda x: x['win_rate'], reverse=True)
            trending_cards = trending_cards[:5]
            
            # Generate adaptations based on data
            adaptations = []
            if trending_cards:
                top_card = trending_cards[0]
                adaptations.append(f"{top_card['card']} showing {top_card['win_rate']:.1%} win rate in your trophy range")
                
                # Analyze meta trends
                spell_cards = [c for c in trending_cards if c['card'] in ['Fireball', 'Zap', 'Lightning', 'Arrows']]
                if len(spell_cards) >= 2:
                    adaptations.append("Spell-heavy meta detected - consider spell coverage")
                
                building_cards = [c for c in trending_cards if c['card'] in ['Tesla', 'Cannon', 'Inferno Tower']]
                if building_cards:
                    adaptations.append("Defensive buildings trending - prepare for building counters")
            
            return {
                "trending_cards": trending_cards,
                "recommended_adaptations": adaptations,
                "data_source": f"Kaggle dataset (Trophy range: {trophies-500}-{trophies+500})",
                "sample_size": sum(stats.get('usage_count', 0) for stats in card_stats.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting Kaggle meta insights: {e}")
            return None
    
    async def _calculate_enhanced_pci(
        self, 
        player_tag: str, 
        battle_history: List[Dict], 
        player_stats: Dict, 
        current_deck: List[Dict], 
        trophies: int
    ) -> float:
        """Calculate enhanced PCI using Kaggle dataset benchmarking"""
        try:
            # Base PCI calculation
            base_pci = self.pci_calculator.calculate_pci(
                player_tag=player_tag,
                battle_history=battle_history,
                player_stats=player_stats
            )
            
            if not self.kaggle_integration:
                return base_pci
            
            # Get performance benchmarks from Kaggle dataset
            deck_performance = await self.kaggle_integration.get_deck_performance_stats(
                deck_cards=[card.get('name', '') for card in current_deck],
                trophy_range=(max(4000, trophies - 500), trophies + 500)
            )
            
            if deck_performance:
                # Adjust PCI based on deck performance in similar trophy ranges
                expected_win_rate = deck_performance.get('average_win_rate', 0.5)
                player_win_rate = player_stats.get('wins', 0) / max(1, player_stats.get('wins', 0) + player_stats.get('losses', 0))
                
                # Performance adjustment factor
                performance_factor = player_win_rate / max(0.1, expected_win_rate)
                
                # Adjust PCI (cap between 0.1 and 0.95)
                enhanced_pci = base_pci * (0.8 + 0.4 * performance_factor)
                enhanced_pci = max(0.1, min(0.95, enhanced_pci))
                
                logger.info(f"Enhanced PCI: {enhanced_pci:.3f} (base: {base_pci:.3f}, performance factor: {performance_factor:.3f})")
                return enhanced_pci
            
            return base_pci
            
        except Exception as e:
            logger.error(f"Error calculating enhanced PCI: {e}")
            return self.pci_calculator.calculate_pci(player_tag, battle_history, player_stats)
    
    def _validate_and_sanitize_features(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Validate and sanitize feature array for model input"""
        try:
            if features is None:
                logger.warning("Features is None, cannot validate")
                return None
            
            # Convert to numpy array if not already
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Check for NaN values
            if np.any(np.isnan(features)):
                logger.warning("NaN values detected in features, replacing with median")
                # Replace NaN with median of non-NaN values
                median_val = np.nanmedian(features)
                if np.isnan(median_val):
                    median_val = 0.5  # Fallback value
                features = np.nan_to_num(features, nan=median_val)
            
            # Check for infinite values
            if np.any(np.isinf(features)):
                logger.warning("Infinite values detected in features, clamping")
                features = np.clip(features, -1e6, 1e6)
            
            # Clamp outlier values to reasonable range
            features = np.clip(features, -10, 10)
            
            # Ensure proper shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            logger.debug(f"Features validated: shape={features.shape}, range=[{features.min():.3f}, {features.max():.3f}]")
            return features
            
        except Exception as e:
            logger.error(f"Error validating features: {e}")
            return None
    
    def _generate_live_analysis(
        self, 
        battle_data: Dict[str, Any], 
        win_probability: float, 
        pci_value: float
    ) -> Dict[str, Any]:
        """Generate live battle analysis"""
        try:
            battle_phase = self._determine_battle_phase(battle_data.get("time_remaining", 180))
            
            analysis = {
                "battle_phase": battle_phase,
                "momentum": "Positive" if win_probability > 0.6 else "Negative" if win_probability < 0.4 else "Neutral",
                "consistency_factor": pci_value,
                "recommended_strategy": self._get_phase_strategy(battle_phase, win_probability, pci_value),
                "risk_level": "High" if pci_value < 0.4 else "Medium" if pci_value < 0.7 else "Low"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating live analysis: {e}")
            return {}
    
    def _determine_battle_phase(self, time_remaining: int) -> str:
        """Determine current battle phase"""
        if time_remaining > 120:
            return "early_game"
        elif time_remaining > 60:
            return "mid_game"
        elif time_remaining > 0:
            return "late_game"
        else:
            return "overtime"
    
    def _get_phase_strategy(self, phase: str, win_prob: float, pci: float) -> str:
        """Get recommended strategy for battle phase"""
        if phase == "early_game":
            return "Learn opponent deck and make positive trades"
        elif phase == "mid_game":
            if win_prob > 0.6:
                return "Build strong push to capitalize on advantage"
            else:
                return "Defend efficiently and look for counter-attacks"
        elif phase == "late_game":
            if pci < 0.4:  # Inconsistent player
                return "Play conservatively - opponent may make mistakes"
            else:
                return "Execute planned strategy - consistent opponent"
        else:  # overtime
            return "High-risk plays acceptable - focus on tower damage"
    
    def _generate_visual_output(
        self, 
        win_probability: float, 
        confidence: float, 
        pci_value: float
    ) -> Dict[str, Any]:
        """Generate animated accuracy gauge configuration"""
        try:
            # Determine gauge color based on win probability and confidence
            if win_probability > 0.7 and confidence > 0.8:
                color = "#00c853"  # High confidence win
            elif win_probability > 0.5 and confidence > 0.6:
                color = "#ffd700"  # Medium confidence
            else:
                color = "#ff7b00"  # Low confidence
            
            return {
                "type": "animated_accuracy_gauge",
                "win_probability": win_probability,
                "confidence": confidence,
                "pci_value": pci_value,
                "color": color,
                "animation_speed": 1.5,
                "tooltip": f"Confidence dynamically adjusted with PCI ({pci_value:.2f}) and real match outcomes",
                "gauge_segments": [
                    {"threshold": 0.3, "color": "#ff7b00", "label": "Low"},
                    {"threshold": 0.7, "color": "#ffd700", "label": "Medium"},
                    {"threshold": 1.0, "color": "#00c853", "label": "High"}
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating visual output: {e}")
            return {"type": "static", "value": win_probability}
    
    async def process_battle_outcome(
        self,
        player_tag: str,
        prediction_data: Dict[str, Any],
        actual_outcome: bool,
        battle_data: Dict[str, Any]
    ):
        """Process battle outcome for RL learning"""
        try:
            if self.rl_loop:
                await self.rl_loop.process_battle_outcome(
                    player_tag=player_tag,
                    prediction_data=prediction_data,
                    actual_outcome=actual_outcome,
                    battle_data=battle_data
                )
            
            # Update accuracy tracking
            predicted_win = prediction_data.get('win_probability', 0.5) > 0.5
            was_correct = predicted_win == actual_outcome
            self.accuracy_rolling.append(1.0 if was_correct else 0.0)
            
            # Keep only last 100 predictions
            if len(self.accuracy_rolling) > 100:
                self.accuracy_rolling = self.accuracy_rolling[-100:]
            
        except Exception as e:
            logger.error(f"Error processing battle outcome: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            if self.rl_loop:
                rl_metrics = self.rl_loop.get_performance_metrics()
            else:
                rl_metrics = {}
            
            current_accuracy = np.mean(self.accuracy_rolling) if self.accuracy_rolling else 0.0
            
            metrics = {
                "model_version": self.model_version,
                "prediction_count": self.prediction_count,
                "current_accuracy": current_accuracy,
                "confidence_avg": np.mean(self.confidence_history) if self.confidence_history else 0.0,
                "is_ready": self.is_ready,
                **rl_metrics
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    async def save_model(self):
        """Save enhanced model and components"""
        try:
            if self.model:
                self.model.save_weights(self.model_path)
            
            if self.feature_engineer:
                self.feature_engineer.save_scaler(self.scaler_path)
            
            logger.info("Enhanced model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving enhanced model: {e}")
            raise
