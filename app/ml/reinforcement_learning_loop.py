import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
from collections import deque
import json

logger = logging.getLogger(__name__)

class ReinforcementLearningLoop:
    """
    Adaptive self-learning system for ClashVision v3.0
    
    Implements post-match learning with prediction verification,
    micro-batch retraining, and continuous model improvement.
    """
    
    def __init__(
        self,
        model,
        feature_engineer,
        pci_calculator,
        max_feedback_buffer_size: int = 1000,
        micro_batch_size: int = 32,
        learning_rate_adjustment: float = 0.1,
        confidence_threshold: float = 0.8
    ):
        self.model = model
        self.feature_engineer = feature_engineer
        self.pci_calculator = pci_calculator
        
        # Feedback buffer for storing prediction outcomes
        self.feedback_buffer = deque(maxlen=max_feedback_buffer_size)
        self.micro_batch_size = micro_batch_size
        self.learning_rate_adjustment = learning_rate_adjustment
        self.confidence_threshold = confidence_threshold
        
        # Performance tracking
        self.prediction_accuracy_rolling = deque(maxlen=100)
        self.model_confidence_history = deque(maxlen=100)
        self.pci_correlation_history = deque(maxlen=100)
        
        # Reinforcement learning parameters
        self.reward_weights = {
            'correct_high_confidence': 1.0,
            'correct_low_confidence': 0.5,
            'incorrect_high_confidence': -1.0,
            'incorrect_low_confidence': -0.3
        }
        
        # Model versioning
        self.model_version = "v3.0-PCI-RL"
        self.last_retrain_time = datetime.now()
        self.retrain_interval_hours = 24  # Weekly retraining
        
    async def process_battle_outcome(
        self,
        player_tag: str,
        prediction_data: Dict[str, Any],
        actual_outcome: bool,
        battle_data: Dict[str, Any]
    ):
        """Process actual battle outcome and update model"""
        try:
            # Extract prediction details
            predicted_win_prob = prediction_data.get('win_probability', 0.5)
            prediction_confidence = prediction_data.get('confidence', 0.5)
            pci_value = prediction_data.get('pci_value', 0.5)
            
            # Determine if prediction was correct
            predicted_win = predicted_win_prob > 0.5
            was_correct = predicted_win == actual_outcome
            
            # Calculate reward
            reward = self._calculate_reward(
                was_correct, prediction_confidence, pci_value
            )
            
            # Store feedback
            feedback_entry = {
                'player_tag': player_tag,
                'timestamp': datetime.now().isoformat(),
                'predicted_prob': predicted_win_prob,
                'actual_outcome': actual_outcome,
                'confidence': prediction_confidence,
                'pci_value': pci_value,
                'was_correct': was_correct,
                'reward': reward,
                'battle_data': battle_data,
                'features': prediction_data.get('input_features', {})
            }
            
            self.feedback_buffer.append(feedback_entry)
            
            # Update performance tracking
            self.prediction_accuracy_rolling.append(1.0 if was_correct else 0.0)
            self.model_confidence_history.append(prediction_confidence)
            
            # Log outcome
            logger.info(
                f"Battle outcome processed - Player: {player_tag}, "
                f"Predicted: {predicted_win_prob:.3f}, Actual: {actual_outcome}, "
                f"Correct: {was_correct}, Reward: {reward:.3f}, PCI: {pci_value:.3f}"
            )
            
            # Trigger micro-batch learning if buffer is ready
            if len(self.feedback_buffer) >= self.micro_batch_size:
                await self._perform_micro_batch_update()
            
            # Check if full retraining is needed
            await self._check_retrain_triggers()
            
        except Exception as e:
            logger.error(f"Error processing battle outcome: {e}")
    
    def _calculate_reward(
        self,
        was_correct: bool,
        confidence: float,
        pci_value: float
    ) -> float:
        """Calculate reward for reinforcement learning"""
        try:
            # Base reward based on correctness and confidence
            if was_correct:
                if confidence > self.confidence_threshold:
                    base_reward = self.reward_weights['correct_high_confidence']
                else:
                    base_reward = self.reward_weights['correct_low_confidence']
            else:
                if confidence > self.confidence_threshold:
                    base_reward = self.reward_weights['incorrect_high_confidence']
                else:
                    base_reward = self.reward_weights['incorrect_low_confidence']
            
            # PCI-based reward modulation
            # Reward higher for consistent players, penalize more for inconsistent
            pci_modifier = 1.0 + (pci_value - 0.5) * 0.5
            
            final_reward = base_reward * pci_modifier
            
            return final_reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    async def _perform_micro_batch_update(self):
        """Perform micro-batch model update with recent feedback"""
        try:
            logger.info("Performing micro-batch model update...")
            
            # Prepare training data from feedback buffer
            recent_feedback = list(self.feedback_buffer)[-self.micro_batch_size:]
            
            X_batch = []
            y_batch = []
            pci_batch = []
            weights_batch = []
            
            for feedback in recent_feedback:
                # Prepare features
                features = feedback['features']
                feature_vector = self.feature_engineer.features_to_vector(features)
                
                # Prepare sequence input
                sequence_input = np.expand_dims(feature_vector, axis=0)
                sequence_input = np.expand_dims(sequence_input, axis=1)
                
                # Pad to expected sequence length
                target_seq_len = 50
                if sequence_input.shape[1] < target_seq_len:
                    padding = np.zeros((1, target_seq_len - sequence_input.shape[1], sequence_input.shape[2]))
                    sequence_input = np.concatenate([sequence_input, padding], axis=1)
                
                X_batch.append(sequence_input[0])
                y_batch.append(1.0 if feedback['actual_outcome'] else 0.0)
                pci_batch.append(feedback['pci_value'])
                
                # Use reward as sample weight
                weight = max(0.1, abs(feedback['reward']))  # Minimum weight of 0.1
                weights_batch.append(weight)
            
            if not X_batch:
                return
            
            # Convert to numpy arrays
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            pci_batch = np.array(pci_batch).reshape(-1, 1)
            weights_batch = np.array(weights_batch)
            
            # Perform gradient update
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = self.model([X_batch, pci_batch], training=True)
                
                # Calculate loss
                pred_probs = predictions['prediction']
                loss = tf.keras.losses.binary_crossentropy(y_batch, pred_probs)
                
                # Apply sample weights
                weighted_loss = tf.reduce_mean(loss * weights_batch)
            
            # Calculate gradients
            gradients = tape.gradient(weighted_loss, self.model.trainable_variables)
            
            # Apply gradients with adjusted learning rate
            adjusted_lr = self.model.optimizer.learning_rate * self.learning_rate_adjustment
            scaled_gradients = [g * adjusted_lr for g in gradients if g is not None]
            
            # Update model weights
            self.model.optimizer.apply_gradients(
                zip(scaled_gradients, [v for v, g in zip(self.model.trainable_variables, gradients) if g is not None])
            )
            
            logger.info(f"Micro-batch update completed with {len(X_batch)} samples")
            
        except Exception as e:
            logger.error(f"Error in micro-batch update: {e}")
    
    async def _check_retrain_triggers(self):
        """Check if full model retraining should be triggered"""
        try:
            current_time = datetime.now()
            time_since_retrain = (current_time - self.last_retrain_time).total_seconds() / 3600
            
            # Calculate current performance metrics
            if len(self.prediction_accuracy_rolling) >= 50:
                current_accuracy = np.mean(list(self.prediction_accuracy_rolling)[-50:])
                older_accuracy = np.mean(list(self.prediction_accuracy_rolling)[-100:-50]) if len(self.prediction_accuracy_rolling) >= 100 else current_accuracy
                accuracy_drop = older_accuracy - current_accuracy
            else:
                accuracy_drop = 0.0
                current_accuracy = 0.5
            
            # Calculate PCI correlation
            pci_correlation = self._calculate_pci_correlation()
            
            # Check retrain triggers
            should_retrain = False
            retrain_reason = ""
            
            # Time-based trigger (weekly)
            if time_since_retrain >= self.retrain_interval_hours:
                should_retrain = True
                retrain_reason = "scheduled_weekly_retrain"
            
            # Accuracy drop trigger (>5%)
            elif accuracy_drop > 0.05:
                should_retrain = True
                retrain_reason = f"accuracy_drop_{accuracy_drop:.3f}"
            
            # PCI correlation trigger (>0.25)
            elif abs(pci_correlation) > 0.25:
                should_retrain = True
                retrain_reason = f"pci_correlation_{pci_correlation:.3f}"
            
            # Population shift trigger (simplified)
            elif self._detect_population_shift():
                should_retrain = True
                retrain_reason = "population_shift"
            
            if should_retrain:
                logger.info(f"Triggering full model retrain: {retrain_reason}")
                await self._trigger_full_retrain(retrain_reason)
            
        except Exception as e:
            logger.error(f"Error checking retrain triggers: {e}")
    
    def _calculate_pci_correlation(self) -> float:
        """Calculate correlation between PCI and prediction accuracy"""
        try:
            if len(self.feedback_buffer) < 20:
                return 0.0
            
            recent_feedback = list(self.feedback_buffer)[-50:]
            pci_values = [f['pci_value'] for f in recent_feedback]
            accuracies = [1.0 if f['was_correct'] else 0.0 for f in recent_feedback]
            
            correlation = np.corrcoef(pci_values, accuracies)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating PCI correlation: {e}")
            return 0.0
    
    def _detect_population_shift(self) -> bool:
        """Detect if there's been a significant shift in player population"""
        try:
            if len(self.feedback_buffer) < 100:
                return False
            
            # Compare PCI distribution in recent vs older samples
            recent_feedback = list(self.feedback_buffer)[-50:]
            older_feedback = list(self.feedback_buffer)[-100:-50]
            
            recent_pci = [f['pci_value'] for f in recent_feedback]
            older_pci = [f['pci_value'] for f in older_feedback]
            
            # Simple distribution shift detection using mean difference
            recent_mean = np.mean(recent_pci)
            older_mean = np.mean(older_pci)
            
            shift_magnitude = abs(recent_mean - older_mean)
            
            return shift_magnitude > 0.1  # 10% shift threshold
            
        except Exception as e:
            logger.error(f"Error detecting population shift: {e}")
            return False
    
    async def _trigger_full_retrain(self, reason: str):
        """Trigger full model retraining"""
        try:
            logger.info(f"Starting full model retrain due to: {reason}")
            
            # Prepare full training dataset from feedback buffer
            training_data = []
            for feedback in self.feedback_buffer:
                training_data.append({
                    'player_data': {'features': feedback['features']},
                    'opponent_data': {},
                    'battle_context': feedback['battle_data'],
                    'result': 'win' if feedback['actual_outcome'] else 'loss',
                    'pci_value': feedback['pci_value']
                })
            
            if len(training_data) < 100:
                logger.warning("Insufficient data for full retrain, skipping...")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(training_data)
            
            # Prepare training data
            X, y, pci_values = self._prepare_full_training_data(df)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            pci_train, pci_val = pci_values[:split_idx], pci_values[split_idx:]
            
            # Train model
            history = self.model.fit(
                [X_train, pci_train], 
                {'prediction': y_train, 'confidence': y_train},  # Use same target for both heads
                validation_data=([X_val, pci_val], {'prediction': y_val, 'confidence': y_val}),
                epochs=20,
                batch_size=32,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            
            # Update last retrain time
            self.last_retrain_time = datetime.now()
            
            # Log retrain completion
            final_val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0.0
            logger.info(f"Full retrain completed. Final validation accuracy: {final_val_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Error in full model retrain: {e}")
    
    def _prepare_full_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare full training dataset"""
        try:
            X = []
            y = []
            pci_values = []
            
            for _, row in df.iterrows():
                # Extract features
                features = row['player_data']['features']
                feature_vector = self.feature_engineer.features_to_vector(features)
                
                # Prepare sequence input
                sequence_input = np.expand_dims(feature_vector, axis=0)
                sequence_input = np.expand_dims(sequence_input, axis=1)
                
                # Pad to expected sequence length
                target_seq_len = 50
                if sequence_input.shape[1] < target_seq_len:
                    padding = np.zeros((1, target_seq_len - sequence_input.shape[1], sequence_input.shape[2]))
                    sequence_input = np.concatenate([sequence_input, padding], axis=1)
                
                X.append(sequence_input[0])
                y.append(1.0 if row['result'] == 'win' else 0.0)
                pci_values.append(row['pci_value'])
            
            return np.array(X), np.array(y), np.array(pci_values).reshape(-1, 1)
            
        except Exception as e:
            logger.error(f"Error preparing full training data: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            if not self.prediction_accuracy_rolling:
                return {}
            
            current_accuracy = np.mean(list(self.prediction_accuracy_rolling))
            recent_accuracy = np.mean(list(self.prediction_accuracy_rolling)[-20:]) if len(self.prediction_accuracy_rolling) >= 20 else current_accuracy
            
            metrics = {
                'prediction_accuracy_rolling_100': current_accuracy,
                'recent_accuracy_20': recent_accuracy,
                'total_predictions': len(self.prediction_accuracy_rolling),
                'feedback_buffer_size': len(self.feedback_buffer),
                'pci_correlation': self._calculate_pci_correlation(),
                'model_version': self.model_version,
                'last_retrain_time': self.last_retrain_time.isoformat(),
                'confidence_distribution': {
                    'mean': np.mean(list(self.model_confidence_history)) if self.model_confidence_history else 0.0,
                    'std': np.std(list(self.model_confidence_history)) if self.model_confidence_history else 0.0
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def save_feedback_data(self, filepath: str):
        """Save feedback buffer to file for analysis"""
        try:
            feedback_data = list(self.feedback_buffer)
            with open(filepath, 'w') as f:
                json.dump(feedback_data, f, indent=2, default=str)
            
            logger.info(f"Feedback data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
    
    def load_feedback_data(self, filepath: str):
        """Load feedback data from file"""
        try:
            with open(filepath, 'r') as f:
                feedback_data = json.load(f)
            
            # Convert back to deque
            self.feedback_buffer = deque(feedback_data, maxlen=self.feedback_buffer.maxlen)
            
            logger.info(f"Feedback data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
    
    async def cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        try:
            # Clear old PCI cache entries
            self.pci_calculator.clear_old_cache_entries()
            
            # The deque automatically handles size limits, so no additional cleanup needed
            logger.debug("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
