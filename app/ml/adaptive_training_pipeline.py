import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from app.ml.enhanced_model_architecture import create_enhanced_model, create_enhanced_callbacks
from app.ml.player_consistency_index import PlayerConsistencyIndex
from app.ml.feature_engineering import FeatureEngineer
from app.ml.reinforcement_learning_loop import ReinforcementLearningLoop

logger = logging.getLogger(__name__)

class AdaptiveTrainingPipeline:
    """
    Adaptive training pipeline for ClashVision v3.0-PCI-RL
    
    Features:
    - Weekly + event-triggered retraining
    - PCI drift detection and adaptation
    - Automated hyperparameter tuning
    - Model versioning and rollback
    - Performance monitoring and alerting
    """
    
    def __init__(
        self,
        model_path: str = "app/ml/models/",
        data_path: str = "data/",
        target_accuracy: float = 0.91,
        pci_drift_threshold: float = 0.10,
        min_training_samples: int = 1000
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.target_accuracy = target_accuracy
        self.pci_drift_threshold = pci_drift_threshold
        self.min_training_samples = min_training_samples
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.pci_calculator = PlayerConsistencyIndex()
        self.model = None
        self.rl_loop = None
        
        # Training configuration
        self.training_config = {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'validation_split': 0.15,
            'test_split': 0.10,
            'early_stopping_patience': 10
        }
        
        # Performance tracking
        self.training_history = []
        self.model_versions = []
        self.performance_metrics = {}
        
        # Scheduling
        self.last_training_time = None
        self.weekly_training_schedule = True
        self.auto_retrain_enabled = True
        
    async def initialize(self):
        """Initialize the training pipeline"""
        try:
            logger.info("Initializing Adaptive Training Pipeline...")
            
            # Create directories
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.data_path, exist_ok=True)
            os.makedirs(f"{self.model_path}/checkpoints", exist_ok=True)
            os.makedirs(f"{self.model_path}/versions", exist_ok=True)
            
            # Initialize or load existing model
            await self._initialize_model()
            
            # Initialize RL loop
            self.rl_loop = ReinforcementLearningLoop(
                model=self.model,
                feature_engineer=self.feature_engineer,
                pci_calculator=self.pci_calculator
            )
            
            logger.info("Adaptive Training Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing training pipeline: {e}")
            raise
    
    async def _initialize_model(self):
        """Initialize or load the model"""
        try:
            latest_model_path = self._get_latest_model_path()
            
            if latest_model_path and os.path.exists(latest_model_path):
                logger.info(f"Loading existing model from {latest_model_path}")
                self.model = create_enhanced_model()
                
                # Build model with dummy data
                dummy_features = np.random.random((1, 50, 64))
                dummy_pci = np.random.random((1, 1))
                _ = self.model([dummy_features, dummy_pci])
                
                # Load weights
                self.model.load_weights(latest_model_path)
                
            else:
                logger.info("Creating new model")
                self.model = create_enhanced_model()
                
                # Build model with dummy data
                dummy_features = np.random.random((1, 50, 64))
                dummy_pci = np.random.random((1, 1))
                _ = self.model([dummy_features, dummy_pci])
                
                # Save initial model
                await self._save_model_version("v3.0-PCI-RL-initial")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    async def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        force_retrain: bool = False,
        hyperparameter_tuning: bool = False
    ) -> Dict[str, Any]:
        """Train the model with adaptive configuration"""
        try:
            logger.info("Starting adaptive model training...")
            
            # Validate training data
            if len(training_data) < self.min_training_samples:
                raise ValueError(f"Insufficient training data: {len(training_data)} < {self.min_training_samples}")
            
            # Check if retraining is needed
            if not force_retrain and not await self._should_retrain(training_data):
                logger.info("Retraining not needed based on current metrics")
                return self.performance_metrics
            
            # Prepare training data with PCI integration
            X, y, pci_values, sample_weights = await self._prepare_training_data(training_data)
            
            # Split data
            if validation_data is None:
                X_train, X_temp, y_train, y_temp, pci_train, pci_temp, weights_train, weights_temp = train_test_split(
                    X, y, pci_values, sample_weights,
                    test_size=self.training_config['validation_split'] + self.training_config['test_split'],
                    stratify=y,
                    random_state=42
                )
                
                X_val, X_test, y_val, y_test, pci_val, pci_test, weights_val, weights_test = train_test_split(
                    X_temp, y_temp, pci_temp, weights_temp,
                    test_size=self.training_config['test_split'] / (self.training_config['validation_split'] + self.training_config['test_split']),
                    stratify=y_temp,
                    random_state=42
                )
            else:
                # Use provided validation data
                X_val, y_val, pci_val, weights_val = await self._prepare_training_data(validation_data)
                X_train, X_test, y_train, y_test, pci_train, pci_test, weights_train, weights_test = train_test_split(
                    X, y, pci_values, sample_weights,
                    test_size=self.training_config['test_split'],
                    stratify=y,
                    random_state=42
                )
            
            # Hyperparameter tuning if requested
            if hyperparameter_tuning:
                best_config = await self._tune_hyperparameters(X_train, y_train, pci_train, X_val, y_val, pci_val)
                self.training_config.update(best_config)
            
            # Create model with optimized configuration
            self.model = create_enhanced_model(
                learning_rate=self.training_config['learning_rate']
            )
            
            # Prepare callbacks
            callbacks = create_enhanced_callbacks(
                model_path=f"{self.model_path}/checkpoints",
                patience=self.training_config['early_stopping_patience']
            )
            
            # Add custom callbacks
            callbacks.extend([
                PCIDriftMonitoringCallback(self.pci_calculator, self.pci_drift_threshold),
                AdaptiveLearningRateCallback(),
                ModelVersioningCallback(self.model_path)
            ])
            
            # Train model
            logger.info(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")
            
            history = self.model.fit(
                [X_train, pci_train],
                {'prediction': y_train, 'confidence': y_train},
                validation_data=([X_val, pci_val], {'prediction': y_val, 'confidence': y_val}),
                sample_weight={'prediction': weights_train, 'confidence': weights_train},
                epochs=self.training_config['epochs'],
                batch_size=self.training_config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate on test set
            test_metrics = await self._evaluate_model(X_test, y_test, pci_test)
            
            # Update performance tracking
            training_result = {
                'timestamp': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'final_train_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1],
                'test_metrics': test_metrics,
                'training_config': self.training_config.copy(),
                'model_version': f"v3.0-PCI-RL-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            self.training_history.append(training_result)
            self.performance_metrics = test_metrics
            self.last_training_time = datetime.now()
            
            # Save model version if performance is good
            if test_metrics['accuracy'] >= self.target_accuracy:
                await self._save_model_version(training_result['model_version'])
                logger.info(f"Model achieved target accuracy: {test_metrics['accuracy']:.4f}")
            else:
                logger.warning(f"Model accuracy {test_metrics['accuracy']:.4f} below target {self.target_accuracy}")
            
            # Log training completion
            logger.info(f"Training completed. Test accuracy: {test_metrics['accuracy']:.4f}")
            
            return training_result
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise
    
    async def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data with PCI integration"""
        try:
            X = []
            y = []
            pci_values = []
            sample_weights = []
            
            for _, row in df.iterrows():
                # Extract features
                player_data = row.get('player_data', {})
                opponent_data = row.get('opponent_data', {})
                battle_context = row.get('battle_context', {})
                
                features = self.feature_engineer.extract_features(
                    player_data=player_data,
                    opponent_data=opponent_data,
                    battle_context=battle_context
                )
                
                # Calculate PCI
                battle_history = player_data.get('battlelog', [])
                player_stats = player_data.get('stats', {})
                player_tag = player_data.get('tag', 'unknown')
                
                pci_value = self.pci_calculator.calculate_pci(
                    player_tag=player_tag,
                    battle_history=battle_history,
                    player_stats=player_stats
                )
                
                # Prepare feature vector
                feature_vector = self.feature_engineer.features_to_vector(features)
                
                # Create sequence input
                sequence_input = np.expand_dims(feature_vector, axis=0)
                sequence_input = np.expand_dims(sequence_input, axis=1)
                
                # Pad to expected sequence length
                target_seq_len = 50
                if sequence_input.shape[1] < target_seq_len:
                    padding = np.zeros((1, target_seq_len - sequence_input.shape[1], sequence_input.shape[2]))
                    sequence_input = np.concatenate([sequence_input, padding], axis=1)
                elif sequence_input.shape[1] > target_seq_len:
                    sequence_input = sequence_input[:, :target_seq_len, :]
                
                X.append(sequence_input[0])
                y.append(1.0 if row.get('result', 'loss') == 'win' else 0.0)
                pci_values.append(pci_value)
                
                # Calculate sample weight based on PCI (give more weight to consistent players)
                sample_weight = 1.0 + abs(pci_value - 0.5)  # Weight between 0.5 and 1.5
                sample_weights.append(sample_weight)
            
            return (
                np.array(X),
                np.array(y),
                np.array(pci_values).reshape(-1, 1),
                np.array(sample_weights)
            )
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    async def _should_retrain(self, training_data: pd.DataFrame) -> bool:
        """Determine if model retraining is needed"""
        try:
            if not self.auto_retrain_enabled:
                return False
            
            # Time-based trigger (weekly)
            if self.last_training_time:
                time_since_training = datetime.now() - self.last_training_time
                if time_since_training.days >= 7:
                    logger.info("Weekly retraining trigger activated")
                    return True
            
            # Performance degradation trigger
            if self.performance_metrics and 'accuracy' in self.performance_metrics:
                current_accuracy = self.performance_metrics['accuracy']
                if current_accuracy < self.target_accuracy - 0.05:  # 5% below target
                    logger.info(f"Performance degradation trigger: {current_accuracy:.4f} < {self.target_accuracy - 0.05:.4f}")
                    return True
            
            # PCI drift detection
            if await self._detect_pci_drift(training_data):
                logger.info("PCI drift detected, triggering retraining")
                return True
            
            # Data volume trigger (significant new data)
            if self.last_training_time:
                new_data_count = len(training_data)  # Simplified - should check actual new samples
                if new_data_count > self.min_training_samples * 0.5:  # 50% new data
                    logger.info(f"New data trigger: {new_data_count} new samples")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
            return False
    
    async def _detect_pci_drift(self, training_data: pd.DataFrame) -> bool:
        """Detect PCI distribution drift"""
        try:
            if len(training_data) < 100:
                return False
            
            # Calculate PCI for recent data
            recent_pci_values = []
            for _, row in training_data.tail(100).iterrows():
                player_data = row.get('player_data', {})
                battle_history = player_data.get('battlelog', [])
                player_stats = player_data.get('stats', {})
                player_tag = player_data.get('tag', 'unknown')
                
                pci_value = self.pci_calculator.calculate_pci(
                    player_tag=player_tag,
                    battle_history=battle_history,
                    player_stats=player_stats
                )
                recent_pci_values.append(pci_value)
            
            # Compare with historical PCI distribution (simplified)
            if hasattr(self, '_historical_pci_mean'):
                recent_mean = np.mean(recent_pci_values)
                drift_magnitude = abs(recent_mean - self._historical_pci_mean)
                
                if drift_magnitude > self.pci_drift_threshold:
                    logger.info(f"PCI drift detected: {drift_magnitude:.4f} > {self.pci_drift_threshold}")
                    return True
            
            # Update historical mean
            self._historical_pci_mean = np.mean(recent_pci_values)
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting PCI drift: {e}")
            return False
    
    async def _tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        pci_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        pci_val: np.ndarray
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        try:
            logger.info("Starting hyperparameter tuning...")
            
            # Define hyperparameter search space
            param_grid = {
                'learning_rate': [0.0005, 0.001, 0.002],
                'batch_size': [16, 32, 64],
                'dropout_rate': [0.1, 0.2, 0.3]
            }
            
            best_score = 0.0
            best_params = self.training_config.copy()
            
            # Grid search (simplified)
            for lr in param_grid['learning_rate']:
                for batch_size in param_grid['batch_size']:
                    for dropout in param_grid['dropout_rate']:
                        try:
                            # Create model with current parameters
                            test_model = create_enhanced_model(learning_rate=lr)
                            
                            # Quick training (few epochs)
                            history = test_model.fit(
                                [X_train, pci_train],
                                {'prediction': y_train, 'confidence': y_train},
                                validation_data=([X_val, pci_val], {'prediction': y_val, 'confidence': y_val}),
                                epochs=5,
                                batch_size=batch_size,
                                verbose=0
                            )
                            
                            # Get validation score
                            val_score = max(history.history.get('val_accuracy', [0.0]))
                            
                            if val_score > best_score:
                                best_score = val_score
                                best_params.update({
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'dropout_rate': dropout
                                })
                            
                            logger.debug(f"Params: lr={lr}, batch={batch_size}, dropout={dropout}, score={val_score:.4f}")
                            
                        except Exception as e:
                            logger.warning(f"Error testing hyperparameters: {e}")
                            continue
            
            logger.info(f"Best hyperparameters found: {best_params}, score: {best_score:.4f}")
            return best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            return self.training_config.copy()
    
    async def _evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        pci_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Get predictions
            predictions = self.model([X_test, pci_test])
            y_pred_proba = predictions['prediction'].numpy()
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            # PCI-stratified performance
            pci_bins = np.digitize(pci_test.flatten(), bins=[0.0, 0.25, 0.5, 0.75, 1.0])
            for i, bin_name in enumerate(['very_low', 'low', 'medium', 'high'], 1):
                mask = pci_bins == i
                if np.sum(mask) > 0:
                    bin_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                    metrics[f'accuracy_pci_{bin_name}'] = bin_accuracy
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'accuracy': 0.0}
    
    async def _save_model_version(self, version_name: str):
        """Save model version with metadata"""
        try:
            version_path = f"{self.model_path}/versions/{version_name}"
            os.makedirs(version_path, exist_ok=True)
            
            # Save model weights
            self.model.save_weights(f"{version_path}/model_weights.h5")
            
            # Save feature scaler
            self.feature_engineer.save_scaler(f"{version_path}/feature_scaler.pkl")
            
            # Save metadata
            metadata = {
                'version': version_name,
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self.performance_metrics,
                'training_config': self.training_config,
                'model_architecture': 'EnhancedTransformerLSTM-PCI-RL'
            }
            
            import json
            with open(f"{version_path}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.model_versions.append(version_name)
            logger.info(f"Model version saved: {version_name}")
            
        except Exception as e:
            logger.error(f"Error saving model version: {e}")
    
    def _get_latest_model_path(self) -> Optional[str]:
        """Get path to latest model version"""
        try:
            versions_dir = f"{self.model_path}/versions"
            if not os.path.exists(versions_dir):
                return None
            
            versions = [d for d in os.listdir(versions_dir) if os.path.isdir(os.path.join(versions_dir, d))]
            if not versions:
                return None
            
            # Sort by creation time
            versions.sort(key=lambda x: os.path.getctime(os.path.join(versions_dir, x)), reverse=True)
            latest_version = versions[0]
            
            model_path = f"{versions_dir}/{latest_version}/model_weights.h5"
            return model_path if os.path.exists(model_path) else None
            
        except Exception as e:
            logger.error(f"Error getting latest model path: {e}")
            return None
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training pipeline status"""
        return {
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'auto_retrain_enabled': self.auto_retrain_enabled,
            'target_accuracy': self.target_accuracy,
            'current_performance': self.performance_metrics,
            'model_versions_count': len(self.model_versions),
            'training_history_count': len(self.training_history)
        }

# Custom Callbacks

class PCIDriftMonitoringCallback(tf.keras.callbacks.Callback):
    """Monitor PCI drift during training"""
    
    def __init__(self, pci_calculator, drift_threshold):
        super().__init__()
        self.pci_calculator = pci_calculator
        self.drift_threshold = drift_threshold
        self.pci_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Simplified PCI drift monitoring
        if logs and 'val_accuracy' in logs:
            val_acc = logs['val_accuracy']
            if len(self.pci_history) > 5:
                recent_trend = np.mean(self.pci_history[-5:])
                if abs(val_acc - recent_trend) > 0.1:
                    logger.warning(f"Potential PCI-related performance drift at epoch {epoch + 1}")
            
            self.pci_history.append(val_acc)

class AdaptiveLearningRateCallback(tf.keras.callbacks.Callback):
    """Adaptive learning rate based on PCI distribution"""
    
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def on_epoch_end(self, epoch, logs=None):
        if logs and 'val_loss' in logs:
            val_loss = logs['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= 3:
                    old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                    new_lr = old_lr * 0.8
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                    logger.info(f"Adaptive LR: {old_lr:.6f} -> {new_lr:.6f}")
                    self.patience_counter = 0

class ModelVersioningCallback(tf.keras.callbacks.Callback):
    """Automatic model versioning during training"""
    
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.best_val_auc = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        if logs and 'val_auc' in logs:
            val_auc = logs['val_auc']
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                # Save checkpoint
                checkpoint_path = f"{self.model_path}/checkpoints/best_epoch_{epoch + 1}_auc_{val_auc:.4f}.h5"
                self.model.save_weights(checkpoint_path)
                logger.debug(f"Checkpoint saved: {checkpoint_path}")
