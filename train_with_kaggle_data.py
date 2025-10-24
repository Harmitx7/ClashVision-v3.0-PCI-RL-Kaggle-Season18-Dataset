#!/usr/bin/env python3
"""
ClashVision v3.0-PCI-RL Training Script with Kaggle Dataset
===========================================================

This script trains the enhanced ClashVision model using the massive
Kaggle Clash Royale Season 18 dataset (37.9M matches).

Usage:
    python train_with_kaggle_data.py --sample-size 100000 --epochs 50
    python train_with_kaggle_data.py --full-dataset --hyperparameter-tuning
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.ml.kaggle_data_integration import KaggleDataIntegration
from app.ml.adaptive_training_pipeline import AdaptiveTrainingPipeline
from app.ml.enhanced_predictor import EnhancedWinPredictor
from app.ml.monitoring_system import ModelMonitoringSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class KaggleTrainingOrchestrator:
    """Orchestrates the training process with Kaggle dataset"""
    
    def __init__(self, args):
        self.args = args
        self.kaggle_integration = KaggleDataIntegration(
            dataset_path=args.dataset_path,
            chunk_size=args.chunk_size,
            max_samples=args.sample_size if not args.full_dataset else None
        )
        self.training_pipeline = None
        self.enhanced_predictor = None
        self.monitoring_system = None
        
    async def initialize_components(self):
        """Initialize all training components"""
        try:
            logger.info("Initializing training components...")
            
            # Initialize training pipeline
            self.training_pipeline = AdaptiveTrainingPipeline(
                target_accuracy=0.91,
                pci_drift_threshold=0.10
            )
            await self.training_pipeline.initialize()
            
            # Initialize enhanced predictor
            self.enhanced_predictor = EnhancedWinPredictor()
            await self.enhanced_predictor.initialize()
            
            # Initialize monitoring system
            self.monitoring_system = ModelMonitoringSystem(self.enhanced_predictor)
            await self.monitoring_system.start_monitoring()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def prepare_dataset(self):
        """Prepare the Kaggle dataset for training"""
        try:
            logger.info("Preparing Kaggle dataset...")
            
            # Download and prepare dataset
            await self.kaggle_integration.download_and_prepare_dataset()
            
            # Create sample dataset if requested
            if self.args.sample_size and not self.args.full_dataset:
                logger.info(f"Creating sample dataset with {self.args.sample_size:,} matches")
                processed_file = await self.kaggle_integration.create_sample_dataset(
                    sample_size=self.args.sample_size
                )
            else:
                logger.info("Processing full dataset (this may take several hours)")
                processed_file = await self.kaggle_integration.process_dataset_for_training()
            
            # Get dataset statistics
            stats = self.kaggle_integration.get_dataset_statistics()
            logger.info(f"Dataset statistics: {stats}")
            
            return processed_file
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise
    
    async def train_model(self, processed_data_file: str):
        """Train the enhanced model"""
        try:
            logger.info("Starting model training with Kaggle dataset...")
            
            # Configure training parameters
            if self.args.epochs:
                self.training_pipeline.training_config['epochs'] = self.args.epochs
            
            if self.args.batch_size:
                self.training_pipeline.training_config['batch_size'] = self.args.batch_size
            
            if self.args.learning_rate:
                self.training_pipeline.training_config['learning_rate'] = self.args.learning_rate
            
            # Train the model
            training_result = await self.kaggle_integration.train_enhanced_model_with_kaggle_data(
                training_pipeline=self.training_pipeline,
                processed_data_file=processed_data_file
            )
            
            # Log training results
            logger.info("Training completed successfully!")
            logger.info(f"Model version: {training_result['model_version']}")
            logger.info(f"Final accuracy: {training_result['training_result']['test_metrics']['accuracy']:.4f}")
            
            # Save training results
            await self._save_training_results(training_result)
            
            return training_result
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    async def _save_training_results(self, training_result: dict):
        """Save training results and model artifacts"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(f"training_results_{timestamp}")
            results_dir.mkdir(exist_ok=True)
            
            # Save training results
            import json
            with open(results_dir / "training_results.json", 'w') as f:
                json.dump(training_result, f, indent=2, default=str)
            
            # Save model
            if self.enhanced_predictor:
                await self.enhanced_predictor.save_model()
            
            # Save monitoring data
            if self.monitoring_system:
                self.monitoring_system.save_monitoring_data(
                    str(results_dir / "monitoring_data.json")
                )
            
            logger.info(f"Training results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
    
    async def evaluate_model(self):
        """Evaluate the trained model"""
        try:
            logger.info("Evaluating trained model...")
            
            if self.enhanced_predictor:
                metrics = self.enhanced_predictor.get_performance_metrics()
                logger.info(f"Model performance metrics: {metrics}")
                
                # Test prediction
                sample_prediction = await self._test_sample_prediction()
                logger.info(f"Sample prediction test: {sample_prediction}")
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
    
    async def _test_sample_prediction(self):
        """Test the model with a sample prediction"""
        try:
            # Create sample player data
            sample_player_data = {
                'tag': '#TEST123',
                'trophies': 5000,
                'expLevel': 11,
                'currentDeck': [
                    {'name': 'Hog Rider', 'level': 9, 'elixirCost': 4},
                    {'name': 'Fireball', 'level': 9, 'elixirCost': 4},
                    {'name': 'Zap', 'level': 11, 'elixirCost': 2},
                    {'name': 'Musketeer', 'level': 9, 'elixirCost': 4},
                    {'name': 'Valkyrie', 'level': 9, 'elixirCost': 4},
                    {'name': 'Cannon', 'level': 9, 'elixirCost': 3},
                    {'name': 'Ice Spirit', 'level': 11, 'elixirCost': 1},
                    {'name': 'Skeletons', 'level': 11, 'elixirCost': 1}
                ]
            }
            
            sample_opponent_data = {
                'tag': '#OPPONENT123',
                'trophies': 4950,
                'expLevel': 10,
                'currentDeck': [
                    {'name': 'Giant', 'level': 8, 'elixirCost': 5},
                    {'name': 'Wizard', 'level': 8, 'elixirCost': 5},
                    {'name': 'Arrows', 'level': 10, 'elixirCost': 3},
                    {'name': 'Minions', 'level': 10, 'elixirCost': 3},
                    {'name': 'Barbarians', 'level': 10, 'elixirCost': 5},
                    {'name': 'Tombstone', 'level': 8, 'elixirCost': 3},
                    {'name': 'Lightning', 'level': 6, 'elixirCost': 6},
                    {'name': 'Mega Minion', 'level': 8, 'elixirCost': 3}
                ]
            }
            
            # Make prediction
            prediction = await self.enhanced_predictor.predict(
                player_data=sample_player_data,
                opponent_data=sample_opponent_data,
                battle_context={'type': 'PvP'}
            )
            
            return {
                'win_probability': prediction['win_probability'],
                'confidence': prediction['confidence'],
                'pci_value': prediction['pci_value'],
                'model_used': prediction['model_used']
            }
            
        except Exception as e:
            logger.error(f"Error in sample prediction test: {e}")
            return None

async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ClashVision v3.0 with Kaggle dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset-path', type=str, default='data/kaggle_clash_royale_s18/',
                       help='Path to store/load the Kaggle dataset')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of matches to use for training (default: use full dataset)')
    parser.add_argument('--full-dataset', action='store_true',
                       help='Use the full 37.9M match dataset')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for processing large dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Enable hyperparameter tuning')
    
    # Execution arguments
    parser.add_argument('--skip-dataset-prep', action='store_true',
                       help='Skip dataset preparation (use existing processed data)')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model, skip training')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.full_dataset and args.sample_size:
        logger.warning("Both --full-dataset and --sample-size specified. Using full dataset.")
        args.sample_size = None
    
    if not args.sample_size and not args.full_dataset:
        logger.info("No dataset size specified. Using sample of 100,000 matches.")
        args.sample_size = 100000
    
    try:
        # Initialize orchestrator
        orchestrator = KaggleTrainingOrchestrator(args)
        
        # Initialize components
        await orchestrator.initialize_components()
        
        if args.evaluate_only:
            # Only evaluate existing model
            await orchestrator.evaluate_model()
        else:
            # Full training pipeline
            if not args.skip_dataset_prep:
                # Prepare dataset
                processed_data_file = await orchestrator.prepare_dataset()
            else:
                # Use existing processed data
                processed_data_file = args.dataset_path + "/processed_training_data.parquet"
                if not os.path.exists(processed_data_file):
                    raise FileNotFoundError(f"Processed data file not found: {processed_data_file}")
            
            # Train model
            training_result = await orchestrator.train_model(processed_data_file)
            
            # Evaluate model
            await orchestrator.evaluate_model()
            
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Enhanced model v3.0-PCI-RL trained with Kaggle dataset")
            logger.info(f"Target accuracy (91%) {'ACHIEVED' if training_result['training_result']['test_metrics']['accuracy'] >= 0.91 else 'NOT ACHIEVED'}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the training pipeline
    asyncio.run(main())
