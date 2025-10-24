#!/usr/bin/env python3
"""
ClashVision v3.0-PCI-RL Main Entry Point
========================================

Main application entry point that works without database dependencies
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.ml.predictor import WinPredictor
from app.core.database import is_database_available

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Main application function"""
    print("🎯 ClashVision v3.0-PCI-RL")
    print("=" * 30)
    
    try:
        # Check system status
        print("📊 System Status:")
        print(f"   Database: {'✅ Available' if is_database_available() else '⚠️  Not configured'}")
        
        # Initialize predictor with Kaggle integration
        print("\n🤖 Initializing Production Predictor...")
        predictor = WinPredictor()
        predictor.use_enhanced_model = True  # Use enhanced model with Kaggle integration
        predictor.auto_train_with_kaggle = True  # Enable auto-training
        await predictor.initialize()
        
        print("✅ Predictor initialized successfully")
        
        # Demo prediction
        print("\n🔮 Demo Prediction:")
        sample_player = {
            'trophies': 5000,
            'wins': 150,
            'losses': 100,
            'expLevel': 11
        }
        
        result = await predictor.predict(sample_player)
        
        print(f"   Player: {sample_player['trophies']} trophies, {sample_player['wins']}/{sample_player['losses']} W/L")
        print(f"   Win Probability: {result['win_probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Model Version: {result['model_version']}")
        
        # Show available features and Kaggle integration status
        print(f"\n🚀 Next Steps:")
        print(f"1. Open frontend: frontend/index.html")
        print(f"2. Start API server: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print(f"3. View strategic analysis in real-time")
        print(f"4. Train with Kaggle data: python train_with_kaggle_data.py --sample-size 10000")
        
        print(f"\n🎯 Frontend Features Available:")
        print(f"   ✅ Player Consistency Index (PCI) with animated gauge")
        print(f"   ✅ Strategic Battle Tactics with real-time updates")
        print(f"   ✅ Smart Card Suggestions (Add/Remove with reasons)")
        print(f"   ✅ Counter Strategies for current meta")
        print(f"   ✅ Meta Insights with trending cards")
        print(f"   ✅ Hybrid Training Status (Kaggle + Recent matches)")
        
        print(f"\n📊 Strategic Analysis Dashboard:")
        print(f"   🎯 Battle Tactics: Real-time strategic recommendations")
        print(f"   🃏 Card Suggestions: AI-powered deck optimization")
        print(f"   📈 Meta Intelligence: Trending cards and adaptations")
        print(f"   🛡️ Counter Strategies: Specific plays against meta decks")
        
        # Show enhanced metrics
        metrics = predictor.get_enhanced_metrics()
        if 'kaggle_dataset' in metrics:
            kaggle_stats = metrics['kaggle_dataset']
            print(f"\n📊 Kaggle Dataset Status:")
            print(f"   Total Matches: {kaggle_stats.get('dataset_info', {}).get('total_matches', 0):,}")
            print(f"   Processed: {kaggle_stats.get('dataset_info', {}).get('processed_matches', 0):,}")
            print(f"   Card Mappings: {kaggle_stats.get('card_mappings', {}).get('total_cards', 0)}")
        
        # Show hybrid training status
        if 'hybrid_training' in metrics:
            hybrid_stats = metrics['hybrid_training']
            print(f"\n🔄 Hybrid Training Status:")
            print(f"   Recent Matches: {hybrid_stats.get('recent_matches_collected', 0):,}")
            print(f"   Buffer Capacity: {hybrid_stats.get('recent_match_coverage', '0/10000')}")
            print(f"   Training Mode: {'Kaggle + Recent' if hybrid_stats.get('enabled') else 'Kaggle Only'}")
        
        if predictor.kaggle_integration:
            print(f"   🎯 Kaggle Integration: ACTIVE")
        else:
            print(f"   ⚠️  Kaggle Integration: Not Available")
        
        if predictor.use_hybrid_training:
            print(f"   🔄 Hybrid Training: ENABLED (Higher Accuracy)")
        
        print(f"\n🎯 ClashVision is ready for predictions!")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

async def predict_for_player(predictor: WinPredictor, player_data: dict):
    """Make a prediction for a player"""
    try:
        result = await predictor.predict(player_data)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

if __name__ == "__main__":
    # Run the main application
    predictor = asyncio.run(main())
    
    # Interactive mode (optional)
    print(f"\n" + "=" * 50)
    print(f"ClashVision v3.0-PCI-RL is now running!")
    print(f"Use the predictor object to make predictions.")
    print(f"Example: await predict_for_player(predictor, player_data)")
    print(f"=" * 50)
