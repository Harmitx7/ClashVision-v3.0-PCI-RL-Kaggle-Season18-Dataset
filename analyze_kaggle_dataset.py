#!/usr/bin/env python3
"""
Kaggle Dataset Analysis and Validation Script
==============================================

Analyzes the Clash Royale Season 18 dataset to understand:
- Data distribution and quality
- Card usage patterns
- Trophy ranges and player levels
- PCI distribution validation
- Training data suitability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import asyncio
import logging
from typing import Dict, List, Any

from app.ml.kaggle_data_integration import KaggleDataIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleDatasetAnalyzer:
    """Analyzes the Kaggle Clash Royale dataset"""
    
    def __init__(self, dataset_path: str = "data/kaggle_clash_royale_s18/"):
        self.dataset_path = Path(dataset_path)
        self.kaggle_integration = KaggleDataIntegration(dataset_path)
        
    async def analyze_raw_dataset(self, sample_size: int = 50000):
        """Analyze raw dataset structure and content"""
        try:
            logger.info("Analyzing raw Kaggle dataset...")
            
            # Load sample of raw data
            battles_file = self.dataset_path / "battles.csv"
            if not battles_file.exists():
                logger.error("Dataset not found. Please download first.")
                return
            
            # Read sample
            df = pd.read_csv(battles_file, nrows=sample_size)
            logger.info(f"Loaded {len(df):,} sample battles")
            
            # Basic statistics
            print("\n=== DATASET OVERVIEW ===")
            print(f"Sample size: {len(df):,} battles")
            print(f"Columns: {list(df.columns)}")
            print(f"Data types:\n{df.dtypes}")
            print(f"Missing values:\n{df.isnull().sum()}")
            
            # Analyze battle outcomes
            self._analyze_battle_outcomes(df)
            
            # Analyze trophy distributions
            self._analyze_trophy_distribution(df)
            
            # Analyze card usage
            await self._analyze_card_usage(df)
            
            # Analyze game modes
            self._analyze_game_modes(df)
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
    
    def _analyze_battle_outcomes(self, df: pd.DataFrame):
        """Analyze battle outcomes and crown distributions"""
        print("\n=== BATTLE OUTCOMES ===")
        
        # Extract crown counts
        team_crowns = []
        opponent_crowns = []
        
        for _, battle in df.iterrows():
            team = battle.get('team', [])
            opponent = battle.get('opponent', [])
            
            if team and opponent:
                team_crowns.append(team[0].get('crowns', 0) if isinstance(team, list) else team.get('crowns', 0))
                opponent_crowns.append(opponent[0].get('crowns', 0) if isinstance(opponent, list) else opponent.get('crowns', 0))
        
        team_crowns = np.array(team_crowns)
        opponent_crowns = np.array(opponent_crowns)
        
        # Calculate win rates
        wins = np.sum(team_crowns > opponent_crowns)
        losses = np.sum(team_crowns < opponent_crowns)
        draws = np.sum(team_crowns == opponent_crowns)
        
        print(f"Wins: {wins:,} ({wins/len(team_crowns)*100:.1f}%)")
        print(f"Losses: {losses:,} ({losses/len(team_crowns)*100:.1f}%)")
        print(f"Draws: {draws:,} ({draws/len(team_crowns)*100:.1f}%)")
        
        # Crown distribution
        print(f"\nCrown distribution:")
        print(f"0 crowns: {np.sum(team_crowns == 0):,}")
        print(f"1 crown: {np.sum(team_crowns == 1):,}")
        print(f"2 crowns: {np.sum(team_crowns == 2):,}")
        print(f"3 crowns: {np.sum(team_crowns == 3):,}")
    
    def _analyze_trophy_distribution(self, df: pd.DataFrame):
        """Analyze trophy distribution across players"""
        print("\n=== TROPHY DISTRIBUTION ===")
        
        trophies = []
        for _, battle in df.iterrows():
            team = battle.get('team', [])
            opponent = battle.get('opponent', [])
            
            if team:
                player = team[0] if isinstance(team, list) else team
                trophies.append(player.get('startingTrophies', 0))
            
            if opponent:
                player = opponent[0] if isinstance(opponent, list) else opponent
                trophies.append(player.get('startingTrophies', 0))
        
        trophies = np.array(trophies)
        trophies = trophies[trophies > 0]  # Remove invalid entries
        
        print(f"Trophy statistics:")
        print(f"Mean: {np.mean(trophies):.0f}")
        print(f"Median: {np.median(trophies):.0f}")
        print(f"Min: {np.min(trophies):.0f}")
        print(f"Max: {np.max(trophies):.0f}")
        print(f"Std: {np.std(trophies):.0f}")
        
        # Trophy ranges
        ranges = [
            (0, 1000, "Training"),
            (1000, 2000, "Bronze"),
            (2000, 3000, "Silver"), 
            (3000, 4000, "Gold"),
            (4000, 5000, "Challenger I-II"),
            (5000, 6000, "Challenger III - Master I"),
            (6000, 7000, "Master II-III"),
            (7000, float('inf'), "Champion")
        ]
        
        print(f"\nTrophy range distribution:")
        for min_t, max_t, name in ranges:
            count = np.sum((trophies >= min_t) & (trophies < max_t))
            pct = count / len(trophies) * 100
            print(f"{name}: {count:,} ({pct:.1f}%)")
    
    async def _analyze_card_usage(self, df: pd.DataFrame):
        """Analyze card usage patterns"""
        print("\n=== CARD USAGE ANALYSIS ===")
        
        # Load card mappings
        await self.kaggle_integration._load_card_mappings()
        
        card_usage = {}
        total_decks = 0
        
        for _, battle in df.iterrows():
            team = battle.get('team', [])
            opponent = battle.get('opponent', [])
            
            for player_data in [team, opponent]:
                if not player_data:
                    continue
                    
                player = player_data[0] if isinstance(player_data, list) else player_data
                cards = player.get('cards', [])
                
                if cards:
                    total_decks += 1
                    for card in cards:
                        card_id = card.get('id', 0)
                        card_name = self.kaggle_integration.card_id_mapping.get(card_id, f"Unknown_{card_id}")
                        card_usage[card_name] = card_usage.get(card_name, 0) + 1
        
        # Sort by usage
        sorted_cards = sorted(card_usage.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Total decks analyzed: {total_decks:,}")
        print(f"Unique cards found: {len(sorted_cards)}")
        print(f"\nTop 20 most used cards:")
        
        for i, (card_name, usage) in enumerate(sorted_cards[:20]):
            usage_pct = usage / total_decks * 100
            print(f"{i+1:2d}. {card_name:<20} {usage:,} ({usage_pct:.1f}%)")
    
    def _analyze_game_modes(self, df: pd.DataFrame):
        """Analyze game mode distribution"""
        print("\n=== GAME MODE ANALYSIS ===")
        
        game_modes = {}
        for _, battle in df.iterrows():
            game_mode = battle.get('gameMode', {})
            if isinstance(game_mode, dict):
                mode_name = game_mode.get('name', 'Unknown')
            else:
                mode_name = str(game_mode)
            
            game_modes[mode_name] = game_modes.get(mode_name, 0) + 1
        
        print(f"Game mode distribution:")
        for mode, count in sorted(game_modes.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(df) * 100
            print(f"{mode:<20} {count:,} ({pct:.1f}%)")
    
    async def validate_processed_data(self, processed_file: str):
        """Validate processed training data"""
        try:
            logger.info("Validating processed training data...")
            
            df = pd.read_parquet(processed_file)
            print(f"\n=== PROCESSED DATA VALIDATION ===")
            print(f"Total training samples: {len(df):,}")
            
            # Check data completeness
            print(f"\nData completeness:")
            for col in df.columns:
                missing = df[col].isnull().sum()
                print(f"{col}: {missing:,} missing ({missing/len(df)*100:.1f}%)")
            
            # Validate PCI distribution
            if 'pci_value' in df.columns:
                pci_values = df['pci_value'].dropna()
                print(f"\nPCI Distribution:")
                print(f"Mean: {pci_values.mean():.3f}")
                print(f"Std: {pci_values.std():.3f}")
                print(f"Min: {pci_values.min():.3f}")
                print(f"Max: {pci_values.max():.3f}")
                
                # PCI ranges
                ranges = [
                    (0.0, 0.25, "Very Unstable"),
                    (0.25, 0.5, "Unstable"),
                    (0.5, 0.75, "Stable"),
                    (0.75, 1.0, "Very Stable")
                ]
                
                for min_p, max_p, label in ranges:
                    count = np.sum((pci_values >= min_p) & (pci_values < max_p))
                    pct = count / len(pci_values) * 100
                    print(f"{label}: {count:,} ({pct:.1f}%)")
            
            # Validate win/loss distribution
            if 'result' in df.columns:
                results = df['result'].value_counts()
                print(f"\nWin/Loss distribution:")
                for result, count in results.items():
                    pct = count / len(df) * 100
                    print(f"{result}: {count:,} ({pct:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error validating processed data: {e}")

async def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Kaggle Clash Royale dataset')
    parser.add_argument('--dataset-path', default='data/kaggle_clash_royale_s18/')
    parser.add_argument('--sample-size', type=int, default=50000)
    parser.add_argument('--processed-file', help='Path to processed training data file')
    
    args = parser.parse_args()
    
    analyzer = KaggleDatasetAnalyzer(args.dataset_path)
    
    # Analyze raw dataset
    await analyzer.analyze_raw_dataset(args.sample_size)
    
    # Validate processed data if provided
    if args.processed_file:
        await analyzer.validate_processed_data(args.processed_file)

if __name__ == "__main__":
    asyncio.run(main())
