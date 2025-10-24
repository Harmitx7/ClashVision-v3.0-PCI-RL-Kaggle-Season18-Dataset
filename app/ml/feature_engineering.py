import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Card mappings and metadata
        self.card_metadata = self._load_card_metadata()
        self.deck_archetypes = self._load_deck_archetypes()
    
    def extract_features(
        self,
        player_data: Dict[str, Any],
        opponent_data: Optional[Dict[str, Any]] = None,
        battle_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract features for prediction"""
        try:
            features = {}
            
            # Player features
            features.update(self._extract_player_features(player_data))
            
            # Opponent features (if available)
            if opponent_data:
                features.update(self._extract_opponent_features(opponent_data))
            
            # Battle context features
            if battle_context:
                features.update(self._extract_battle_context_features(battle_context))
            
            # Deck analysis features
            if "currentDeck" in player_data or "cards" in player_data:
                deck_cards = player_data.get("currentDeck", player_data.get("cards", []))
                features.update(self._extract_deck_features(deck_cards))
            
            # Interaction features
            if opponent_data:
                features.update(self._extract_interaction_features(player_data, opponent_data))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def extract_live_features(
        self,
        player_tag: str,
        battle_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract features from live battle data"""
        try:
            features = {}
            
            # Battle state features
            features["battle_time_elapsed"] = battle_data.get("time_elapsed", 0)
            features["battle_time_remaining"] = battle_data.get("time_remaining", 180)
            features["battle_phase"] = self._encode_battle_phase(battle_data.get("time_remaining", 180))
            
            # Tower status
            features["player_towers_remaining"] = battle_data.get("player_towers", 3)
            features["opponent_towers_remaining"] = battle_data.get("opponent_towers", 3)
            features["tower_advantage"] = features["player_towers_remaining"] - features["opponent_towers_remaining"]
            
            # Elixir management
            features["current_elixir"] = battle_data.get("player_elixir", 10)
            features["opponent_elixir"] = battle_data.get("opponent_elixir", 10)
            features["elixir_advantage"] = features["current_elixir"] - features["opponent_elixir"]
            
            # Card cycle
            features["cards_in_hand"] = len(battle_data.get("cards_in_hand", []))
            features["next_card"] = battle_data.get("next_card", "")
            
            # Damage dealt
            features["damage_dealt"] = battle_data.get("damage_dealt", 0)
            features["damage_received"] = battle_data.get("damage_received", 0)
            features["damage_ratio"] = (
                features["damage_dealt"] / max(1, features["damage_received"])
            )
            
            # Troop deployment
            features["troops_deployed"] = battle_data.get("troops_deployed", 0)
            features["spells_used"] = battle_data.get("spells_used", 0)
            features["buildings_placed"] = battle_data.get("buildings_placed", 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting live features: {e}")
            return {}
    
    def _extract_player_features(self, player_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract player-specific features"""
        features = {}
        
        # Basic stats
        features["player_trophies"] = player_data.get("trophies", 0)
        features["player_best_trophies"] = player_data.get("bestTrophies", 0)
        features["player_level"] = player_data.get("expLevel", 1)
        features["player_wins"] = player_data.get("wins", 0)
        features["player_losses"] = player_data.get("losses", 0)
        features["player_three_crown_wins"] = player_data.get("threeCrownWins", 0)
        
        # Calculated stats
        total_battles = features["player_wins"] + features["player_losses"]
        features["player_win_rate"] = (
            features["player_wins"] / max(1, total_battles)
        )
        features["player_three_crown_rate"] = (
            features["player_three_crown_wins"] / max(1, features["player_wins"])
        )
        
        # Arena and league
        features["player_arena_id"] = player_data.get("arena", {}).get("id", 0)
        features["player_league_id"] = player_data.get("league", {}).get("id", 0)
        
        # Clan features
        clan_data = player_data.get("clan", {})
        features["in_clan"] = 1 if clan_data else 0
        features["clan_score"] = clan_data.get("clanScore", 0)
        features["clan_war_trophies"] = clan_data.get("clanWarTrophies", 0)
        
        # Card collection
        features["cards_found"] = player_data.get("cardsFound", 0)
        features["total_donations"] = player_data.get("totalDonations", 0)
        
        return features
    
    def _extract_opponent_features(self, opponent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract opponent-specific features"""
        features = {}
        
        # Basic stats with opponent prefix
        features["opponent_trophies"] = opponent_data.get("trophies", 0)
        features["opponent_best_trophies"] = opponent_data.get("bestTrophies", 0)
        features["opponent_level"] = opponent_data.get("expLevel", 1)
        features["opponent_wins"] = opponent_data.get("wins", 0)
        features["opponent_losses"] = opponent_data.get("losses", 0)
        
        # Calculated opponent stats
        total_battles = features["opponent_wins"] + features["opponent_losses"]
        features["opponent_win_rate"] = (
            features["opponent_wins"] / max(1, total_battles)
        )
        
        # Relative features
        features["trophy_difference"] = (
            features.get("player_trophies", 0) - features["opponent_trophies"]
        )
        features["level_difference"] = (
            features.get("player_level", 1) - features["opponent_level"]
        )
        
        return features
    
    def _extract_battle_context_features(self, battle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract battle context features"""
        features = {}
        
        # Battle type
        battle_type = battle_context.get("type", "PvP")
        features["is_ladder"] = 1 if battle_type == "PvP" else 0
        features["is_tournament"] = 1 if "tournament" in battle_type.lower() else 0
        features["is_challenge"] = 1 if "challenge" in battle_type.lower() else 0
        
        # Arena
        features["battle_arena_id"] = battle_context.get("arena", {}).get("id", 0)
        
        # Game mode
        game_mode = battle_context.get("gameMode", {})
        features["game_mode_id"] = game_mode.get("id", 0)
        
        # Deck selection
        deck_selection = battle_context.get("deckSelection", "")
        features["is_draft"] = 1 if "draft" in deck_selection.lower() else 0
        
        return features
    
    def _extract_deck_features(self, deck_cards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract deck composition features"""
        features = {}
        
        if not deck_cards:
            return features
        
        # Basic deck stats
        elixir_costs = [card.get("elixirCost", 0) for card in deck_cards if "elixirCost" in card]
        card_levels = [card.get("level", 1) for card in deck_cards if "level" in card]
        
        features["deck_size"] = len(deck_cards)
        features["average_elixir_cost"] = np.mean(elixir_costs) if elixir_costs else 0
        features["elixir_cost_std"] = np.std(elixir_costs) if elixir_costs else 0
        features["average_card_level"] = np.mean(card_levels) if card_levels else 1
        features["card_level_std"] = np.std(card_levels) if card_levels else 0
        
        # Card type distribution
        card_types = [self._get_card_type(card.get("name", "")) for card in deck_cards]
        features["troop_count"] = card_types.count("troop")
        features["spell_count"] = card_types.count("spell")
        features["building_count"] = card_types.count("building")
        
        # Rarity distribution
        rarities = [self._get_card_rarity(card.get("name", "")) for card in deck_cards]
        features["common_count"] = rarities.count("common")
        features["rare_count"] = rarities.count("rare")
        features["epic_count"] = rarities.count("epic")
        features["legendary_count"] = rarities.count("legendary")
        
        # Deck archetype
        features["deck_archetype"] = self._identify_deck_archetype(deck_cards)
        
        # Synergy score
        features["deck_synergy_score"] = self._calculate_deck_synergy(deck_cards)
        
        return features
    
    def _extract_interaction_features(
        self,
        player_data: Dict[str, Any],
        opponent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract player vs opponent interaction features"""
        features = {}
        
        # Get decks
        player_deck = player_data.get("currentDeck", player_data.get("cards", []))
        opponent_deck = opponent_data.get("currentDeck", opponent_data.get("cards", []))
        
        if player_deck and opponent_deck:
            # Counter analysis
            features["counter_score"] = self._calculate_counter_score(player_deck, opponent_deck)
            features["opponent_counter_score"] = self._calculate_counter_score(opponent_deck, player_deck)
            
            # Deck similarity
            features["deck_similarity"] = self._calculate_deck_similarity(player_deck, opponent_deck)
            
            # Elixir advantage potential
            features["elixir_advantage_potential"] = self._calculate_elixir_advantage_potential(
                player_deck, opponent_deck
            )
        
        return features
    
    def _get_card_type(self, card_name: str) -> str:
        """Get card type from metadata"""
        return self.card_metadata.get(card_name, {}).get("type", "troop")
    
    def _get_card_rarity(self, card_name: str) -> str:
        """Get card rarity from metadata"""
        return self.card_metadata.get(card_name, {}).get("rarity", "common")
    
    def _identify_deck_archetype(self, deck_cards: List[Dict[str, Any]]) -> int:
        """Identify deck archetype and return encoded value"""
        card_names = [card.get("name", "") for card in deck_cards]
        
        # Simple archetype identification based on key cards
        for archetype_id, archetype_data in self.deck_archetypes.items():
            key_cards = archetype_data.get("key_cards", [])
            matches = sum(1 for card in key_cards if card in card_names)
            
            if matches >= archetype_data.get("min_matches", 2):
                return archetype_id
        
        return 0  # Unknown archetype
    
    def _calculate_deck_synergy(self, deck_cards: List[Dict[str, Any]]) -> float:
        """Calculate deck synergy score"""
        if len(deck_cards) < 2:
            return 0.0
        
        # Simplified synergy calculation
        # In a real implementation, this would use card interaction matrices
        
        synergy_score = 0.0
        card_names = [card.get("name", "") for card in deck_cards]
        
        # Check for known synergies
        synergy_pairs = [
            ("Giant", "Wizard"),
            ("Hog Rider", "Freeze"),
            ("Golem", "Night Witch"),
            ("X-Bow", "Tesla"),
            ("Miner", "Poison")
        ]
        
        for card1, card2 in synergy_pairs:
            if card1 in card_names and card2 in card_names:
                synergy_score += 0.2
        
        # Normalize to 0-1 range
        return min(1.0, synergy_score)
    
    def _calculate_counter_score(
        self,
        player_deck: List[Dict[str, Any]],
        opponent_deck: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well player deck counters opponent deck"""
        if not player_deck or not opponent_deck:
            return 0.5
        
        # Simplified counter calculation
        # In a real implementation, this would use detailed counter matrices
        
        player_cards = [card.get("name", "") for card in player_deck]
        opponent_cards = [card.get("name", "") for card in opponent_deck]
        
        counter_score = 0.0
        total_matchups = 0
        
        # Simple counter relationships
        counters = {
            "Fireball": ["Wizard", "Witch", "Musketeer"],
            "Lightning": ["Inferno Tower", "Sparky"],
            "Freeze": ["Inferno Tower", "Inferno Dragon"],
            "Rocket": ["Elixir Collector", "Sparky"]
        }
        
        for player_card in player_cards:
            if player_card in counters:
                for opponent_card in opponent_cards:
                    if opponent_card in counters[player_card]:
                        counter_score += 1.0
                    total_matchups += 1
        
        return counter_score / max(1, total_matchups)
    
    def _calculate_deck_similarity(
        self,
        deck1: List[Dict[str, Any]],
        deck2: List[Dict[str, Any]]
    ) -> float:
        """Calculate similarity between two decks"""
        if not deck1 or not deck2:
            return 0.0
        
        cards1 = set(card.get("name", "") for card in deck1)
        cards2 = set(card.get("name", "") for card in deck2)
        
        intersection = len(cards1.intersection(cards2))
        union = len(cards1.union(cards2))
        
        return intersection / max(1, union)  # Jaccard similarity
    
    def _calculate_elixir_advantage_potential(
        self,
        player_deck: List[Dict[str, Any]],
        opponent_deck: List[Dict[str, Any]]
    ) -> float:
        """Calculate potential for elixir advantages"""
        if not player_deck or not opponent_deck:
            return 0.0
        
        player_avg_cost = np.mean([card.get("elixirCost", 0) for card in player_deck])
        opponent_avg_cost = np.mean([card.get("elixirCost", 0) for card in opponent_deck])
        
        # Lower cost deck has potential for elixir advantages
        cost_advantage = (opponent_avg_cost - player_avg_cost) / 10.0
        
        return max(0.0, min(1.0, cost_advantage + 0.5))
    
    def _encode_battle_phase(self, time_remaining: int) -> int:
        """Encode battle phase as integer"""
        if time_remaining > 120:
            return 0  # early_game
        elif time_remaining > 60:
            return 1  # mid_game
        elif time_remaining > 0:
            return 2  # late_game
        else:
            return 3  # overtime
    
    def features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy vector"""
        try:
            # Define expected feature order
            if not self.feature_names:
                self.feature_names = sorted(features.keys())
            
            # Create feature vector
            vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0.0)
                
                # Handle different data types
                if isinstance(value, (int, float)):
                    vector.append(float(value))
                elif isinstance(value, str):
                    # Encode string features
                    if feature_name not in self.label_encoders:
                        self.label_encoders[feature_name] = LabelEncoder()
                        # Fit with dummy data if not fitted
                        self.label_encoders[feature_name].fit([value, "unknown"])
                    
                    try:
                        encoded_value = self.label_encoders[feature_name].transform([value])[0]
                    except ValueError:
                        encoded_value = 0  # Unknown category
                    
                    vector.append(float(encoded_value))
                else:
                    vector.append(0.0)
            
            # Ensure consistent vector size
            target_size = 64  # Expected feature size
            if len(vector) < target_size:
                vector.extend([0.0] * (target_size - len(vector)))
            elif len(vector) > target_size:
                vector = vector[:target_size]
            
            return np.array(vector, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error converting features to vector: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from DataFrame"""
        try:
            # Extract features for each row
            X = []
            y = []
            
            for _, row in df.iterrows():
                # Extract features (this would depend on your data format)
                features = self.extract_features(
                    player_data=row.get("player_data", {}),
                    opponent_data=row.get("opponent_data", {}),
                    battle_context=row.get("battle_context", {})
                )
                
                feature_vector = self.features_to_vector(features)
                X.append(feature_vector)
                
                # Extract target (win/loss)
                result = row.get("result", "loss")
                y.append(1.0 if result == "win" else 0.0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            if not self.is_fitted:
                X = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                X = self.scaler.transform(X)
            
            # Reshape for sequence input
            X = np.expand_dims(X, axis=1)  # Add sequence dimension
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def initialize_scaler(self):
        """Initialize scaler with dummy data"""
        dummy_data = np.random.random((100, 64))
        self.scaler.fit(dummy_data)
        self.is_fitted = True
    
    def save_scaler(self, filepath: str):
        """Save scaler to file"""
        try:
            scaler_data = {
                "scaler": self.scaler,
                "label_encoders": self.label_encoders,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(scaler_data, f)
                
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")
    
    def load_scaler(self, filepath: str):
        """Load scaler from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    scaler_data = pickle.load(f)
                
                self.scaler = scaler_data["scaler"]
                self.label_encoders = scaler_data["label_encoders"]
                self.feature_names = scaler_data["feature_names"]
                self.is_fitted = scaler_data["is_fitted"]
            else:
                self.initialize_scaler()
                
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            self.initialize_scaler()
    
    def _load_card_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load card metadata (simplified)"""
        # In a real implementation, this would load from a comprehensive card database
        return {
            "Giant": {"type": "troop", "rarity": "rare", "elixir_cost": 5},
            "Wizard": {"type": "troop", "rarity": "rare", "elixir_cost": 5},
            "Fireball": {"type": "spell", "rarity": "rare", "elixir_cost": 4},
            "Hog Rider": {"type": "troop", "rarity": "rare", "elixir_cost": 4},
            # Add more cards as needed
        }
    
    def _load_deck_archetypes(self) -> Dict[int, Dict[str, Any]]:
        """Load deck archetype definitions"""
        return {
            1: {"name": "Beatdown", "key_cards": ["Giant", "Golem"], "min_matches": 1},
            2: {"name": "Control", "key_cards": ["X-Bow", "Mortar"], "min_matches": 1},
            3: {"name": "Cycle", "key_cards": ["Hog Rider", "Ice Spirit"], "min_matches": 1},
            4: {"name": "Siege", "key_cards": ["X-Bow", "Tesla"], "min_matches": 2},
            # Add more archetypes as needed
        }
