"""
Data Integrity and Validation System for ClashVision
Implements comprehensive validation rules and auto-fix mechanisms
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation and integrity system"""
    
    def __init__(self):
        self.validation_rules = {
            'player_data': self._validate_player_data,
            'battle_data': self._validate_battle_data,
            'prediction_data': self._validate_prediction_data,
            'pci_data': self._validate_pci_data
        }
        
        self.auto_fix_enabled = True
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'auto_fixes_applied': 0,
            'data_drops': 0
        }
    
    def validate_data(self, data: Dict[str, Any], data_type: str) -> Tuple[Dict[str, Any], bool]:
        """
        Validate data according to type-specific rules
        Returns: (validated_data, is_valid)
        """
        self.validation_stats['total_validations'] += 1
        
        try:
            if data_type not in self.validation_rules:
                logger.warning(f"No validation rules for data type: {data_type}")
                return data, True
            
            validator = self.validation_rules[data_type]
            validated_data, is_valid = validator(data)
            
            if not is_valid:
                self.validation_stats['failed_validations'] += 1
                logger.warning(f"Data validation failed for type: {data_type}")
            
            return validated_data, is_valid
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            self.validation_stats['failed_validations'] += 1
            return data, False
    
    def _validate_player_data(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Validate player data with auto-fix capabilities"""
        is_valid = True
        validated_data = data.copy()
        
        # Required fields validation
        required_fields = ['tag', 'name', 'trophies', 'expLevel']
        for field in required_fields:
            if field not in validated_data or validated_data[field] is None:
                if self.auto_fix_enabled:
                    validated_data[field] = self._get_default_value(field)
                    self.validation_stats['auto_fixes_applied'] += 1
                    logger.info(f"Auto-fixed missing field: {field}")
                else:
                    is_valid = False
        
        # Numeric field validation and clamping
        numeric_fields = {
            'trophies': (0, 10000),
            'bestTrophies': (0, 15000),
            'wins': (0, 1000000),
            'losses': (0, 1000000),
            'expLevel': (1, 50),
            'battleCount': (0, 1000000)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in validated_data:
                try:
                    value = float(validated_data[field])
                    if np.isnan(value) or np.isinf(value):
                        if self.auto_fix_enabled:
                            validated_data[field] = self._get_default_value(field)
                            self.validation_stats['auto_fixes_applied'] += 1
                        else:
                            is_valid = False
                    else:
                        # Clamp to valid range
                        clamped_value = max(min_val, min(max_val, value))
                        if clamped_value != value:
                            validated_data[field] = clamped_value
                            self.validation_stats['auto_fixes_applied'] += 1
                            logger.debug(f"Clamped {field} from {value} to {clamped_value}")
                except (ValueError, TypeError):
                    if self.auto_fix_enabled:
                        validated_data[field] = self._get_default_value(field)
                        self.validation_stats['auto_fixes_applied'] += 1
                    else:
                        is_valid = False
        
        # Validate current deck
        if 'currentDeck' in validated_data:
            validated_data['currentDeck'], deck_valid = self._validate_deck_data(validated_data['currentDeck'])
            if not deck_valid:
                is_valid = False
        
        # Cross-field validation
        if 'bestTrophies' in validated_data and 'trophies' in validated_data:
            if validated_data['bestTrophies'] < validated_data['trophies']:
                if self.auto_fix_enabled:
                    validated_data['bestTrophies'] = validated_data['trophies']
                    self.validation_stats['auto_fixes_applied'] += 1
                else:
                    is_valid = False
        
        return validated_data, is_valid
    
    def _validate_battle_data(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Validate battle log data"""
        is_valid = True
        validated_data = data.copy()
        
        # Validate timestamp
        if 'battleTime' in validated_data:
            try:
                battle_time = datetime.fromisoformat(validated_data['battleTime'].replace('Z', '+00:00'))
                # Check if battle is not too old (more than 30 days)
                if (datetime.now() - battle_time.replace(tzinfo=None)) > timedelta(days=30):
                    logger.warning("Battle data is older than 30 days")
                    if not self.auto_fix_enabled:
                        is_valid = False
            except (ValueError, AttributeError):
                if self.auto_fix_enabled:
                    validated_data['battleTime'] = datetime.now().isoformat()
                    self.validation_stats['auto_fixes_applied'] += 1
                else:
                    is_valid = False
        
        # Validate game mode
        valid_game_modes = ['ladder', 'tournament', 'friendly', 'challenge', 'clanWar']
        if 'gameMode' in validated_data:
            if 'name' in validated_data['gameMode']:
                if validated_data['gameMode']['name'] not in valid_game_modes:
                    if self.auto_fix_enabled:
                        validated_data['gameMode']['name'] = 'ladder'
                        self.validation_stats['auto_fixes_applied'] += 1
                    else:
                        is_valid = False
        
        # Validate team data
        for team_key in ['team', 'opponent']:
            if team_key in validated_data:
                for player in validated_data[team_key]:
                    if 'crowns' in player:
                        crowns = player['crowns']
                        if not isinstance(crowns, int) or crowns < 0 or crowns > 3:
                            if self.auto_fix_enabled:
                                player['crowns'] = max(0, min(3, int(crowns) if isinstance(crowns, (int, float)) else 0))
                                self.validation_stats['auto_fixes_applied'] += 1
                            else:
                                is_valid = False
        
        return validated_data, is_valid
    
    def _validate_prediction_data(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Validate prediction data"""
        is_valid = True
        validated_data = data.copy()
        
        # Validate probability values
        probability_fields = ['win_probability', 'confidence']
        for field in probability_fields:
            if field in validated_data:
                try:
                    value = float(validated_data[field])
                    if np.isnan(value) or np.isinf(value) or value < 0 or value > 1:
                        if self.auto_fix_enabled:
                            validated_data[field] = max(0.0, min(1.0, 0.5 if np.isnan(value) else value))
                            self.validation_stats['auto_fixes_applied'] += 1
                        else:
                            is_valid = False
                except (ValueError, TypeError):
                    if self.auto_fix_enabled:
                        validated_data[field] = 0.5
                        self.validation_stats['auto_fixes_applied'] += 1
                    else:
                        is_valid = False
        
        # Validate timestamp alignment
        if 'timestamp' in validated_data:
            try:
                timestamp = datetime.fromisoformat(validated_data['timestamp'])
                # Check if timestamp is not in the future
                if timestamp > datetime.now():
                    if self.auto_fix_enabled:
                        validated_data['timestamp'] = datetime.now().isoformat()
                        self.validation_stats['auto_fixes_applied'] += 1
                    else:
                        is_valid = False
            except (ValueError, AttributeError):
                if self.auto_fix_enabled:
                    validated_data['timestamp'] = datetime.now().isoformat()
                    self.validation_stats['auto_fixes_applied'] += 1
                else:
                    is_valid = False
        
        return validated_data, is_valid
    
    def _validate_pci_data(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Validate PCI calculation data"""
        is_valid = True
        validated_data = data.copy()
        
        # Validate PCI value
        if 'pci_value' in validated_data:
            try:
                pci_value = float(validated_data['pci_value'])
                if np.isnan(pci_value) or np.isinf(pci_value) or pci_value < 0 or pci_value > 1:
                    if self.auto_fix_enabled:
                        # Recompute PCI if corrupted
                        validated_data['pci_value'] = self._recompute_pci(validated_data)
                        self.validation_stats['auto_fixes_applied'] += 1
                        logger.info("Recomputed corrupted PCI value")
                    else:
                        is_valid = False
            except (ValueError, TypeError):
                if self.auto_fix_enabled:
                    validated_data['pci_value'] = self._recompute_pci(validated_data)
                    self.validation_stats['auto_fixes_applied'] += 1
                else:
                    is_valid = False
        
        return validated_data, is_valid
    
    def _validate_deck_data(self, deck: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        """Validate deck composition"""
        is_valid = True
        validated_deck = []
        
        if not isinstance(deck, list):
            return [], False
        
        # Check deck size
        if len(deck) != 8:
            logger.warning(f"Invalid deck size: {len(deck)}, expected 8")
            if not self.auto_fix_enabled:
                is_valid = False
        
        for card in deck:
            if isinstance(card, dict):
                validated_card = card.copy()
                
                # Validate card level
                if 'level' in validated_card:
                    level = validated_card['level']
                    if not isinstance(level, int) or level < 1 or level > 14:
                        if self.auto_fix_enabled:
                            validated_card['level'] = max(1, min(14, int(level) if isinstance(level, (int, float)) else 1))
                            self.validation_stats['auto_fixes_applied'] += 1
                        else:
                            is_valid = False
                
                # Validate elixir cost
                if 'elixirCost' in validated_card:
                    cost = validated_card['elixirCost']
                    if not isinstance(cost, int) or cost < 0 or cost > 10:
                        if self.auto_fix_enabled:
                            validated_card['elixirCost'] = max(0, min(10, int(cost) if isinstance(cost, (int, float)) else 3))
                            self.validation_stats['auto_fixes_applied'] += 1
                        else:
                            is_valid = False
                
                validated_deck.append(validated_card)
        
        return validated_deck, is_valid
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields"""
        defaults = {
            'tag': '',
            'name': 'Unknown',
            'trophies': 5000,
            'bestTrophies': 5000,
            'wins': 0,
            'losses': 0,
            'expLevel': 13,
            'battleCount': 0,
            'pci_value': 0.5
        }
        return defaults.get(field, None)
    
    def _recompute_pci(self, data: Dict[str, Any]) -> float:
        """Recompute PCI if corrupted"""
        try:
            # Simple PCI recomputation based on available data
            wins = data.get('wins', 0)
            losses = data.get('losses', 0)
            total_battles = wins + losses
            
            if total_battles > 0:
                win_rate = wins / total_battles
                # Simple consistency metric based on win rate stability
                pci = min(1.0, max(0.1, win_rate * 1.2))  # Scale and clamp
            else:
                pci = 0.5  # Default for new players
            
            logger.info(f"Recomputed PCI: {pci}")
            return pci
            
        except Exception as e:
            logger.error(f"Error recomputing PCI: {e}")
            return 0.5
    
    def remove_duplicates(self, battle_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate battle log entries"""
        seen_battles = set()
        unique_battles = []
        
        for battle in battle_logs:
            # Create a unique identifier for the battle
            battle_id = f"{battle.get('battleTime', '')}-{battle.get('gameMode', {}).get('name', '')}"
            
            if battle_id not in seen_battles:
                seen_battles.add(battle_id)
                unique_battles.append(battle)
            else:
                self.validation_stats['data_drops'] += 1
                logger.debug(f"Removed duplicate battle: {battle_id}")
        
        return unique_battles
    
    def fill_missing_values(self, player_data: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fill missing values using player history averages"""
        filled_data = player_data.copy()
        
        if not history:
            return filled_data
        
        # Calculate averages from history
        numeric_fields = ['trophies', 'wins', 'losses']
        for field in numeric_fields:
            if field not in filled_data or filled_data[field] is None:
                values = [h.get(field, 0) for h in history if h.get(field) is not None]
                if values:
                    filled_data[field] = sum(values) / len(values)
                    self.validation_stats['auto_fixes_applied'] += 1
                    logger.info(f"Filled missing {field} with historical average: {filled_data[field]}")
        
        return filled_data
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        if stats['total_validations'] > 0:
            stats['success_rate'] = 1 - (stats['failed_validations'] / stats['total_validations'])
            stats['auto_fix_rate'] = stats['auto_fixes_applied'] / stats['total_validations']
        else:
            stats['success_rate'] = 1.0
            stats['auto_fix_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'auto_fixes_applied': 0,
            'data_drops': 0
        }
