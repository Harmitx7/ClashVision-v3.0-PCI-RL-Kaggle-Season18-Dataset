import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class PlayerConsistencyIndex:
    """
    Player Consistency Index (PCI) calculator for ClashVision v3.0
    
    PCI quantifies player stability, form, and tilt probability to improve
    accuracy and confidence calibration in win/loss predictions.
    
    Range: [0.0, 1.0]
    - 0.0: Inconsistent / tilted / unstable
    - 0.5: Moderately stable
    - 1.0: Highly consistent performer
    """
    
    def __init__(self):
        self.pci_cache = {}  # Cache for computed PCI values
        self.update_frequency_hours = 1  # Update PCI every hour
        
    def calculate_pci(
        self,
        player_tag: str,
        battle_history: List[Dict[str, Any]],
        player_stats: Dict[str, Any]
    ) -> float:
        """
        Calculate Player Consistency Index based on recent performance
        
        Formula: PCI = clamp(
            0.3 * recent_win_rate_20 + 
            0.2 * tanh(streak_length/10) + 
            0.2 * (1 - std_winrate_rolling_50) + 
            0.1 * (1 - sigmoid((avg_match_interval_hours-24)/24)) + 
            0.1 * (1 - time_of_day_variance) + 
            0.1 * (1 - opponent_strength_variance), 
            0, 1
        )
        """
        try:
            # Check cache first
            cache_key = f"{player_tag}_{datetime.now().hour}"
            if cache_key in self.pci_cache:
                return self.pci_cache[cache_key]
            
            if not battle_history:
                return 0.5  # Default moderate stability
            
            # Extract input features
            recent_win_rate_20 = self._calculate_recent_win_rate(battle_history, 20)
            streak_length = self._calculate_streak_length(battle_history)
            std_winrate_rolling_50 = self._calculate_winrate_std(battle_history, 50)
            avg_match_interval_hours = self._calculate_avg_match_interval(battle_history)
            time_of_day_variance = self._calculate_time_of_day_variance(battle_history)
            opponent_strength_variance = self._calculate_opponent_strength_variance(battle_history)
            
            # Apply PCI formula
            pci = (
                0.3 * recent_win_rate_20 +
                0.2 * math.tanh(streak_length / 10.0) +
                0.2 * (1.0 - std_winrate_rolling_50) +
                0.1 * (1.0 - self._sigmoid((avg_match_interval_hours - 24) / 24.0)) +
                0.1 * (1.0 - time_of_day_variance) +
                0.1 * (1.0 - opponent_strength_variance)
            )
            
            # Clamp to [0, 1] range
            pci = max(0.0, min(1.0, pci))
            
            # Cache result
            self.pci_cache[cache_key] = pci
            
            logger.debug(f"PCI calculated for {player_tag}: {pci:.3f}")
            return pci
            
        except Exception as e:
            logger.error(f"Error calculating PCI for {player_tag}: {e}")
            return 0.5  # Default moderate stability
    
    def _calculate_recent_win_rate(self, battle_history: List[Dict[str, Any]], n_battles: int) -> float:
        """Calculate win rate over last n battles"""
        try:
            recent_battles = battle_history[:n_battles]
            if not recent_battles:
                return 0.5
            
            wins = sum(1 for battle in recent_battles if self._is_win(battle))
            return wins / len(recent_battles)
            
        except Exception as e:
            logger.error(f"Error calculating recent win rate: {e}")
            return 0.5
    
    def _calculate_streak_length(self, battle_history: List[Dict[str, Any]]) -> int:
        """Calculate current win/loss streak (positive = wins, negative = losses)"""
        try:
            if not battle_history:
                return 0
            
            streak = 0
            last_result = None
            
            for battle in battle_history:
                current_result = self._is_win(battle)
                
                if last_result is None:
                    last_result = current_result
                    streak = 1 if current_result else -1
                elif current_result == last_result:
                    if current_result:
                        streak += 1
                    else:
                        streak -= 1
                else:
                    break
            
            return streak
            
        except Exception as e:
            logger.error(f"Error calculating streak length: {e}")
            return 0
    
    def _calculate_winrate_std(self, battle_history: List[Dict[str, Any]], window_size: int) -> float:
        """Calculate standard deviation of win rate in rolling window"""
        try:
            if len(battle_history) < window_size:
                return 0.0
            
            win_rates = []
            for i in range(len(battle_history) - window_size + 1):
                window = battle_history[i:i + window_size]
                wins = sum(1 for battle in window if self._is_win(battle))
                win_rate = wins / window_size
                win_rates.append(win_rate)
            
            return np.std(win_rates) if win_rates else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating winrate std: {e}")
            return 0.0
    
    def _calculate_avg_match_interval(self, battle_history: List[Dict[str, Any]]) -> float:
        """Calculate average time between matches in hours"""
        try:
            if len(battle_history) < 2:
                return 24.0  # Default 24 hours
            
            intervals = []
            for i in range(len(battle_history) - 1):
                time1 = self._parse_battle_time(battle_history[i])
                time2 = self._parse_battle_time(battle_history[i + 1])
                
                if time1 and time2:
                    interval_hours = abs((time1 - time2).total_seconds()) / 3600.0
                    intervals.append(interval_hours)
            
            return np.mean(intervals) if intervals else 24.0
            
        except Exception as e:
            logger.error(f"Error calculating avg match interval: {e}")
            return 24.0
    
    def _calculate_time_of_day_variance(self, battle_history: List[Dict[str, Any]]) -> float:
        """Calculate performance variance by time of day"""
        try:
            if len(battle_history) < 10:
                return 0.0
            
            # Group battles by hour of day
            hourly_performance = {}
            for battle in battle_history:
                battle_time = self._parse_battle_time(battle)
                if battle_time:
                    hour = battle_time.hour
                    if hour not in hourly_performance:
                        hourly_performance[hour] = []
                    hourly_performance[hour].append(1.0 if self._is_win(battle) else 0.0)
            
            # Calculate win rate for each hour
            hourly_win_rates = []
            for hour, results in hourly_performance.items():
                if len(results) >= 3:  # Need at least 3 battles for meaningful stats
                    win_rate = np.mean(results)
                    hourly_win_rates.append(win_rate)
            
            return np.std(hourly_win_rates) if len(hourly_win_rates) > 1 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating time of day variance: {e}")
            return 0.0
    
    def _calculate_opponent_strength_variance(self, battle_history: List[Dict[str, Any]]) -> float:
        """Calculate variance in opponent trophy levels"""
        try:
            opponent_trophies = []
            for battle in battle_history:
                opponent = battle.get("opponent", {})
                trophies = opponent.get("startingTrophies", opponent.get("trophies", 0))
                if trophies > 0:
                    opponent_trophies.append(trophies)
            
            if len(opponent_trophies) < 5:
                return 0.0
            
            # Normalize by mean to get coefficient of variation
            mean_trophies = np.mean(opponent_trophies)
            std_trophies = np.std(opponent_trophies)
            
            return (std_trophies / mean_trophies) if mean_trophies > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating opponent strength variance: {e}")
            return 0.0
    
    def _is_win(self, battle: Dict[str, Any]) -> bool:
        """Determine if battle was a win"""
        team = battle.get("team", [])
        if not team:
            return False
        
        player_crowns = team[0].get("crowns", 0)
        opponent = battle.get("opponent", [])
        opponent_crowns = opponent[0].get("crowns", 0) if opponent else 0
        
        return player_crowns > opponent_crowns
    
    def _parse_battle_time(self, battle: Dict[str, Any]) -> Optional[datetime]:
        """Parse battle timestamp"""
        try:
            battle_time_str = battle.get("battleTime", "")
            if battle_time_str:
                # Parse ISO format: 20231023T123045.000Z
                return datetime.strptime(battle_time_str, "%Y%m%dT%H%M%S.%fZ")
            return None
        except Exception:
            return None
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def get_pci_interpretation(self, pci: float) -> Dict[str, Any]:
        """Get human-readable interpretation of PCI value"""
        if pci < 0.25:
            stability = "Very Unstable"
            description = "Player is likely tilted or experiencing significant performance issues"
            confidence_modifier = 0.4
        elif pci < 0.4:
            stability = "Unstable"
            description = "Player showing inconsistent performance patterns"
            confidence_modifier = 0.6
        elif pci < 0.6:
            stability = "Moderate"
            description = "Player has average consistency levels"
            confidence_modifier = 0.8
        elif pci < 0.8:
            stability = "Stable"
            description = "Player demonstrates good consistency"
            confidence_modifier = 0.9
        else:
            stability = "Very Stable"
            description = "Player shows excellent consistency and form"
            confidence_modifier = 1.0
        
        return {
            "pci_value": pci,
            "stability_level": stability,
            "description": description,
            "confidence_modifier": confidence_modifier,
            "should_route_to_specialized_model": pci < 0.25 or pci > 0.85
        }
    
    def calculate_confidence_modulation(self, base_confidence: float, pci: float) -> float:
        """Modulate prediction confidence based on PCI"""
        # Reduce confidence for unstable players
        pci_factor = 1.0 - abs(0.5 - pci) * 0.6
        return base_confidence * pci_factor
    
    def should_use_specialized_model(self, pci: float) -> Tuple[bool, str]:
        """Determine if specialized model should be used based on PCI"""
        if pci < 0.25:
            return True, "tilt_model"
        elif pci > 0.85:
            return True, "elite_model"
        else:
            return False, "standard_model"
    
    def update_pci_cache(self, player_tag: str, pci_value: float):
        """Update PCI cache with new value"""
        cache_key = f"{player_tag}_{datetime.now().hour}"
        self.pci_cache[cache_key] = pci_value
    
    def clear_old_cache_entries(self):
        """Clear old cache entries to prevent memory bloat"""
        current_hour = datetime.now().hour
        keys_to_remove = []
        
        for key in self.pci_cache.keys():
            try:
                _, cached_hour = key.rsplit("_", 1)
                if abs(int(cached_hour) - current_hour) > 2:  # Keep only last 2 hours
                    keys_to_remove.append(key)
            except (ValueError, IndexError):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.pci_cache[key]
