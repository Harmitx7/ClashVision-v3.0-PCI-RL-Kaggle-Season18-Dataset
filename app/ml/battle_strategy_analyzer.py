#!/usr/bin/env python3
"""
Battle Strategy Analyzer - ClashVision v3.0-PCI-RL
==================================================

Provides precise battle strategies and card suggestions based on:
- Recent match data analysis
- Meta trends from hybrid training
- Player-specific performance patterns
- Opponent deck counters
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class BattleStrategyAnalyzer:
    def __init__(self):
        self.card_synergies = self._load_card_synergies()
        self.meta_counters = self._load_meta_counters()
        self.card_effectiveness = defaultdict(lambda: {'wins': 0, 'losses': 0, 'usage': 0})
        self.recent_meta_trends = {}
        
    def analyze_battle_strategy(
        self, 
        player_data: Dict[str, Any],
        opponent_data: Optional[Dict[str, Any]] = None,
        recent_matches: List[Dict[str, Any]] = None,
        pci_value: float = 0.5
    ) -> Dict[str, Any]:
        """
        Provide comprehensive battle strategy analysis
        """
        try:
            strategy = {
                'win_probability': 0.0,
                'confidence': 0.0,
                'strategic_insights': [],
                'card_suggestions': {
                    'cards_to_add': [],
                    'cards_to_remove': [],
                    'deck_improvements': []
                },
                'battle_tactics': [],
                'counter_strategies': [],
                'meta_analysis': {},
                'pci_based_recommendations': []
            }
            
            # Analyze current deck
            current_deck = player_data.get('currentDeck', [])
            if current_deck:
                strategy.update(self._analyze_deck_composition(current_deck, recent_matches))
            
            # Analyze opponent if available
            if opponent_data:
                strategy.update(self._analyze_opponent_matchup(current_deck, opponent_data))
            
            # PCI-based strategic recommendations
            strategy['pci_based_recommendations'] = self._get_pci_recommendations(pci_value, current_deck)
            
            # Recent match pattern analysis
            if recent_matches:
                strategy.update(self._analyze_recent_patterns(recent_matches, current_deck))
            
            # Meta trend analysis
            strategy['meta_analysis'] = self._analyze_current_meta(recent_matches)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_deck_composition(self, deck: List[Dict[str, Any]], recent_matches: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze deck composition and suggest improvements"""
        try:
            analysis = {
                'deck_balance': {},
                'synergy_score': 0.0,
                'weaknesses': [],
                'strengths': []
            }
            
            # Extract card names and costs
            cards = [(card.get('name', ''), card.get('elixirCost', 0)) for card in deck]
            card_names = [name for name, _ in cards]
            elixir_costs = [cost for _, cost in cards]
            
            # Deck balance analysis
            avg_elixir = np.mean(elixir_costs) if elixir_costs else 0
            analysis['deck_balance'] = {
                'average_elixir': round(avg_elixir, 1),
                'elixir_distribution': self._analyze_elixir_distribution(elixir_costs),
                'card_types': self._analyze_card_types(card_names),
                'balance_score': self._calculate_balance_score(cards)
            }
            
            # Synergy analysis
            analysis['synergy_score'] = self._calculate_synergy_score(card_names)
            
            # Identify strengths and weaknesses
            analysis['strengths'] = self._identify_deck_strengths(card_names)
            analysis['weaknesses'] = self._identify_deck_weaknesses(card_names)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Deck composition analysis failed: {e}")
            return {}
    
    def _analyze_opponent_matchup(self, player_deck: List[Dict[str, Any]], opponent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze matchup against opponent"""
        try:
            matchup_analysis = {
                'matchup_advantage': 'neutral',
                'counter_strategies': [],
                'key_interactions': [],
                'tactical_recommendations': []
            }
            
            player_cards = [card.get('name', '') for card in player_deck]
            opponent_trophies = opponent_data.get('trophies', 0)
            player_trophies = sum(card.get('trophies', 0) for card in player_deck if 'trophies' in card)
            
            # Trophy-based analysis
            trophy_diff = player_trophies - opponent_trophies
            if trophy_diff > 200:
                matchup_analysis['matchup_advantage'] = 'favorable'
                matchup_analysis['tactical_recommendations'].append(
                    "Play aggressively - you have a trophy advantage"
                )
            elif trophy_diff < -200:
                matchup_analysis['matchup_advantage'] = 'unfavorable'
                matchup_analysis['tactical_recommendations'].append(
                    "Play defensively and look for counter-attacks"
                )
            
            # Card-specific counter strategies
            matchup_analysis['counter_strategies'] = self._generate_counter_strategies(player_cards)
            
            return matchup_analysis
            
        except Exception as e:
            logger.error(f"Opponent matchup analysis failed: {e}")
            return {}
    
    def _get_pci_recommendations(self, pci_value: float, deck: List[Dict[str, Any]]) -> List[str]:
        """Get PCI-based strategic recommendations"""
        recommendations = []
        
        try:
            if pci_value < 0.3:  # Very unstable player
                recommendations.extend([
                    "🎯 Focus on consistent, reliable cards (Musketeer, Knight, Fireball)",
                    "⚡ Avoid high-skill cards that require precise timing",
                    "🛡️ Play more defensively to build confidence",
                    "📚 Practice with training mode before ranked battles"
                ])
            elif pci_value < 0.5:  # Unstable player
                recommendations.extend([
                    "🎯 Use balanced deck with good synergies",
                    "⚡ Avoid overly aggressive strategies",
                    "🛡️ Focus on positive elixir trades",
                    "📈 Build consistency with familiar card combinations"
                ])
            elif pci_value < 0.7:  # Stable player
                recommendations.extend([
                    "🎯 You can handle moderate complexity decks",
                    "⚡ Mix of offensive and defensive strategies work well",
                    "🛡️ Trust your instincts on card timing",
                    "📈 Consider meta-trending cards for improvement"
                ])
            else:  # Very stable player
                recommendations.extend([
                    "🎯 High-skill decks and complex strategies recommended",
                    "⚡ Aggressive playstyles suit your consistency",
                    "🛡️ You can handle high-risk, high-reward plays",
                    "📈 Experiment with off-meta cards for surprise factor"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"PCI recommendations failed: {e}")
            return ["Unable to generate PCI-based recommendations"]
    
    def _analyze_recent_patterns(self, recent_matches: List[Dict[str, Any]], current_deck: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns from recent matches"""
        try:
            pattern_analysis = {
                'win_rate_by_card': {},
                'losing_patterns': [],
                'winning_patterns': [],
                'performance_trends': {}
            }
            
            if not recent_matches:
                return pattern_analysis
            
            # Analyze card performance
            card_performance = defaultdict(lambda: {'wins': 0, 'losses': 0})
            
            for match in recent_matches[-50:]:  # Last 50 matches
                outcome = match.get('actual_outcome', False)
                battle_data = match.get('battle_data', {})
                
                # Extract cards used (simplified)
                cards_used = self._extract_cards_from_battle(battle_data)
                
                for card in cards_used:
                    if outcome:
                        card_performance[card]['wins'] += 1
                    else:
                        card_performance[card]['losses'] += 1
            
            # Calculate win rates
            for card, stats in card_performance.items():
                total = stats['wins'] + stats['losses']
                if total > 0:
                    win_rate = stats['wins'] / total
                    pattern_analysis['win_rate_by_card'][card] = {
                        'win_rate': round(win_rate, 3),
                        'games_played': total,
                        'confidence': min(1.0, total / 10)  # More games = higher confidence
                    }
            
            # Identify patterns
            pattern_analysis['losing_patterns'] = self._identify_losing_patterns(recent_matches)
            pattern_analysis['winning_patterns'] = self._identify_winning_patterns(recent_matches)
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"Recent pattern analysis failed: {e}")
            return {}
    
    def _analyze_current_meta(self, recent_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze current meta trends"""
        try:
            meta_analysis = {
                'trending_cards': [],
                'meta_shifts': [],
                'recommended_adaptations': []
            }
            
            if not recent_matches:
                return meta_analysis
            
            # Count card usage in recent matches
            card_usage = Counter()
            win_rates = defaultdict(lambda: {'wins': 0, 'total': 0})
            
            for match in recent_matches[-100:]:  # Last 100 matches
                cards = self._extract_cards_from_battle(match.get('battle_data', {}))
                outcome = match.get('actual_outcome', False)
                
                for card in cards:
                    card_usage[card] += 1
                    win_rates[card]['total'] += 1
                    if outcome:
                        win_rates[card]['wins'] += 1
            
            # Identify trending cards
            trending_threshold = len(recent_matches) * 0.1  # 10% usage rate
            for card, usage in card_usage.most_common(10):
                if usage >= trending_threshold:
                    wr = win_rates[card]['wins'] / max(1, win_rates[card]['total'])
                    meta_analysis['trending_cards'].append({
                        'card': card,
                        'usage_rate': round(usage / len(recent_matches), 3),
                        'win_rate': round(wr, 3),
                        'trend_strength': 'high' if usage > trending_threshold * 2 else 'medium'
                    })
            
            # Generate meta recommendations
            meta_analysis['recommended_adaptations'] = self._generate_meta_adaptations(meta_analysis['trending_cards'])
            
            return meta_analysis
            
        except Exception as e:
            logger.error(f"Meta analysis failed: {e}")
            return {}
    
    def generate_card_suggestions(
        self, 
        current_deck: List[Dict[str, Any]], 
        strategy_analysis: Dict[str, Any],
        recent_performance: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate specific card addition/removal suggestions"""
        try:
            suggestions = {
                'cards_to_add': [],
                'cards_to_remove': [],
                'deck_improvements': []
            }
            
            current_cards = [card.get('name', '') for card in current_deck]
            
            # Cards to remove (underperforming)
            win_rates = recent_performance.get('win_rate_by_card', {})
            for card, stats in win_rates.items():
                if card in current_cards and stats['win_rate'] < 0.4 and stats['games_played'] >= 5:
                    suggestions['cards_to_remove'].append({
                        'card': card,
                        'reason': f"Low win rate: {stats['win_rate']:.1%} over {stats['games_played']} games",
                        'priority': 'high' if stats['win_rate'] < 0.3 else 'medium',
                        'alternative_suggestions': self._get_card_alternatives(card)
                    })
            
            # Cards to add (trending/strong)
            meta_cards = strategy_analysis.get('meta_analysis', {}).get('trending_cards', [])
            for card_info in meta_cards:
                card = card_info['card']
                if card not in current_cards and card_info['win_rate'] > 0.6:
                    suggestions['cards_to_add'].append({
                        'card': card,
                        'reason': f"High meta win rate: {card_info['win_rate']:.1%}",
                        'synergy_score': self._calculate_card_synergy(card, current_cards),
                        'priority': 'high' if card_info['win_rate'] > 0.7 else 'medium',
                        'replaces': self._suggest_replacement_for(card, current_cards)
                    })
            
            # Deck improvement suggestions
            suggestions['deck_improvements'] = self._generate_deck_improvements(
                current_deck, strategy_analysis
            )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Card suggestion generation failed: {e}")
            return {'cards_to_add': [], 'cards_to_remove': [], 'deck_improvements': []}
    
    # Helper methods
    def _load_card_synergies(self) -> Dict[str, List[str]]:
        """Load card synergy data"""
        return {
            'Hog Rider': ['Fireball', 'Zap', 'Ice Spirit', 'Musketeer'],
            'Giant': ['Wizard', 'Musketeer', 'Arrows', 'Skeleton Army'],
            'Golem': ['Night Witch', 'Baby Dragon', 'Lightning', 'Tornado'],
            'X-Bow': ['Tesla', 'Ice Golem', 'Archers', 'Log'],
            'Miner': ['Poison', 'Bats', 'Wall Breakers', 'Spear Goblins']
        }
    
    def _load_meta_counters(self) -> Dict[str, List[str]]:
        """Load meta counter data"""
        return {
            'Hog Rider': ['Cannon', 'Tesla', 'Tornado', 'Building'],
            'Giant': ['Inferno Tower', 'Mini P.E.K.K.A', 'Barbarians'],
            'Balloon': ['Musketeer', 'Wizard', 'Inferno Dragon'],
            'Elite Barbarians': ['Valkyrie', 'Bowler', 'Fireball + Zap']
        }
    
    def _analyze_elixir_distribution(self, costs: List[int]) -> Dict[str, Any]:
        """Analyze elixir cost distribution"""
        if not costs:
            return {}
        
        distribution = Counter(costs)
        return {
            'low_cost': sum(1 for cost in costs if cost <= 2),
            'medium_cost': sum(1 for cost in costs if 3 <= cost <= 4),
            'high_cost': sum(1 for cost in costs if cost >= 5),
            'most_common_cost': distribution.most_common(1)[0][0] if distribution else 0
        }
    
    def _analyze_card_types(self, cards: List[str]) -> Dict[str, int]:
        """Analyze card type distribution"""
        # Simplified card type analysis
        spell_cards = ['Fireball', 'Zap', 'Lightning', 'Arrows', 'Poison', 'Freeze']
        building_cards = ['Tesla', 'Cannon', 'Inferno Tower', 'X-Bow']
        
        types = {
            'spells': sum(1 for card in cards if card in spell_cards),
            'buildings': sum(1 for card in cards if card in building_cards),
            'troops': len(cards) - sum(1 for card in cards if card in spell_cards + building_cards)
        }
        
        return types
    
    def _calculate_balance_score(self, cards: List[Tuple[str, int]]) -> float:
        """Calculate deck balance score"""
        if not cards:
            return 0.0
        
        costs = [cost for _, cost in cards]
        avg_cost = np.mean(costs)
        
        # Ideal average is around 3.5-4.0
        cost_score = 1.0 - abs(avg_cost - 3.7) / 3.7
        
        # Check for variety in costs
        unique_costs = len(set(costs))
        variety_score = min(1.0, unique_costs / 6)
        
        return round((cost_score + variety_score) / 2, 2)
    
    def _calculate_synergy_score(self, cards: List[str]) -> float:
        """Calculate deck synergy score"""
        synergy_count = 0
        total_possible = 0
        
        for card in cards:
            if card in self.card_synergies:
                synergies = self.card_synergies[card]
                synergy_count += sum(1 for synergy_card in synergies if synergy_card in cards)
                total_possible += len(synergies)
        
        return round(synergy_count / max(1, total_possible), 2)
    
    def _identify_deck_strengths(self, cards: List[str]) -> List[str]:
        """Identify deck strengths"""
        strengths = []
        
        # Check for strong combinations
        if 'Hog Rider' in cards and 'Fireball' in cards:
            strengths.append("Strong Hog + Fireball combo for tower damage")
        
        if 'Giant' in cards and any(support in cards for support in ['Wizard', 'Musketeer']):
            strengths.append("Solid Giant beatdown with support troops")
        
        # Check spell coverage
        spells = [card for card in cards if card in ['Fireball', 'Zap', 'Lightning', 'Arrows']]
        if len(spells) >= 2:
            strengths.append("Good spell coverage for various situations")
        
        return strengths
    
    def _identify_deck_weaknesses(self, cards: List[str]) -> List[str]:
        """Identify deck weaknesses"""
        weaknesses = []
        
        # Check for air defense
        air_defense = ['Musketeer', 'Wizard', 'Arrows', 'Fireball', 'Tesla']
        if not any(card in cards for card in air_defense):
            weaknesses.append("Weak against air attacks (Balloon, Lava Hound)")
        
        # Check for building
        buildings = ['Tesla', 'Cannon', 'Inferno Tower']
        if not any(card in cards for card in buildings):
            weaknesses.append("No defensive building - vulnerable to Hog Rider, Giant")
        
        # Check average elixir cost
        costs = [4, 3, 2, 5, 4, 3, 2, 4]  # Simplified cost lookup
        if np.mean(costs) > 4.2:
            weaknesses.append("High average elixir cost - may struggle with cycle speed")
        
        return weaknesses
    
    def _generate_counter_strategies(self, cards: List[str]) -> List[str]:
        """Generate counter strategies"""
        strategies = []
        
        if 'Fireball' in cards:
            strategies.append("Use Fireball to counter Wizard, Musketeer, and Barbarians")
        
        if 'Zap' in cards:
            strategies.append("Save Zap for Skeleton Army, Goblin Gang, or Inferno Tower reset")
        
        if any(building in cards for building in ['Tesla', 'Cannon']):
            strategies.append("Place defensive building to counter Hog Rider and Giant pushes")
        
        return strategies
    
    def _extract_cards_from_battle(self, battle_data: Dict[str, Any]) -> List[str]:
        """Extract cards from battle data"""
        # Simplified extraction - in real implementation, parse actual battle data
        return ['Hog Rider', 'Fireball', 'Zap', 'Musketeer']  # Placeholder
    
    def _identify_losing_patterns(self, matches: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in losing matches"""
        patterns = []
        
        losses = [m for m in matches if not m.get('actual_outcome', True)]
        if len(losses) >= 3:
            patterns.append("Recent losing streak - consider deck adjustment")
        
        return patterns
    
    def _identify_winning_patterns(self, matches: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in winning matches"""
        patterns = []
        
        wins = [m for m in matches if m.get('actual_outcome', False)]
        if len(wins) >= 5:
            patterns.append("Strong recent performance - current strategy working well")
        
        return patterns
    
    def _generate_meta_adaptations(self, trending_cards: List[Dict[str, Any]]) -> List[str]:
        """Generate meta adaptation recommendations"""
        adaptations = []
        
        for card_info in trending_cards[:3]:  # Top 3 trending
            card = card_info['card']
            adaptations.append(f"Consider adding {card} - trending with {card_info['win_rate']:.1%} win rate")
        
        return adaptations
    
    def _get_card_alternatives(self, card: str) -> List[str]:
        """Get alternative cards for replacement"""
        alternatives = {
            'Wizard': ['Musketeer', 'Executioner', 'Baby Dragon'],
            'Barbarians': ['Valkyrie', 'Knight', 'Mini P.E.K.K.A'],
            'Arrows': ['Zap', 'Log', 'Snowball']
        }
        
        return alternatives.get(card, ['Consider meta alternatives'])
    
    def _calculate_card_synergy(self, card: str, current_cards: List[str]) -> float:
        """Calculate synergy score for adding a card"""
        if card not in self.card_synergies:
            return 0.5  # Neutral synergy
        
        synergies = self.card_synergies[card]
        synergy_count = sum(1 for synergy_card in synergies if synergy_card in current_cards)
        
        return min(1.0, synergy_count / len(synergies))
    
    def _suggest_replacement_for(self, new_card: str, current_cards: List[str]) -> Optional[str]:
        """Suggest which card to replace when adding a new one"""
        # Simplified logic - replace lowest synergy card
        if len(current_cards) >= 8:
            return current_cards[0]  # Replace first card (placeholder)
        return None
    
    def _generate_deck_improvements(self, deck: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[str]:
        """Generate overall deck improvement suggestions"""
        improvements = []
        
        balance = analysis.get('deck_balance', {})
        avg_elixir = balance.get('average_elixir', 0)
        
        if avg_elixir > 4.2:
            improvements.append("Consider replacing a high-cost card with a cheaper alternative")
        elif avg_elixir < 3.0:
            improvements.append("Add a win condition or heavy damage dealer")
        
        synergy_score = analysis.get('synergy_score', 0)
        if synergy_score < 0.3:
            improvements.append("Improve card synergies - add cards that work well together")
        
        return improvements
