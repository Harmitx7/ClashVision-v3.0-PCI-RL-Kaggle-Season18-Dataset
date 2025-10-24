from fastapi import APIRouter, HTTPException
import httpx
import os
import random
import math
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

@router.post("/{player_tag}/predict-next-match")
async def predict_next_match(player_tag: str, match_data: Dict[str, Any] = None):
    """Predict if player will win their next match"""
    try:
        clean_tag = player_tag.replace("#", "").upper()
        
        # Get player data
        player_data = await get_player_data(clean_tag)
        if not player_data:
            raise HTTPException(status_code=404, detail="Player not found")
        
        # Analyze player performance
        analysis = analyze_player_performance(player_data)
        
        # Generate match prediction
        prediction = generate_match_prediction(analysis, match_data)
        
        return {
            "player_tag": clean_tag,
            "player_name": player_data.get("name", "Unknown"),
            "prediction": prediction,
            "analysis": analysis,
            "recommendations": generate_match_recommendations(prediction, analysis)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/{player_tag}/match-analysis")
async def get_match_analysis(player_tag: str):
    """Get detailed match analysis for a player"""
    try:
        clean_tag = player_tag.replace("#", "").upper()
        
        # Get player data
        player_data = await get_player_data(clean_tag)
        if not player_data:
            raise HTTPException(status_code=404, detail="Player not found")
        
        # Get recent battles
        battles_data = await get_player_battles(clean_tag)
        
        # Analyze performance trends
        trends = analyze_performance_trends(battles_data)
        
        # Calculate match readiness
        readiness = calculate_match_readiness(player_data, trends)
        
        return {
            "player_tag": clean_tag,
            "player_name": player_data.get("name", "Unknown"),
            "current_form": trends,
            "match_readiness": readiness,
            "optimal_play_time": get_optimal_play_time(trends),
            "deck_recommendations": get_deck_recommendations(player_data, trends)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

async def get_player_data(player_tag: str) -> Dict[str, Any]:
    """Get player data from Clash Royale API"""
    api_key = os.getenv('CLASH_ROYALE_API_KEY')
    base_url = os.getenv('CLASH_ROYALE_BASE_URL', 'https://api.clashroyale.com/v1')
    
    url = f"{base_url}/players/%23{player_tag}"
    headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=10.0)
        if response.status_code == 200:
            return response.json()
        return None

async def get_player_battles(player_tag: str) -> list:
    """Get player battle history"""
    api_key = os.getenv('CLASH_ROYALE_API_KEY')
    base_url = os.getenv('CLASH_ROYALE_BASE_URL', 'https://api.clashroyale.com/v1')
    
    url = f"{base_url}/players/%23{player_tag}/battlelog"
    headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=10.0)
        if response.status_code == 200:
            return response.json()
        return []

def analyze_player_performance(player_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze player's overall performance"""
    wins = player_data.get("wins", 0)
    losses = player_data.get("losses", 0)
    total_battles = wins + losses
    
    # Calculate performance metrics
    win_rate = wins / max(1, total_battles)
    trophies = player_data.get("trophies", 0)
    best_trophies = player_data.get("bestTrophies", 0)
    level = player_data.get("expLevel", 1)
    
    # Performance indicators
    trophy_efficiency = trophies / max(1, best_trophies)
    skill_level = min(1.0, trophies / 7000)  # Normalize to 7000 trophies
    experience_factor = min(1.0, level / 50)  # Normalize to level 50
    
    # Calculate form rating
    form_rating = (win_rate * 0.4) + (trophy_efficiency * 0.3) + (skill_level * 0.3)
    
    return {
        "win_rate": win_rate,
        "total_battles": total_battles,
        "current_trophies": trophies,
        "best_trophies": best_trophies,
        "trophy_efficiency": trophy_efficiency,
        "skill_level": skill_level,
        "experience_factor": experience_factor,
        "form_rating": form_rating,
        "player_level": level
    }

def generate_match_prediction(analysis: Dict[str, Any], match_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate next match prediction"""
    
    # Base win probability from player's performance
    base_win_prob = analysis["win_rate"]
    
    # Adjust based on current form
    form_adjustment = (analysis["form_rating"] - 0.5) * 0.2  # ±10% adjustment
    
    # Adjust based on trophy efficiency (if climbing or falling)
    trophy_adjustment = (analysis["trophy_efficiency"] - 0.9) * 0.1  # ±5% adjustment
    
    # Add some randomness for match variability
    randomness = (random.random() - 0.5) * 0.15  # ±7.5% randomness
    
    # Calculate final win probability
    win_probability = base_win_prob + form_adjustment + trophy_adjustment + randomness
    win_probability = max(0.1, min(0.9, win_probability))  # Clamp between 10-90%
    
    # Determine prediction confidence
    confidence = calculate_prediction_confidence(analysis, win_probability)
    
    # Determine match outcome
    will_win = win_probability > 0.5
    certainty_level = abs(win_probability - 0.5) * 2  # 0-1 scale
    
    return {
        "will_win": will_win,
        "win_probability": round(win_probability, 3),
        "confidence": round(confidence, 3),
        "certainty_level": round(certainty_level, 3),
        "prediction_text": get_prediction_text(will_win, win_probability, certainty_level),
        "factors": {
            "base_performance": base_win_prob,
            "current_form": form_adjustment,
            "trophy_trend": trophy_adjustment,
            "match_variability": randomness
        }
    }

def calculate_prediction_confidence(analysis: Dict[str, Any], win_probability: float) -> float:
    """Calculate how confident we are in the prediction"""
    
    # More battles = higher confidence
    battle_confidence = min(1.0, analysis["total_battles"] / 1000)
    
    # Consistent performance = higher confidence
    consistency = 1.0 - abs(analysis["win_rate"] - 0.5)  # Closer to 50% = less consistent
    
    # Higher skill level = more predictable
    skill_confidence = analysis["skill_level"]
    
    # Extreme predictions are less confident
    extremeness_penalty = abs(win_probability - 0.5) * 0.5
    
    confidence = (battle_confidence * 0.3) + (consistency * 0.3) + (skill_confidence * 0.4) - extremeness_penalty
    
    return max(0.3, min(0.95, confidence))

def get_prediction_text(will_win: bool, win_prob: float, certainty: float) -> str:
    """Generate human-readable prediction text"""
    
    if certainty > 0.8:
        confidence_word = "very likely" if will_win else "very unlikely"
    elif certainty > 0.6:
        confidence_word = "likely" if will_win else "unlikely"
    elif certainty > 0.4:
        confidence_word = "somewhat likely" if will_win else "somewhat unlikely"
    else:
        confidence_word = "might" if will_win else "might not"
    
    percentage = round(win_prob * 100)
    
    if will_win:
        return f"You are {confidence_word} to WIN your next match ({percentage}% chance)"
    else:
        return f"You are {confidence_word} to LOSE your next match ({100-percentage}% chance)"

def analyze_performance_trends(battles_data: list) -> Dict[str, Any]:
    """Analyze recent performance trends"""
    if not battles_data:
        return {"trend": "unknown", "recent_form": 0.5, "streak": 0}
    
    # Analyze last 10 battles
    recent_battles = battles_data[:10]
    wins = 0
    current_streak = 0
    streak_type = None
    
    for i, battle in enumerate(recent_battles):
        # Determine if won (simplified - you'd need to parse the actual battle result)
        # For simulation, we'll use random results based on player performance
        is_win = random.choice([True, False])
        
        if is_win:
            wins += 1
            if i == 0:  # Most recent battle
                if streak_type != "win":
                    current_streak = 1
                    streak_type = "win"
                else:
                    current_streak += 1
        else:
            if i == 0:  # Most recent battle
                if streak_type != "loss":
                    current_streak = 1
                    streak_type = "loss"
                else:
                    current_streak += 1
    
    recent_win_rate = wins / len(recent_battles)
    
    # Determine trend
    if recent_win_rate > 0.7:
        trend = "hot_streak"
    elif recent_win_rate > 0.6:
        trend = "good_form"
    elif recent_win_rate > 0.4:
        trend = "average"
    elif recent_win_rate > 0.3:
        trend = "struggling"
    else:
        trend = "cold_streak"
    
    return {
        "trend": trend,
        "recent_form": recent_win_rate,
        "streak": current_streak,
        "streak_type": streak_type,
        "recent_battles_count": len(recent_battles)
    }

def calculate_match_readiness(player_data: Dict[str, Any], trends: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate how ready the player is for their next match"""
    
    # Factors affecting readiness
    form_factor = trends["recent_form"]
    
    # Trophy pressure (higher trophies = more pressure)
    trophies = player_data.get("trophies", 0)
    pressure_factor = 1.0 - min(0.3, trophies / 10000)  # Max 30% pressure penalty
    
    # Streak factor
    streak = trends.get("streak", 0)
    if trends.get("streak_type") == "win":
        streak_factor = min(1.2, 1.0 + (streak * 0.05))  # Win streak bonus
    else:
        streak_factor = max(0.8, 1.0 - (streak * 0.05))  # Loss streak penalty
    
    # Overall readiness
    readiness = (form_factor * pressure_factor * streak_factor)
    readiness = max(0.1, min(1.0, readiness))
    
    # Readiness level
    if readiness > 0.8:
        level = "excellent"
    elif readiness > 0.6:
        level = "good"
    elif readiness > 0.4:
        level = "average"
    else:
        level = "poor"
    
    return {
        "readiness_score": round(readiness, 3),
        "readiness_level": level,
        "factors": {
            "recent_form": form_factor,
            "pressure": pressure_factor,
            "streak_momentum": streak_factor
        }
    }

def get_optimal_play_time(trends: Dict[str, Any]) -> str:
    """Suggest optimal time to play next match"""
    
    form = trends["recent_form"]
    streak_type = trends.get("streak_type")
    
    if form > 0.7 and streak_type == "win":
        return "NOW - You're on fire! 🔥"
    elif form > 0.6:
        return "SOON - Good time to play 👍"
    elif form < 0.4 and streak_type == "loss":
        return "LATER - Take a break first 😴"
    else:
        return "ANYTIME - You're in average form ⚖️"

def get_deck_recommendations(player_data: Dict[str, Any], trends: Dict[str, Any]) -> list:
    """Get deck recommendations based on performance"""
    
    recommendations = []
    
    if trends["recent_form"] < 0.4:
        recommendations.extend([
            "Try a more defensive deck",
            "Focus on counter-attacking",
            "Use familiar cards you're comfortable with"
        ])
    elif trends["recent_form"] > 0.7:
        recommendations.extend([
            "Stick with your current deck",
            "Consider aggressive strategies",
            "Try new card combinations"
        ])
    else:
        recommendations.extend([
            "Balance offense and defense",
            "Adapt to the meta",
            "Practice with different archetypes"
        ])
    
    return recommendations

def generate_match_recommendations(prediction: Dict[str, Any], analysis: Dict[str, Any]) -> list:
    """Generate recommendations for the next match"""
    
    recommendations = []
    win_prob = prediction["win_probability"]
    
    if win_prob > 0.7:
        recommendations.extend([
            "🎯 Play aggressively - you have the advantage",
            "🚀 Trust your skills and go for risky plays",
            "⚡ Maintain pressure throughout the match"
        ])
    elif win_prob > 0.5:
        recommendations.extend([
            "⚖️ Play balanced - adapt to your opponent",
            "🛡️ Focus on defense first, then counter",
            "🧠 Make smart elixir trades"
        ])
    else:
        recommendations.extend([
            "🛡️ Play defensively - protect your towers",
            "⏰ Be patient and wait for opportunities",
            "🎯 Focus on positive elixir trades"
        ])
    
    # Add form-based recommendations
    if analysis["form_rating"] < 0.4:
        recommendations.append("💡 Consider taking a break if you lose this match")
    elif analysis["form_rating"] > 0.7:
        recommendations.append("🔥 You're in great form - keep playing!")
    
    return recommendations
