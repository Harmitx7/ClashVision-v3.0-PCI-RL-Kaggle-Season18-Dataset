from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from app.core.database import get_db
from app.services.clash_royale_service import ClashRoyaleService
from app.models.clan import Clan, ClanWar
from app.schemas.clan import ClanResponse, ClanWarResponse

router = APIRouter()
logger = logging.getLogger(__name__)
clash_service = ClashRoyaleService()

@router.get("/{clan_tag}", response_model=ClanResponse)
async def get_clan(
    clan_tag: str,
    db: Session = Depends(get_db),
    refresh: bool = Query(False, description="Force refresh from API")
):
    """Get clan information by tag"""
    try:
        clean_tag = clan_tag.replace("#", "").upper()
        
        if refresh:
            # Fetch fresh data from API
            clan_data = await clash_service.get_clan(clean_tag)
            if not clan_data:
                raise HTTPException(status_code=404, detail="Clan not found")
            
            # Update database
            await clash_service.update_clan_in_db(db, clan_data)
        
        # Get from database
        clan = db.query(Clan).filter(Clan.tag == clean_tag).first()
        if not clan:
            # Try to fetch from API if not in database
            clan_data = await clash_service.get_clan(clean_tag)
            if not clan_data:
                raise HTTPException(status_code=404, detail="Clan not found")
            
            clan = await clash_service.update_clan_in_db(db, clan_data)
        
        return ClanResponse.from_orm(clan)
        
    except Exception as e:
        logger.error(f"Error getting clan {clan_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{clan_tag}/members")
async def get_clan_members(
    clan_tag: str,
    db: Session = Depends(get_db)
):
    """Get clan members"""
    try:
        clean_tag = clan_tag.replace("#", "").upper()
        
        # Get clan data from API (includes members)
        clan_data = await clash_service.get_clan(clean_tag)
        if not clan_data:
            raise HTTPException(status_code=404, detail="Clan not found")
        
        members = clan_data.get("memberList", [])
        
        return {
            "clan_tag": clean_tag,
            "clan_name": clan_data.get("name", ""),
            "member_count": len(members),
            "members": members
        }
        
    except Exception as e:
        logger.error(f"Error getting clan members {clan_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{clan_tag}/war")
async def get_clan_war(
    clan_tag: str,
    db: Session = Depends(get_db)
):
    """Get current clan war information"""
    try:
        clean_tag = clan_tag.replace("#", "").upper()
        
        # Get current war from API
        war_data = await clash_service.get_clan_current_war(clean_tag)
        if not war_data:
            return {
                "clan_tag": clean_tag,
                "in_war": False,
                "message": "Clan is not currently in war"
            }
        
        return {
            "clan_tag": clean_tag,
            "in_war": True,
            "war_data": war_data
        }
        
    except Exception as e:
        logger.error(f"Error getting clan war {clan_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{clan_tag}/river-race")
async def get_clan_river_race(
    clan_tag: str,
    db: Session = Depends(get_db)
):
    """Get current river race information"""
    try:
        clean_tag = clan_tag.replace("#", "").upper()
        
        # Get current river race from API
        river_race_data = await clash_service.get_clan_current_river_race(clean_tag)
        if not river_race_data:
            return {
                "clan_tag": clean_tag,
                "in_river_race": False,
                "message": "Clan is not currently in river race"
            }
        
        return {
            "clan_tag": clean_tag,
            "in_river_race": True,
            "river_race_data": river_race_data
        }
        
    except Exception as e:
        logger.error(f"Error getting clan river race {clan_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{clan_tag}/performance")
async def get_clan_performance(
    clan_tag: str,
    db: Session = Depends(get_db)
):
    """Get clan performance analytics"""
    try:
        clean_tag = clan_tag.replace("#", "").upper()
        
        # Get clan data
        clan_data = await clash_service.get_clan(clean_tag)
        if not clan_data:
            raise HTTPException(status_code=404, detail="Clan not found")
        
        members = clan_data.get("memberList", [])
        
        # Calculate performance metrics
        performance = {
            "clan_tag": clean_tag,
            "clan_name": clan_data.get("name", ""),
            "member_count": len(members),
            "total_trophies": sum(member.get("trophies", 0) for member in members),
            "average_trophies": sum(member.get("trophies", 0) for member in members) / max(1, len(members)),
            "total_donations": sum(member.get("donations", 0) for member in members),
            "average_donations": sum(member.get("donations", 0) for member in members) / max(1, len(members)),
            "clan_war_trophies": clan_data.get("clanWarTrophies", 0),
            "required_trophies": clan_data.get("requiredTrophies", 0),
            "member_distribution": _calculate_member_distribution(members),
            "activity_score": _calculate_activity_score(members)
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"Error getting clan performance {clan_tag}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def _calculate_member_distribution(members: List[dict]) -> dict:
    """Calculate member distribution by trophy ranges"""
    try:
        distribution = {
            "legendary": 0,    # 5000+
            "master": 0,       # 4000-4999
            "champion": 0,     # 3000-3999
            "challenger": 0,   # 2000-2999
            "arena": 0         # <2000
        }
        
        for member in members:
            trophies = member.get("trophies", 0)
            
            if trophies >= 5000:
                distribution["legendary"] += 1
            elif trophies >= 4000:
                distribution["master"] += 1
            elif trophies >= 3000:
                distribution["champion"] += 1
            elif trophies >= 2000:
                distribution["challenger"] += 1
            else:
                distribution["arena"] += 1
        
        return distribution
        
    except Exception as e:
        logger.error(f"Error calculating member distribution: {e}")
        return {}

def _calculate_activity_score(members: List[dict]) -> float:
    """Calculate clan activity score based on donations and participation"""
    try:
        if not members:
            return 0.0
        
        total_donations = sum(member.get("donations", 0) for member in members)
        total_received = sum(member.get("donationsReceived", 0) for member in members)
        
        # Activity score based on donation activity
        donation_activity = min(1.0, (total_donations + total_received) / (len(members) * 100))
        
        # Factor in member participation (members with recent activity)
        active_members = sum(1 for member in members if member.get("donations", 0) > 0)
        participation_rate = active_members / len(members)
        
        # Combined activity score
        activity_score = (donation_activity * 0.6) + (participation_rate * 0.4)
        
        return round(activity_score, 2)
        
    except Exception as e:
        logger.error(f"Error calculating activity score: {e}")
        return 0.0
