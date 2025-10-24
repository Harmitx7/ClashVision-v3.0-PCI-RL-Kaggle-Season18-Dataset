from fastapi import WebSocket
from typing import Dict, List, Set, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        # Store active connections by player tag
        self.prediction_connections: Dict[str, Set[WebSocket]] = {}
        self.battle_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, player_tag: str):
        """Connect a WebSocket for predictions"""
        await websocket.accept()
        
        if player_tag not in self.prediction_connections:
            self.prediction_connections[player_tag] = set()
        
        self.prediction_connections[player_tag].add(websocket)
        logger.info(f"WebSocket connected for predictions: {player_tag}")
        
        # Send welcome message
        await self.send_personal_message(websocket, {
            "type": "connection_established",
            "player_tag": player_tag,
            "message": "Connected to prediction stream",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def connect_battle(self, websocket: WebSocket, player_tag: str):
        """Connect a WebSocket for battle monitoring"""
        await websocket.accept()
        
        if player_tag not in self.battle_connections:
            self.battle_connections[player_tag] = set()
        
        self.battle_connections[player_tag].add(websocket)
        logger.info(f"WebSocket connected for battles: {player_tag}")
        
        # Send welcome message
        await self.send_personal_message(websocket, {
            "type": "battle_connection_established",
            "player_tag": player_tag,
            "message": "Connected to battle stream",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket, player_tag: str):
        """Disconnect a prediction WebSocket"""
        if player_tag in self.prediction_connections:
            self.prediction_connections[player_tag].discard(websocket)
            if not self.prediction_connections[player_tag]:
                del self.prediction_connections[player_tag]
        
        logger.info(f"WebSocket disconnected for predictions: {player_tag}")
    
    def disconnect_battle(self, websocket: WebSocket, player_tag: str):
        """Disconnect a battle WebSocket"""
        if player_tag in self.battle_connections:
            self.battle_connections[player_tag].discard(websocket)
            if not self.battle_connections[player_tag]:
                del self.battle_connections[player_tag]
        
        logger.info(f"WebSocket disconnected for battles: {player_tag}")
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to a specific WebSocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def send_prediction(self, player_tag: str, prediction: Dict[str, Any]):
        """Send prediction update to all connected clients for a player"""
        if player_tag not in self.prediction_connections:
            return
        
        message = {
            "type": "prediction_update",
            "player_tag": player_tag,
            "prediction": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all connected clients for this player
        disconnected = set()
        for websocket in self.prediction_connections[player_tag]:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending prediction to {player_tag}: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.prediction_connections[player_tag].discard(websocket)
    
    async def send_battle_update(self, player_tag: str, battle_data: Dict[str, Any]):
        """Send battle update to all connected clients for a player"""
        if player_tag not in self.battle_connections:
            return
        
        message = {
            "type": "battle_update",
            "player_tag": player_tag,
            "battle_data": battle_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all connected clients for this player
        disconnected = set()
        for websocket in self.battle_connections[player_tag]:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending battle update to {player_tag}: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.battle_connections[player_tag].discard(websocket)
    
    async def broadcast_prediction(self, prediction: Dict[str, Any]):
        """Broadcast prediction to all connected clients"""
        message = {
            "type": "global_prediction",
            "prediction": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all prediction connections
        for player_tag, connections in self.prediction_connections.items():
            disconnected = set()
            for websocket in connections:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to {player_tag}: {e}")
                    disconnected.add(websocket)
            
            # Remove disconnected clients
            for websocket in disconnected:
                connections.discard(websocket)
    
    def get_connection_count(self) -> Dict[str, int]:
        """Get current connection counts"""
        return {
            "prediction_connections": sum(len(connections) for connections in self.prediction_connections.values()),
            "battle_connections": sum(len(connections) for connections in self.battle_connections.values()),
            "unique_players": len(set(list(self.prediction_connections.keys()) + list(self.battle_connections.keys())))
        }
