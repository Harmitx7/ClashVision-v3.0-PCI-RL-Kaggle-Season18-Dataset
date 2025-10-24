// WebSocket management for real-time updates
class WebSocketManager {
    constructor() {
        this.connections = new Map();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }
    
    connect(playerTag, type = 'predictions') {
        const wsUrl = `ws://localhost:8000/ws/${type}/${playerTag}`;
        const connectionKey = `${type}_${playerTag}`;
        
        if (this.connections.has(connectionKey)) {
            this.disconnect(connectionKey);
        }
        
        try {
            const ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log(`WebSocket connected: ${connectionKey}`);
                this.reconnectAttempts = 0;
                this.onConnectionOpen(type, playerTag);
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(type, data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            ws.onclose = (event) => {
                console.log(`WebSocket closed: ${connectionKey}`, event.code, event.reason);
                this.connections.delete(connectionKey);
                this.onConnectionClose(type, playerTag);
                
                // Auto-reconnect if not a normal closure
                if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.scheduleReconnect(playerTag, type);
                }
            };
            
            ws.onerror = (error) => {
                console.error(`WebSocket error: ${connectionKey}`, error);
                this.onConnectionError(type, playerTag, error);
            };
            
            this.connections.set(connectionKey, {
                ws,
                playerTag,
                type,
                connected: false
            });
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.onConnectionError(type, playerTag, error);
        }
    }
    
    disconnect(connectionKey) {
        const connection = this.connections.get(connectionKey);
        if (connection && connection.ws) {
            connection.ws.close(1000, 'Manual disconnect');
            this.connections.delete(connectionKey);
        }
    }
    
    disconnectAll() {
        for (const [key, connection] of this.connections) {
            if (connection.ws) {
                connection.ws.close(1000, 'Disconnect all');
            }
        }
        this.connections.clear();
    }
    
    send(connectionKey, data) {
        const connection = this.connections.get(connectionKey);
        if (connection && connection.ws && connection.ws.readyState === WebSocket.OPEN) {
            connection.ws.send(JSON.stringify(data));
            return true;
        }
        return false;
    }
    
    scheduleReconnect(playerTag, type) {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        
        console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
        
        setTimeout(() => {
            if (this.reconnectAttempts <= this.maxReconnectAttempts) {
                console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
                this.connect(playerTag, type);
            }
        }, delay);
    }
    
    onConnectionOpen(type, playerTag) {
        const connectionKey = `${type}_${playerTag}`;
        const connection = this.connections.get(connectionKey);
        if (connection) {
            connection.connected = true;
        }
        
        // Notify the app
        if (window.app) {
            window.app.updateConnectionStatus(true);
        }
        
        // Send initial message if needed
        if (type === 'predictions') {
            this.send(connectionKey, {
                type: 'start_prediction',
                player_tag: playerTag
            });
        }
    }
    
    onConnectionClose(type, playerTag) {
        // Notify the app
        if (window.app) {
            window.app.updateConnectionStatus(false);
        }
    }
    
    onConnectionError(type, playerTag, error) {
        console.error(`WebSocket connection error for ${type}/${playerTag}:`, error);
        
        // Notify the app
        if (window.app) {
            window.app.showNotification('Connection error occurred', 'error');
        }
    }
    
    handleMessage(type, data) {
        switch (type) {
            case 'predictions':
                this.handlePredictionMessage(data);
                break;
            case 'battles':
                this.handleBattleMessage(data);
                break;
            default:
                console.log('Unknown message type:', type, data);
        }
    }
    
    handlePredictionMessage(data) {
        if (window.app) {
            window.app.handlePredictionUpdate(data);
        }
        
        // Handle specific message types
        switch (data.type) {
            case 'connection_established':
                console.log('Prediction connection established');
                break;
            case 'prediction_update':
                console.log('Prediction update received:', data.prediction);
                break;
            case 'error':
                console.error('Prediction error:', data.message);
                if (window.app) {
                    window.app.showNotification(`Prediction error: ${data.message}`, 'error');
                }
                break;
        }
    }
    
    handleBattleMessage(data) {
        // Handle battle-specific messages
        switch (data.type) {
            case 'battle_connection_established':
                console.log('Battle connection established');
                break;
            case 'battle_update':
                console.log('Battle update received:', data.battle_data);
                if (window.app) {
                    window.app.updateBattleStatus(data.battle_data);
                }
                break;
            case 'battle_ended':
                console.log('Battle ended:', data.result);
                if (window.app) {
                    window.app.showNotification(`Battle ended: ${data.result}`, 'info');
                }
                break;
        }
    }
    
    getConnectionStatus() {
        const status = {};
        for (const [key, connection] of this.connections) {
            status[key] = {
                connected: connection.connected,
                readyState: connection.ws ? connection.ws.readyState : -1
            };
        }
        return status;
    }
    
    isConnected(connectionKey) {
        const connection = this.connections.get(connectionKey);
        return connection && connection.connected && connection.ws.readyState === WebSocket.OPEN;
    }
}

// Create global WebSocket manager instance
window.wsManager = new WebSocketManager();

// Clean up connections when page unloads
window.addEventListener('beforeunload', () => {
    if (window.wsManager) {
        window.wsManager.disconnectAll();
    }
});
