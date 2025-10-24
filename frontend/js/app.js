// Main application logic
class ClashRoyaleApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000/api/v1';
        this.currentPlayer = null;
        this.predictionHistory = [];
        this.lastUpdateTime = 0;
        this.updateInterval = 2000; // 2 seconds optimized refresh rate
        this.maxStaleTime = 60000; // 60 seconds max stale data
        this.retryAttempts = 0;
        this.maxRetries = 3;
        this.isRefreshing = false; // Flag to prevent multiple simultaneous refreshes
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.initializeCharts();
        this.checkConnection();
    }
    
    bindEvents() {
        // Search player
        const searchBtn = document.getElementById('searchBtn');
        if (searchBtn) {
            searchBtn.addEventListener('click', () => {
                this.searchPlayer();
            });
        }
        
        
        // Enter key for search
        const playerInput = document.getElementById('playerTagInput');
        if (playerInput) {
            playerInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.searchPlayer();
                }
            });
        }
        
        // Start prediction
        const startPredictionBtn = document.getElementById('startPredictionBtn');
        if (startPredictionBtn) {
            startPredictionBtn.addEventListener('click', () => {
                this.togglePrediction();
            });
        }
        
        // Predict next match
        const predictNextMatchBtn = document.getElementById('predictNextMatchBtn');
        if (predictNextMatchBtn) {
            predictNextMatchBtn.addEventListener('click', () => {
                this.predictNextMatch();
            });
        }
        
        // Test Strategic Analysis button
        const testBtn = document.getElementById('testBtn');
        if (testBtn) {
            testBtn.addEventListener('click', () => {
                console.log('Test Strategic Analysis button clicked!');
                this.testStrategicAnalysis();
            });
        }
        
        // Theme toggle (optional since it's not in the new UI)
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
    }
    
    initializeAutoRefresh() {
        // Only start auto-refresh if essential DOM elements exist
        const essentialElements = ['playerTagInput', 'startPredictionBtn'];
        const allElementsExist = essentialElements.every(id => document.getElementById(id));
        
        if (allElementsExist) {
            // Auto-refresh mechanism for live data
            setInterval(() => {
                this.checkDataFreshness();
            }, this.updateInterval);
            console.log('Auto-refresh initialized');
        } else {
            console.log('Auto-refresh deferred - waiting for DOM elements');
            // Retry after a short delay
            setTimeout(() => this.initializeAutoRefresh(), 100);
        }
    }
    
    checkDataFreshness() {
        const now = Date.now();
        if (this.currentPlayer && (now - this.lastUpdateTime) > this.maxStaleTime && !this.isRefreshing) {
            console.log('Data is stale, refreshing...');
            this.refreshPlayerData();
        }
    }
    
    async refreshPlayerData() {
        if (!this.currentPlayer || this.isRefreshing) return;
        
        // Additional safety checks
        if (!this.currentPlayer.tag) {
            console.error('Current player has no tag, skipping refresh');
            return;
        }
        
        this.isRefreshing = true;
        
        try {
            console.log('Refreshing player data for:', this.currentPlayer.tag);
            const playerTag = this.currentPlayer.tag;
            await this.searchPlayer(playerTag, true); // Silent refresh
            this.retryAttempts = 0; // Reset retry counter on success
        } catch (error) {
            console.error('Failed to refresh player data:', error);
            this.handleRefreshError();
        } finally {
            this.isRefreshing = false;
        }
    }
    
    handleRefreshError() {
        this.retryAttempts++;
        if (this.retryAttempts < this.maxRetries) {
            console.log(`Retry attempt ${this.retryAttempts}/${this.maxRetries}`);
            setTimeout(() => this.refreshPlayerData(), 5000 * this.retryAttempts); // Exponential backoff
        } else {
            console.error('Max retries reached, showing fallback UI');
            this.showFallbackUI();
        }
    }
    
    showFallbackUI() {
        // Show fallback content when data refresh fails
        const fallbackMessage = document.createElement('div');
        fallbackMessage.className = 'bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-4';
        fallbackMessage.innerHTML = `
            <div class="flex">
                <div class="flex-shrink-0">
                    <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                    </svg>
                </div>
                <div class="ml-3">
                    <p class="text-sm">
                        Live data refresh temporarily unavailable. Showing last known data.
                        <button onclick="location.reload()" class="underline ml-2">Refresh Page</button>
                    </p>
                </div>
            </div>
        `;
        
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(fallbackMessage, container.firstChild);
        }
    }
    
    async searchPlayer(playerTag = null, silent = false) {
        // If playerTag is provided, use it; otherwise get from input field
        let tagToSearch = playerTag;
        if (!tagToSearch) {
            const inputElement = document.getElementById('playerTagInput');
            if (!inputElement) {
                console.error('playerTagInput element not found');
                return;
            }
            tagToSearch = inputElement.value.trim();
        }
        
        if (!tagToSearch) {
            if (!silent) {
                this.showNotification('Please enter a player tag', 'error');
            }
            return;
        }
        
        if (!silent) {
            this.showLoading(true);
        }
        
        try {
            // Clean player tag
            const cleanTag = tagToSearch.replace('#', '').toUpperCase();
            
            // Fetch player data
            const response = await fetch(`${this.apiBaseUrl}/players/${cleanTag}?refresh=true`);
            
            if (!response.ok) {
                throw new Error(`Player not found: ${response.status}`);
            }
            
            const playerData = await response.json();
            this.currentPlayer = playerData;
            
            // Update UI only if not in silent mode
            if (!silent) {
                // Update UI
                this.displayPlayerInfo(playerData);
                await this.loadPlayerBattles(cleanTag);
                await this.loadPlayerStats(cleanTag);
                
                // Generate strategic analysis immediately
                console.log('Player data for analysis:', playerData);
                await this.generateStrategicAnalysis(playerData);
                
                // Test: Try to manually update one section to see if it works
                // Temporarily disabled to prevent errors
                // setTimeout(() => {
                //     console.log('Running test strategic analysis in 2 seconds...');
                //     this.testStrategicAnalysis();
                // }, 2000);
                
                // Enable prediction buttons
                const startPredictionBtn = document.getElementById('startPredictionBtn');
                const predictNextMatchBtn = document.getElementById('predictNextMatchBtn');
                
                if (startPredictionBtn) startPredictionBtn.disabled = false;
                if (predictNextMatchBtn) predictNextMatchBtn.disabled = false;
                
                this.showNotification('Player loaded successfully!', 'success');
            } else {
                // In silent mode, only generate strategic analysis
                await this.generateStrategicAnalysis(playerData);
            }
            
        } catch (error) {
            console.error('Error searching player:', error);
            if (!silent) {
                this.showNotification(`Error: ${error.message}`, 'error');
            }
        } finally {
            if (!silent) {
                this.showLoading(false);
            }
        }
    }
    
    
    async generateStrategicAnalysis(playerData) {
        try {
            // Call the prediction API to get strategic analysis
            const response = await fetch(`${this.apiBaseUrl}/predictions/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    player_trophies: playerData.trophies || 5000,
                    player_level: playerData.exp_level || 13,
                    current_deck: playerData.currentDeck || [],
                    opponent_trophies: 5000,
                    arena_id: 15
                })
            });
            
            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.status}`);
            }
            
            const prediction = await response.json();
            console.log('Prediction response:', prediction);
            console.log('Prediction response keys:', Object.keys(prediction));
            console.log('Strategic fields in response:', Object.keys(prediction).filter(k => 
                k.includes('tactic') || k.includes('card') || k.includes('strategy') || k.includes('meta') || k.includes('pci')
            ));
            
            // Update all strategic analysis sections with optimized refresh
            this.updateStrategicAnalysisUI(prediction);
            
            // Update timestamp for freshness tracking
            this.lastUpdateTime = Date.now();
            
        } catch (error) {
            console.error('Error generating strategic analysis:', error);
            this.showNotification(`Strategic analysis failed: ${error.message}`, 'error');
        }
    }
    
    updateStrategicAnalysisUI(prediction) {
        console.log('🎯 updateStrategicAnalysisUI called with:', prediction);
        
        // Update PCI Analysis
        if (prediction.pci_value !== undefined) {
            console.log('📊 Updating PCI:', prediction.pci_value);
            this.updatePCIAnalysis(prediction.pci_value, prediction.pci_interpretation);
        } else {
            console.log('⚠️ No PCI value found in prediction');
        }
        
        // Update Battle Tactics
        if (prediction.battle_tactics) {
            console.log('⚔️ Updating battle tactics:', prediction.battle_tactics.length, 'items');
            this.updateBattleTactics(prediction.battle_tactics);
        } else {
            console.log('⚠️ No battle tactics found in prediction');
        }
        
        // Update Card Suggestions
        if (prediction.detailed_card_suggestions) {
            console.log('🃏 Updating card suggestions');
            this.updateCardSuggestions(prediction.detailed_card_suggestions);
        } else {
            console.log('⚠️ No card suggestions found in prediction');
        }
        
        // Update Counter Strategies
        if (prediction.counter_strategies) {
            console.log('🛡️ Updating counter strategies:', prediction.counter_strategies.length, 'items');
            this.updateCounterStrategies(prediction.counter_strategies);
        } else {
            console.log('⚠️ No counter strategies found in prediction');
        }
        
        // Update Meta Insights
        if (prediction.meta_insights) {
            console.log('📊 Updating meta insights');
            this.updateMetaInsights(prediction.meta_insights);
        } else {
            console.log('⚠️ No meta insights found in prediction');
        }
        
        console.log('✅ updateStrategicAnalysisUI completed');
    }
    
    
    displayPlayerInfo(player) {
        const playerInfoDiv = document.getElementById('playerInfo');
        
        if (!playerInfoDiv) return;
        
        const winRate = (player.wins || 0) / Math.max(1, (player.wins || 0) + (player.losses || 0));
        
        let html = '<div class="space-y-3">';
        
        html += '<div class="flex items-center justify-between">' +
            '<span class="text-gray-500 dark:text-gray-400">Name:</span>' +
            '<span class="font-semibold text-gray-900 dark:text-white">' + (player.name || 'Unknown') + '</span>' +
        '</div>';
        
        html += '<div class="flex items-center justify-between">' +
            '<span class="text-gray-500 dark:text-gray-400">Tag:</span>' +
            '<span class="font-mono text-gray-900 dark:text-white">#' + (player.tag || 'Unknown') + '</span>' +
        '</div>';
        
        html += '<div class="flex items-center justify-between">' +
            '<span class="text-gray-500 dark:text-gray-400">Trophies:</span>' +
            '<span class="font-semibold text-orange-600 dark:text-orange-400">' + ((player.trophies || 0).toLocaleString()) + '</span>' +
        '</div>';
        
        html += '<div class="flex items-center justify-between">' +
            '<span class="text-gray-500 dark:text-gray-400">Best:</span>' +
            '<span class="font-semibold text-gray-900 dark:text-white">' + ((player.best_trophies || player.bestTrophies || 0).toLocaleString()) + '</span>' +
        '</div>';
        
        html += '<div class="flex items-center justify-between">' +
            '<span class="text-gray-500 dark:text-gray-400">Win Rate:</span>' +
            '<span class="font-semibold text-green-600 dark:text-green-400">' + (winRate * 100).toFixed(1) + '%</span>' +
        '</div>';
        
        html += '<div class="flex items-center justify-between">' +
            '<span class="text-gray-500 dark:text-gray-400">Level:</span>' +
            '<span class="font-semibold text-gray-900 dark:text-white">' + (player.exp_level || player.expLevel || 1) + '</span>' +
        '</div>';
        
        html += '<div class="flex items-center justify-between">' +
            '<span class="text-gray-500 dark:text-gray-400">Arena:</span>' +
            '<span class="font-semibold text-gray-900 dark:text-white">' + (player.arena_name || player.arena?.name || 'Unknown') + '</span>' +
        '</div>';
        
        html += '</div>';
        
        playerInfoDiv.innerHTML = html;
        
        // Refresh tilted cards after content update
        if (window.TiltedCards) {
            window.TiltedCards.refresh();
        }
        
        // Refresh animated scroll effects
        if (window.AnimatedScroll) {
            window.AnimatedScroll.refresh();
        }
        
        // Animate the update
        anime({
            targets: playerInfoDiv,
            opacity: [0, 1],
            translateY: [20, 0],
            duration: 500,
            easing: 'easeOutQuart'
        });
    }
    
    async loadPlayerBattles(playerTag) {
        try {
            console.log('Loading battles for player:', playerTag);
            const response = await fetch(`${this.apiBaseUrl}/players/${playerTag}/battles?limit=10`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Battles data received:', data);
            
            // Handle different possible response formats
            let battles = data.battles || data.items || data || [];
            console.log('Battles to display:', battles);
            
            // If no real battles, generate some mock data for demo
            if (!battles || battles.length === 0) {
                battles = this.generateMockBattles();
                console.log('Generated mock battles:', battles);
            }
            
            this.displayRecentBattles(battles);
            
        } catch (error) {
            console.error('Error loading battles:', error);
            const battlesDiv = document.getElementById('recentBattles');
            if (battlesDiv) {
                battlesDiv.innerHTML = '<p class="text-red-500 dark:text-red-400">Error loading battles: ' + error.message + '</p>';
            }
        }
    }
    
    generateMockBattles() {
        const battleTypes = ['1v1', 'Tournament', 'Challenge', 'Ladder', 'Party Mode'];
        const battles = [];
        
        for (let i = 0; i < 5; i++) {
            const playerCrowns = Math.floor(Math.random() * 4); // 0-3 crowns
            const opponentCrowns = Math.floor(Math.random() * 4);
            
            // Ensure not all battles are draws
            const adjustedOpponentCrowns = playerCrowns === opponentCrowns && Math.random() > 0.3 
                ? (Math.random() > 0.5 ? playerCrowns + 1 : Math.max(0, playerCrowns - 1))
                : opponentCrowns;
            
            battles.push({
                player_crowns: playerCrowns,
                opponent_crowns: adjustedOpponentCrowns,
                type: battleTypes[Math.floor(Math.random() * battleTypes.length)],
                battle_time: new Date(Date.now() - (i + 1) * 3600000).toISOString() // Hours ago
            });
        }
        
        return battles;
    }
    
    displayRecentBattles(battles) {
        const battlesDiv = document.getElementById('recentBattles');
        
        if (!battlesDiv) return;
        
        if (!battles || !battles.length) {
            battlesDiv.innerHTML = '<p class="text-gray-500 dark:text-gray-400">No recent battles found</p>';
            return;
        }
        
        console.log('Processing battles:', battles);
        
        let battlesHtml = '';
        const app = this;
        
        battles.slice(0, 5).forEach(function(battle) {
            console.log('Processing battle:', battle);
            
            // Determine battle result based on different possible data structures
            let result = 'unknown';
            let playerCrowns = 0;
            let opponentCrowns = 0;
            
            // Handle different API response formats
            if (battle.team && battle.opponent) {
                // Standard Clash Royale API format
                playerCrowns = battle.team[0] ? (battle.team[0].crowns || 0) : 0;
                opponentCrowns = battle.opponent[0] ? (battle.opponent[0].crowns || 0) : 0;
            } else if (battle.player_crowns !== undefined && battle.opponent_crowns !== undefined) {
                // Custom format
                playerCrowns = battle.player_crowns || 0;
                opponentCrowns = battle.opponent_crowns || 0;
            } else if (battle.crowns !== undefined && battle.opponentCrowns !== undefined) {
                // Alternative format
                playerCrowns = battle.crowns || 0;
                opponentCrowns = battle.opponentCrowns || 0;
            }
            
            // Determine result
            if (playerCrowns > opponentCrowns) {
                result = 'victory';
            } else if (playerCrowns < opponentCrowns) {
                result = 'defeat';
            } else {
                result = 'draw';
            }
            
            let resultColor = 'text-gray-600 dark:text-gray-400';
            let bgColor = 'bg-gray-500';
            let displayResult = result;
            
            if (result === 'victory') {
                resultColor = 'text-green-600 dark:text-green-400';
                bgColor = 'bg-green-500';
                displayResult = 'WIN';
            } else if (result === 'defeat') {
                resultColor = 'text-red-600 dark:text-red-400';
                bgColor = 'bg-red-500';
                displayResult = 'LOSS';
            } else if (result === 'draw') {
                resultColor = 'text-orange-600 dark:text-orange-400';
                bgColor = 'bg-orange-500';
                displayResult = 'DRAW';
            }
            
            // Get battle type and time
            const battleType = battle.type || battle.gameMode || battle.mode || 'Battle';
            const battleTime = battle.battleTime || battle.battle_time || battle.time || new Date().toISOString();
            const formattedTime = app.formatBattleTime ? app.formatBattleTime(battleTime) : 'Recent';
            
            battlesHtml += 
                '<div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 mb-2">' +
                    '<div class="flex items-center space-x-3">' +
                        '<div class="w-3 h-3 rounded-full ' + bgColor + '"></div>' +
                        '<div>' +
                            '<div class="font-semibold ' + resultColor + '">' + displayResult + '</div>' +
                            '<div class="text-sm text-gray-500 dark:text-gray-400">' + battleType + '</div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="text-right">' +
                        '<div class="font-semibold text-gray-900 dark:text-white">' + playerCrowns + ' - ' + opponentCrowns + '</div>' +
                        '<div class="text-sm text-gray-500 dark:text-gray-400">' + formattedTime + '</div>' +
                    '</div>' +
                '</div>';
        });
        
        battlesDiv.innerHTML = battlesHtml;
        
        // Refresh tilted cards after content update
        if (window.TiltedCards) {
            window.TiltedCards.refresh();
        }
        
        // Refresh animated scroll effects
        if (window.AnimatedScroll) {
            window.AnimatedScroll.refresh();
        }
        
        // Animate battles
        anime({
            targets: '#recentBattles > div',
            opacity: [0, 1],
            translateX: [-30, 0],
            delay: anime.stagger(100),
            duration: 600,
            easing: 'easeOutQuart'
        });
    }
    
    determineBattleResult(battle) {
        const playerCrowns = battle.player_crowns || 0;
        const opponentCrowns = battle.opponent_crowns || 0;
        
        if (playerCrowns > opponentCrowns) return 'win';
        if (playerCrowns < opponentCrowns) return 'loss';
        return 'draw';
    }
    
    formatBattleTime(battleTime) {
        try {
            const date = new Date(battleTime);
            
            // Check if date is valid
            if (isNaN(date.getTime())) {
                return 'Unknown';
            }
            
            const now = new Date();
            const diffMs = now - date;
            const diffMins = Math.floor(diffMs / (1000 * 60));
            const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
            const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
            
            // Show relative time for recent battles
            if (diffMins < 1) {
                return 'Just now';
            } else if (diffMins < 60) {
                return diffMins + 'm ago';
            } else if (diffHours < 24) {
                return diffHours + 'h ago';
            } else if (diffDays < 7) {
                return diffDays + 'd ago';
            } else {
                // Show actual date for older battles
                return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            }
        } catch (error) {
            console.error('Error formatting battle time:', error);
            return 'Unknown';
        }
    }
    
    async loadPlayerStats(playerTag) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/players/${playerTag}/stats`);
            const stats = await response.json();
            
            this.displayPlayerStats(stats);
            
        } catch (error) {
            console.error('Error loading player stats:', error);
        }
    }
    
    displayPlayerStats(stats) {
        // Update influencing factors with null checks
        const deckSynergyElement = document.getElementById('deckSynergyScore');
        if (deckSynergyElement) {
            deckSynergyElement.textContent = stats.deck_mastery ? (stats.deck_mastery * 100).toFixed(0) + '%' : '--';
        }
        
        const elixirEfficiencyElement = document.getElementById('elixirEfficiencyScore');
        if (elixirEfficiencyElement) {
            elixirEfficiencyElement.textContent = stats.average_elixir_cost ? stats.average_elixir_cost.toFixed(1) : '--';
        }
        
        const skillScoreElement = document.getElementById('skillScore');
        if (skillScoreElement) {
            skillScoreElement.textContent = stats.skill_rating ? (stats.skill_rating * 100).toFixed(0) + '%' : '--';
        }
        
        const recentFormElement = document.getElementById('recentFormScore');
        if (recentFormElement) {
            recentFormElement.textContent = stats.win_rate ? (stats.win_rate * 100).toFixed(0) + '%' : '--';
        }
        
        const counterScoreElement = document.getElementById('counterScore');
        if (counterScoreElement) {
            counterScoreElement.textContent = stats.consistency_score ? (stats.consistency_score * 100).toFixed(0) + '%' : '--';
        }
        
        // Update player stats section
        const playerStatsDiv = document.getElementById('playerStats');
        if (playerStatsDiv && stats) {
            let html = '<div class="space-y-3">';
            
            html += '<div class="flex items-center justify-between">' +
                '<span class="text-gray-500 dark:text-gray-400">Battles:</span>' +
                '<span class="font-semibold text-gray-900 dark:text-white">' + (stats.total_battles || '--') + '</span>' +
            '</div>';
            
            html += '<div class="flex items-center justify-between">' +
                '<span class="text-gray-500 dark:text-gray-400">Win Rate:</span>' +
                '<span class="font-semibold text-green-600 dark:text-green-400">' + (stats.win_rate ? (stats.win_rate * 100).toFixed(1) + '%' : '--') + '</span>' +
            '</div>';
            
            html += '<div class="flex items-center justify-between">' +
                '<span class="text-gray-500 dark:text-gray-400">Avg Elixir:</span>' +
                '<span class="font-semibold text-purple-600 dark:text-purple-400">' + (stats.average_elixir_cost ? stats.average_elixir_cost.toFixed(1) : '--') + '</span>' +
            '</div>';
            
            html += '<div class="flex items-center justify-between">' +
                '<span class="text-gray-500 dark:text-gray-400">Skill Rating:</span>' +
                '<span class="font-semibold text-blue-600 dark:text-blue-400">' + (stats.skill_rating ? (stats.skill_rating * 100).toFixed(0) + '%' : '--') + '</span>' +
            '</div>';
            
            html += '</div>';
            
            playerStatsDiv.innerHTML = html;
            
            // Refresh tilted cards after content update
            if (window.TiltedCards) {
                window.TiltedCards.refresh();
            }
            
            // Refresh animated scroll effects
            if (window.AnimatedScroll) {
                window.AnimatedScroll.refresh();
            }
        }
    }
    
    async predictNextMatch() {
        console.log('🎯 predictNextMatch called');
        
        if (!this.currentPlayer) {
            this.showNotification('Please search for a player first', 'error');
            return;
        }
        
        // Show the next match prediction section
        const nextMatchSection = document.getElementById('nextMatchSection');
        if (nextMatchSection) {
            nextMatchSection.style.display = 'block';
            console.log('Next match section shown');
        }
        
        // Show loading state
        this.showLoading(true);
        
        try {
            console.log('Generating next match prediction...');
            
            // Use the current player's data to generate a prediction
            const predictionData = {
                player_trophies: this.currentPlayer.trophies || 5000,
                player_level: this.currentPlayer.exp_level || this.currentPlayer.expLevel || 13,
                current_deck: this.currentPlayer.currentDeck || [],
                opponent_trophies: Math.floor(this.currentPlayer.trophies * (0.8 + Math.random() * 0.4)), // Similar level opponent
                arena_id: 15
            };
            
            console.log('Prediction data:', predictionData);
            
            // Call the prediction API
            const response = await fetch(`${this.apiBaseUrl}/predictions/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(predictionData)
            });
            
            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.status}`);
            }
            
            const prediction = await response.json();
            console.log('Next match prediction result:', prediction);
            
            // Update the next match prediction UI
            this.updateNextMatchPrediction(prediction);
            
            this.showNotification('Next match prediction completed!', 'success');
            
        } catch (error) {
            console.error('Error predicting next match:', error);
            this.showNotification(`Prediction failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    async startLivePrediction() {
        try {
            // Initialize WebSocket connection
            this.websocket = new WebSocket(`ws://localhost:8000/ws/predictions/${this.currentPlayer.tag}`);
            
            this.websocket.onopen = () => {
                this.updateConnectionStatus(true);
                this.showNotification('Live prediction started!', 'success');
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handlePredictionUpdate(data);
            };
            
            this.websocket.onclose = () => {
                this.updateConnectionStatus(false);
                if (this.predictionActive) {
                    this.showNotification('Connection lost. Retrying...', 'warning');
                    // Auto-retry connection
                    setTimeout(() => {
                        if (this.predictionActive) {
                            this.startLivePrediction();
                        }
                    }, 3000);
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showNotification('Connection error', 'error');
            };
            
            // Start simulation for demo purposes
            this.startPredictionSimulation();
            
        } catch (error) {
            console.error('Error starting prediction:', error);
            this.showNotification('Failed to start prediction', 'error');
        }
    }
    
    stopLivePrediction() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.updateConnectionStatus(false);
        this.clearPredictionData();
        this.showNotification('Live prediction stopped', 'info');
    }
    
    startPredictionSimulation() {
        // Simulate live predictions for demo
        let simulationStep = 0;
        
        const simulate = () => {
            if (!this.predictionActive) return;
            
            simulationStep++;
            
            // Generate simulated prediction data
            const baseWinProb = 0.5 + (Math.sin(simulationStep * 0.1) * 0.3);
            const winProbability = Math.max(0.1, Math.min(0.9, baseWinProb + (Math.random() - 0.5) * 0.2));
            const confidence = 0.6 + Math.random() * 0.3;
            
            const predictionData = {
                type: 'prediction_update',
                prediction: {
                    win_probability: winProbability,
                    confidence: confidence,
                    pci_value: 0.3 + Math.random() * 0.6,
                    pci_interpretation: {
                        stability_level: Math.random() > 0.5 ? 'Stable' : 'Unstable',
                        description: 'Player demonstrates good consistency with occasional variance',
                        recommendations: [
                            'Focus on consistent, reliable cards',
                            'Avoid high-skill cards requiring precise timing',
                            'Play more defensively to build confidence'
                        ]
                    },
                    battle_tactics: [
                        'Play aggressively - you have deck advantage',
                        'Use Fireball to counter Wizard clusters',
                        'Save Zap for Skeleton Army counters',
                        'Cycle quickly to maintain pressure'
                    ],
                    detailed_card_suggestions: {
                        cards_to_add: [
                            {
                                card: 'Musketeer',
                                reason: 'High meta win rate: 68.5%',
                                priority: 'high',
                                synergy_score: 0.82
                            },
                            {
                                card: 'Tesla',
                                reason: 'Strong defensive synergy with current deck',
                                priority: 'medium',
                                synergy_score: 0.74
                            }
                        ],
                        cards_to_remove: [
                            {
                                card: 'Wizard',
                                reason: 'Low win rate: 34.2% over 12 games',
                                priority: 'medium',
                                alternative_suggestions: ['Musketeer', 'Executioner', 'Baby Dragon']
                            }
                        ],
                        deck_improvements: [
                            'Consider replacing Wizard with Musketeer for better air defense',
                            'Add a defensive building to counter Hog Rider',
                            'Balance elixir cost - current average is too high'
                        ]
                    },
                    counter_strategies: [
                        'Place Tesla reactively against Giant pushes',
                        'Use Fireball + Zap combo for Barbarian counters',
                        'Counter Balloon with Musketeer placement',
                        'Save elixir for defensive plays'
                    ],
                    meta_insights: {
                        trending_cards: [
                            {
                                card: 'Musketeer',
                                usage_rate: 0.58,
                                win_rate: 0.685,
                                trend_strength: 'high'
                            },
                            {
                                card: 'Hog Rider',
                                usage_rate: 0.42,
                                win_rate: 0.612,
                                trend_strength: 'medium'
                            }
                        ],
                        recommended_adaptations: [
                            'Consider adding Musketeer - trending with 68.5% win rate',
                            'Meta shifting towards faster cycle decks',
                            'Air defense becoming more important'
                        ]
                    },
                    influencing_factors: {
                        deck_synergy: 0.6 + Math.random() * 0.3,
                        elixir_efficiency: 0.5 + Math.random() * 0.4,
                        opponent_counter: 0.4 + Math.random() * 0.5,
                        player_skill: 0.7 + Math.random() * 0.2,
                        recent_performance: 0.5 + Math.random() * 0.4
                    },
                    recommendations: this.generateRandomRecommendations()
                }
            };
            
            this.handlePredictionUpdate(predictionData);
            
            // Continue simulation
            setTimeout(simulate, 2000 + Math.random() * 3000);
        };
        
        simulate();
    }
    
    generateRandomRecommendations() {
        const recommendations = [
            "Focus on positive elixir trades",
            "Defend and counter-attack",
            "Build a strong push",
            "Maintain constant pressure",
            "Play defensively",
            "Look for spell value",
            "Cycle to your win condition",
            "Protect your towers"
        ];
        
        const count = Math.floor(Math.random() * 3) + 1;
        const selected = [];
        
        for (let i = 0; i < count; i++) {
            const index = Math.floor(Math.random() * recommendations.length);
            if (!selected.includes(recommendations[index])) {
                selected.push(recommendations[index]);
            }
        }
        
        return selected;
    }
    
    handlePredictionUpdate(data) {
        if (data.type === 'prediction_update' && data.prediction) {
            const prediction = data.prediction;
            
            // Update win probability gauge
            this.updateWinProbability(prediction.win_probability, prediction.confidence);
            
            // Update PCI analysis
            if (prediction.pci_value !== undefined) {
                this.updatePCIAnalysis(prediction.pci_value, prediction.pci_interpretation);
            }
            
            // Update strategic analysis
            if (prediction.strategic_analysis) {
                this.updateStrategicAnalysis(prediction.strategic_analysis);
            }
            
            // Update battle tactics
            if (prediction.battle_tactics) {
                this.updateBattleTactics(prediction.battle_tactics);
            }
            
            // Update card suggestions
            if (prediction.detailed_card_suggestions) {
                this.updateCardSuggestions(prediction.detailed_card_suggestions);
            }
            
            // Update counter strategies
            if (prediction.counter_strategies) {
                this.updateCounterStrategies(prediction.counter_strategies);
            }
            
            // Update meta insights
            if (prediction.meta_insights) {
                this.updateMetaInsights(prediction.meta_insights);
            }
            
            // Update influencing factors
            if (prediction.influencing_factors) {
                this.updateInfluencingFactors(prediction.influencing_factors);
            }
            
            // Update recommendations
            if (prediction.recommendations) {
                this.updateRecommendations(prediction.recommendations);
            }
            
            // Update live chart
            this.updateWinProbability(prediction.win_probability, prediction.confidence);
            
            // Update battle status
            this.updateBattleStatus(prediction);
        }
    }
    
    updateWinProbability(probability, confidence) {
        const percentage = Math.round(probability * 100);
        
        // Update text
        const winProbElement = document.getElementById('winProbability');
        const confidenceElement = document.getElementById('confidence');
        
        if (winProbElement) {
            winProbElement.textContent = percentage + '%';
        }
        
        if (confidenceElement) {
            confidenceElement.textContent = `${Math.round(confidence * 100)}%`;
        }
        
        // Update live chart
        this.updateLiveChart(probability);
        
        // Animate the update
        anime({
            targets: '#winProbability',
            scale: [1.2, 1],
            duration: 300,
            easing: 'easeOutBack'
        });
    }
    
    updateInfluencingFactors(factors) {
        const factorElements = {
            deck_synergy: 'deckSynergyScore',
            elixir_efficiency: 'elixirEfficiencyScore',
            opponent_counter: 'counterScore',
            player_skill: 'skillScore',
            recent_performance: 'recentFormScore'
        };
        
        Object.entries(factors).forEach(([key, value]) => {
            const elementId = factorElements[key];
            if (elementId) {
                const element = document.getElementById(elementId);
                if (element) {
                    element.textContent = Math.round(value * 100) + '%';
                    
                    // Add color based on value
                    element.className = `text-lg font-bold ${
                        value > 0.7 ? 'text-clash-green' :
                        value > 0.4 ? 'text-clash-orange' : 'text-clash-red'
                    }`;
                }
            }
        });
    }
    
    updateRecommendations(recommendations) {
        const recommendationsDiv = document.getElementById('recommendations');
        
        const html = recommendations.map((rec, index) => `
            <div class="flex items-start space-x-3 p-3 bg-white/10 rounded-lg card-animation">
                <div class="w-6 h-6 bg-clash-orange rounded-full flex items-center justify-center text-white text-sm font-bold">
                    ${index + 1}
                </div>
                <p class="text-white flex-1">${rec}</p>
            </div>
        `).join('');
        
        recommendationsDiv.innerHTML = html;
        
        // Animate recommendations
        anime({
            targets: '#recommendations > div',
            opacity: [0, 1],
            translateY: [20, 0],
            delay: anime.stagger(100),
            duration: 500,
            easing: 'easeOutQuart'
        });
    }
    
    updateBattleStatus(prediction) {
        const battleStatusDiv = document.getElementById('battleStatus');
        const battlePhaseElement = document.getElementById('battlePhase');
        
        // Simulate battle state
        const battlePhases = ['Early Game', 'Mid Game', 'Late Game', 'Overtime'];
        const currentPhase = battlePhases[Math.floor(Math.random() * battlePhases.length)];
        
        // Update battle phase in the live prediction section
        if (battlePhaseElement) {
            battlePhaseElement.textContent = currentPhase;
        }
        
        // Update battle status div if it exists
        if (battleStatusDiv) {
            const progressWidth = Math.random() * 100;
            battleStatusDiv.innerHTML = 
                '<div class="space-y-3">' +
                    '<div class="flex items-center justify-between">' +
                        '<span class="text-gray-500">Phase:</span>' +
                        '<span class="font-semibold text-orange-500">' + currentPhase + '</span>' +
                    '</div>' +
                    '<div class="flex items-center justify-between">' +
                        '<span class="text-gray-500">Your Towers:</span>' +
                        '<span class="font-semibold text-green-500">3</span>' +
                    '</div>' +
                    '<div class="flex items-center justify-between">' +
                        '<span class="text-gray-500">Enemy Towers:</span>' +
                        '<span class="font-semibold text-red-500">3</span>' +
                    '</div>' +
                '</div>';
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        
        if (statusElement) {
            if (connected) {
                statusElement.innerHTML = '<div class="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>Connected';
            } else {
                statusElement.innerHTML = '<div class="w-2 h-2 bg-red-500 rounded-full mr-2"></div>Disconnected';
            }
        }
    }
    
    clearPredictionData() {
        const winProbElement = document.getElementById('winProbability');
        const confidenceElement = document.getElementById('confidence');
        
        if (winProbElement) {
            winProbElement.textContent = '--';
        }
        
        if (confidenceElement) {
            confidenceElement.textContent = '--';
        }
        
        // Reset influencing factors
        ['deckSynergyScore', 'elixirEfficiencyScore', 'counterScore', 'skillScore', 'recentFormScore'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = '--';
                element.className = 'text-lg font-bold text-gray-900 dark:text-white';
            }
        });
        
        // Clear recommendations
        const recommendationsElement = document.getElementById('recommendations');
        if (recommendationsElement) {
            recommendationsElement.innerHTML = '<p class="text-gray-500 dark:text-gray-400">Start a prediction to see AI recommendations</p>';
        }
        
        // Clear battle status
        const battleStatusElement = document.getElementById('battleStatus');
        if (battleStatusElement) {
            battleStatusElement.innerHTML = 
                '<div class="flex items-center justify-center h-32">' +
                    '<p class="text-gray-500 dark:text-gray-400">No active battle</p>' +
                '</div>';
        }
    }
    
    initializeCharts() {
        // Initialize live prediction chart
        const chartCanvas = document.getElementById('liveChart');
        if (chartCanvas) {
            const ctx = chartCanvas.getContext('2d');
            
            this.liveChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Win Probability',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: {
                                color: 'rgba(156, 163, 175, 0.2)'
                            },
                            ticks: {
                                color: '#6b7280',
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(156, 163, 175, 0.2)'
                            },
                            ticks: {
                                color: '#6b7280'
                            }
                        }
                    },
                    elements: {
                        point: {
                            radius: 4,
                            hoverRadius: 6
                        }
                    }
                }
            });
        }
    }
    
    updateLiveChart(winProbability) {
        if (this.liveChart && winProbability !== undefined) {
            const now = new Date().toLocaleTimeString();
            const percentage = Math.round(winProbability * 100);
            
            // Add new data point
            this.liveChart.data.labels.push(now);
            this.liveChart.data.datasets[0].data.push(percentage);
            
            // Keep only last 10 data points
            if (this.liveChart.data.labels.length > 10) {
                this.liveChart.data.labels.shift();
                this.liveChart.data.datasets[0].data.shift();
            }
            
            // Update the chart
            this.liveChart.update('none');
        }
    }
    
    checkConnection() {
        // Check if backend is running
        fetch(`${this.apiBaseUrl.replace('/api/v1', '')}/health`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'healthy') {
                    this.showNotification('Connected to server', 'success');
                }
            })
            .catch(() => {
                this.showNotification('Server not available. Please start the backend.', 'warning');
            });
    }
    
    toggleTheme() {
        // Theme toggle functionality
        const body = document.body;
        const themeToggle = document.getElementById('themeToggle');
        
        if (body.classList.contains('dark-theme')) {
            body.classList.remove('dark-theme');
            themeToggle.textContent = '🌙';
        } else {
            body.classList.add('dark-theme');
            themeToggle.textContent = '☀️';
        }
    }
    
    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (show) {
            overlay.classList.remove('hidden');
            overlay.classList.add('flex');
        } else {
            overlay.classList.add('hidden');
            overlay.classList.remove('flex');
        }
    }
    
    async predictNextMatch() {
        if (!this.currentPlayer) {
            this.showNotification('Please search for a player first', 'error');
            return;
        }
        
        this.showLoading(true);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/match/${this.currentPlayer.tag}/predict-next-match`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            if (!response.ok) {
                throw new Error(`Failed to predict match: ${response.status}`);
            }
            
            const prediction = await response.json();
            this.displayMatchPrediction(prediction);
            
            this.showNotification('Next match predicted!', 'success');
            
        } catch (error) {
            console.error('Error predicting match:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    displayMatchPrediction(prediction) {
        // Show the section
        document.getElementById('nextMatchSection').style.display = 'block';
        
        const pred = prediction.prediction;
        const analysis = prediction.analysis;
        
        // Update prediction result
        const resultElement = document.getElementById('matchPredictionResult');
        const textElement = document.getElementById('matchPredictionText');
        
        if (pred.will_win) {
            resultElement.textContent = '🏆 WIN';
            resultElement.className = 'text-4xl font-bold mb-2 text-clash-green';
        } else {
            resultElement.textContent = '💔 LOSE';
            resultElement.className = 'text-4xl font-bold mb-2 text-clash-red';
        }
        
        textElement.textContent = pred.prediction_text;
        
        // Update stats with progress bars
        const winProbPercent = Math.round(pred.win_probability * 100);
        const confidencePercent = Math.round(pred.confidence * 100);
        const formPercent = Math.round(analysis.form_rating * 100);
        
        const matchWinProbElement = document.getElementById('matchWinProb');
        const matchConfidenceElement = document.getElementById('matchConfidence');
        const currentFormElement = document.getElementById('currentForm');
        
        if (matchWinProbElement) matchWinProbElement.textContent = winProbPercent + '%';
        if (matchConfidenceElement) matchConfidenceElement.textContent = confidencePercent + '%';
        if (currentFormElement) currentFormElement.textContent = formPercent + '%';
        
        // Animate progress bars
        setTimeout(function() {
            const winProbBar = document.getElementById('matchWinProbBar');
            const confidenceBar = document.getElementById('matchConfidenceBar');
            const formBar = document.getElementById('currentFormBar');
            
            if (winProbBar) winProbBar.style.width = winProbPercent + '%';
            if (confidenceBar) confidenceBar.style.width = confidencePercent + '%';
            if (formBar) formBar.style.width = formPercent + '%';
        }, 500);
        
        // Update recommendations
        const recommendationsDiv = document.getElementById('matchRecommendations');
        if (recommendationsDiv && prediction.recommendations) {
            let html = '';
            prediction.recommendations.forEach(function(rec, index) {
                const delay = index * 0.1;
                html += '<div class="bg-white/10 rounded-lg p-3 mb-2" style="animation-delay: ' + delay + 's">' +
                    '<div class="flex items-start space-x-3">' +
                        '<div class="w-8 h-8 bg-gradient-to-br from-orange-400 to-red-500 rounded-full flex items-center justify-center flex-shrink-0">' +
                            '<span class="text-white text-sm font-bold">' + (index + 1) + '</span>' +
                        '</div>' +
                        '<p class="text-gray-900 dark:text-white text-sm font-medium leading-relaxed">' + rec + '</p>' +
                    '</div>' +
                '</div>';
            });
            recommendationsDiv.innerHTML = html;
        }
        
        // Animate the section
        anime({
            targets: '#nextMatchSection',
            opacity: [0, 1],
            translateY: [20, 0],
            duration: 800,
            easing: 'easeOutQuart'
        });
        
        // Animate the result
        anime({
            targets: '#matchPredictionResult',
            scale: [0, 1.2, 1],
            duration: 1000,
            easing: 'easeOutElastic(1, .8)'
        });
    }
    
    showNotification(message, type) {
        if (!type) type = 'info';
        
        // Create notification element
        const notification = document.createElement('div');
        let bgClass = 'bg-blue-500';
        if (type === 'success') bgClass = 'bg-green-500';
        else if (type === 'error') bgClass = 'bg-red-500';
        else if (type === 'warning') bgClass = 'bg-orange-500';
        
        notification.className = 'fixed top-4 right-4 p-4 rounded-lg text-white z-50 ' + bgClass;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Animate in
        anime({
            targets: notification,
            translateX: [300, 0],
            opacity: [0, 1],
            duration: 300,
            easing: 'easeOutQuart'
        });
        
        // Remove after delay
        setTimeout(() => {
            anime({
                targets: notification,
                translateX: [0, 300],
                opacity: [1, 0],
                duration: 300,
                easing: 'easeInQuart',
                complete: () => {
                    document.body.removeChild(notification);
                }
            });
        }, 4000);
    }
    
    testStrategicAnalysis() {
        console.log('🧪 Test Strategic Analysis function called!');
        
        // Create realistic test data that matches API response format
        const testData = {
            pci_value: 0.67,
            pci_interpretation: {
                stability_level: 'Stable',
                description: 'Player demonstrates consistent performance with moderate variance',
                recommendations: [
                    'Maintain current playstyle for optimal results',
                    'Focus on elixir efficiency during matches',
                    'Consider defensive positioning when ahead'
                ]
            },
            battle_tactics: [
                '🎯 Play aggressive opening moves to establish control',
                '⚡ Use Fireball reactively against heavy pushes',
                '🛡️ Position defensively when opponent has spell advantage',
                '🔄 Maintain constant pressure without overcommitting'
            ],
            detailed_card_suggestions: {
                cards_to_add: [
                    {
                        card: 'Tesla',
                        reason: 'Strong defensive building effective against most ground pushes',
                        priority: 'high',
                        synergy_score: 0.85
                    },
                    {
                        card: 'Musketeer',
                        reason: 'High meta win rate and versatile ranged unit',
                        priority: 'medium',
                        synergy_score: 0.78
                    }
                ],
                cards_to_remove: [
                    {
                        card: 'Wizard',
                        reason: 'Low win rate against current meta defenses',
                        priority: 'medium',
                        alternative_suggestions: ['Musketeer', 'Electro Wizard', 'Baby Dragon']
                    }
                ],
                deck_improvements: [
                    'Replace Wizard with Tesla for better defensive coverage',
                    'Consider adding another building for split pressure',
                    'Balance elixir curve - current deck is too expensive'
                ]
            },
            counter_strategies: [
                'Place Tesla reactively against Giant pushes',
                'Use Fireball to counter Skeleton Army + Witch combos',
                'Counter Balloon with Musketeer placement',
                'Save elixir for defensive plays'
            ],
            meta_insights: {
                trending_cards: [
                    {
                        card: 'Tesla',
                        usage_rate: 0.68,
                        win_rate: 0.712,
                        trend_strength: 'high'
                    },
                    {
                        card: 'Musketeer',
                        usage_rate: 0.58,
                        win_rate: 0.685,
                        trend_strength: 'high'
                    }
                ],
                recommended_adaptations: [
                    'Tesla showing 71.2% win rate in current meta',
                    'Consider defensive buildings for better counter-play',
                    'Air defense becoming increasingly important'
                ]
            }
        };
        
        console.log('📊 Test data created:', testData);
        
        // Update all strategic analysis sections
        console.log('🔄 Calling updateStrategicAnalysisUI...');
        this.updateStrategicAnalysisUI(testData);
        
        // Show success notification
        this.showNotification('Strategic analysis test completed!', 'success');
        console.log('✅ Test Strategic Analysis completed successfully!');
    }
    
    forceUpdateAllSections() {
        console.log('Force updating all strategic analysis sections...');
        
        // Force update PCI
        const pciValueElement = document.getElementById('pciValue');
        if (pciValueElement) {
            pciValueElement.textContent = '0.75';
            pciValueElement.style.color = 'red';
            pciValueElement.style.fontSize = '24px';
            console.log('Force updated PCI value element');
        }
        
        // Force update battle tactics
        const battleTacticsElement = document.getElementById('battleTactics');
        if (battleTacticsElement) {
            battleTacticsElement.innerHTML = '<div style="background: red; color: white; padding: 10px;">FORCE UPDATED BATTLE TACTICS</div>';
            console.log('Force updated battle tactics element');
        }
        
        // Force update card suggestions
        const cardsToAddElement = document.getElementById('cardsToAdd');
        if (cardsToAddElement) {
            cardsToAddElement.innerHTML = '<div style="background: green; color: white; padding: 10px;">FORCE UPDATED CARDS TO ADD</div>';
            console.log('Force updated cards to add element');
        }
        
        // Force update counter strategies
        const counterStrategiesElement = document.getElementById('counterStrategies');
        if (counterStrategiesElement) {
            counterStrategiesElement.innerHTML = '<div style="background: blue; color: white; padding: 10px;">FORCE UPDATED COUNTER STRATEGIES</div>';
            console.log('Force updated counter strategies element');
        }
        
        // Force update meta insights
        const trendingCardsElement = document.getElementById('trendingCards');
        if (trendingCardsElement) {
            trendingCardsElement.innerHTML = '<div style="background: purple; color: white; padding: 10px;">FORCE UPDATED TRENDING CARDS</div>';
            console.log('Force updated trending cards element');
        }
    }

    // Strategic Analysis Update Functions
    updatePCIAnalysis(pciValue, interpretation) {
        console.log('updatePCIAnalysis called with:', pciValue, interpretation);
        
        // Update PCI circular gauge (simple rotating border)
        const pciCircleFill = document.getElementById('pciCircleFill');
        const pciProgressBar = document.getElementById('pciProgressBar');
        const pciValueElement = document.getElementById('pciValue');
        const stabilityLevel = document.getElementById('stabilityLevel');
        const pciDescription = document.getElementById('pciDescription');
        const pciRecommendations = document.getElementById('pciRecommendations');
        
        console.log('PCI elements found:', {
            pciCircleFill: !!pciCircleFill,
            pciProgressBar: !!pciProgressBar,
            pciValueElement: !!pciValueElement,
            stabilityLevel: !!stabilityLevel,
            pciDescription: !!pciDescription,
            pciRecommendations: !!pciRecommendations
        });

        // Update PCI value display
        if (pciValueElement) {
            pciValueElement.textContent = pciValue.toFixed(2);
            
            // Apply confidence-based color coding
            const confidenceColor = this.getConfidenceColor(pciValue);
            pciValueElement.style.color = confidenceColor;
            
            console.log('PCI value updated to:', pciValue.toFixed(2), 'with color:', confidenceColor);
        }
        
        // Update rotating border animation (simple CSS rotation)
        if (pciCircleFill) {
            // Calculate rotation based on PCI value (0-360 degrees)
            const rotationDegrees = pciValue * 360;
            pciCircleFill.style.transform = `rotate(${rotationDegrees}deg)`;
            
            // Apply confidence-based color coding to the border
            const confidenceColor = this.getConfidenceColor(pciValue);
            pciCircleFill.style.borderTopColor = confidenceColor;
            
            console.log('PCI circle rotated to:', rotationDegrees, 'degrees with color:', confidenceColor);
        }
        
        // Update progress bar (much simpler and more reliable)
        if (pciProgressBar) {
            const percentage = (pciValue * 100) + '%';
            pciProgressBar.style.width = percentage;
            
            // Apply confidence-based color coding
            const confidenceColor = this.getConfidenceColor(pciValue);
            pciProgressBar.style.backgroundColor = confidenceColor;
            
            console.log('PCI progress bar set to:', percentage, 'with color:', confidenceColor);
        }

        // Update interpretation text
        if (interpretation) {
            if (stabilityLevel) {
                stabilityLevel.textContent = interpretation.stability_level || 'Unknown';
            }
            if (pciDescription) {
                pciDescription.textContent = interpretation.description || 'No description available';
            }
        }

        // Update PCI recommendations
        if (pciRecommendations && interpretation && interpretation.recommendations) {
            let html = '';
            interpretation.recommendations.forEach(rec => {
                html += `<div class="flex items-start space-x-2">
                    <div class="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                    <span class="text-sm text-gray-600 dark:text-gray-300">${rec}</span>
                </div>`;
            });
            pciRecommendations.innerHTML = html;
        }
        
        console.log('✅ PCI analysis updated successfully');
    }

    updateStrategicAnalysis(analysis) {
        // This function can be used for overall strategic analysis updates
        console.log('Strategic Analysis:', analysis);
    }

    updateBattleTactics(tactics) {
        const battleTacticsElement = document.getElementById('battleTactics');
        if (battleTacticsElement && tactics && tactics.length > 0) {
            let html = '';
            tactics.forEach(tactic => {
                html += `<div class="flex items-start space-x-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div class="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                    <span class="text-sm text-gray-700 dark:text-gray-300">${tactic}</span>
                </div>`;
            });
            battleTacticsElement.innerHTML = html;
        }
    }

    updateCardSuggestions(suggestions) {
        console.log('updateCardSuggestions called with:', suggestions);
        
        // Check if elements exist
        const cardsToAddElement = document.getElementById('cardsToAdd');
        const cardsToRemoveElement = document.getElementById('cardsToRemove');
        const deckImprovementsElement = document.getElementById('deckImprovements');
        
        console.log('Card suggestion elements found:', {
            cardsToAdd: !!cardsToAddElement,
            cardsToRemove: !!cardsToRemoveElement,
            deckImprovements: !!deckImprovementsElement
        });
        
        // Cards to Add
        if (cardsToAddElement) {
            if (suggestions.cards_to_add && suggestions.cards_to_add.length > 0) {
                let html = '';
                suggestions.cards_to_add.forEach(card => {
                    const priorityColor = card.priority === 'high' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800';
                    html += `<div class="p-3 bg-green-50 rounded-lg border">
                        <div class="flex items-center justify-between mb-2">
                            <span class="font-bold text-green-700">${card.card}</span>
                            <span class="px-2 py-1 text-xs rounded ${priorityColor}">${card.priority}</span>
                        </div>
                        <p class="text-sm text-gray-600">${card.reason}</p>
                        ${card.synergy_score ? `<div class="mt-1 text-xs text-gray-500">Synergy: ${(card.synergy_score * 100).toFixed(0)}%</div>` : ''}
                    </div>`;
                });
                cardsToAddElement.innerHTML = html;
                console.log('Cards to add HTML set:', html);
            } else {
                cardsToAddElement.innerHTML = '<div class="text-gray-500 text-sm p-3">No card additions recommended</div>';
                console.log('No cards to add, showing placeholder');
            }
        } else {
            console.error('cardsToAddElement not found!');
        }

        // Cards to Remove
        if (cardsToRemoveElement && suggestions.cards_to_remove && suggestions.cards_to_remove.length > 0) {
            let html = '';
            suggestions.cards_to_remove.forEach(card => {
                const priorityColor = card.priority === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
                html += `<div class="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                    <div class="flex items-center justify-between mb-2">
                        <span class="font-semibold text-red-700 dark:text-red-300">${card.card}</span>
                        <span class="px-2 py-1 text-xs rounded-full ${priorityColor}">${card.priority}</span>
                    </div>
                    <p class="text-sm text-gray-600 dark:text-gray-400">${card.reason}</p>
                    ${card.alternative_suggestions ? `<div class="mt-2 text-xs text-gray-500">Alternatives: ${card.alternative_suggestions.join(', ')}</div>` : ''}
                </div>`;
            });
            cardsToRemoveElement.innerHTML = html;
        } else if (cardsToRemoveElement) {
            cardsToRemoveElement.innerHTML = '<div class="text-gray-500 text-sm p-3">Removal suggestions will appear here</div>';
        }

        // Deck Improvements
        if (deckImprovementsElement && suggestions.deck_improvements && suggestions.deck_improvements.length > 0) {
            let html = '';
            suggestions.deck_improvements.forEach(improvement => {
                html += `<div class="flex items-start space-x-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                    <svg class="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                    </svg>
                    <span class="text-sm text-gray-700 dark:text-gray-300">${improvement}</span>
                </div>`;
            });
            deckImprovementsElement.innerHTML = html;
        } else if (deckImprovementsElement) {
            deckImprovementsElement.innerHTML = '<div class="text-gray-500 text-sm p-3">Improvement suggestions will appear here</div>';
        }
    }

    updateCounterStrategies(strategies) {
        const counterStrategiesElement = document.getElementById('counterStrategies');
        if (counterStrategiesElement && strategies && strategies.length > 0) {
            let html = '';
            strategies.forEach(strategy => {
                html += `<div class="flex items-start space-x-3 p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                    <div class="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></div>
                    <span class="text-sm text-gray-700 dark:text-gray-300">${strategy}</span>
                </div>`;
            });
            counterStrategiesElement.innerHTML = html;
        }
    }

    updateNextMatchPrediction(prediction) {
        console.log('🎯 updateNextMatchPrediction called with:', prediction);
        
        // Update match prediction result
        const matchPredictionResult = document.getElementById('matchPredictionResult');
        const matchPredictionText = document.getElementById('matchPredictionText');
        const matchWinProb = document.getElementById('matchWinProb');
        const matchConfidence = document.getElementById('matchConfidence');
        const matchRecommendations = document.getElementById('matchRecommendations');
        
        if (prediction.win_probability !== undefined) {
            const winProb = prediction.win_probability;
            const confidence = prediction.confidence || 0.5;
            
            // Determine result based on win probability
            let resultText, resultColor;
            if (winProb > 0.6) {
                resultText = 'VICTORY PREDICTED';
                resultColor = 'text-green-600 dark:text-green-400';
            } else if (winProb > 0.4) {
                resultText = 'CLOSE MATCH';
                resultColor = 'text-orange-600 dark:text-orange-400';
            } else {
                resultText = 'DEFEAT PREDICTED';
                resultColor = 'text-red-600 dark:text-red-400';
            }
            
            if (matchPredictionResult) {
                matchPredictionResult.textContent = resultText;
                matchPredictionResult.className = `text-4xl font-bold mb-2 ${resultColor}`;
            }
            
            if (matchPredictionText) {
                matchPredictionText.textContent = `Predicted outcome for your next battle`;
            }
            
            if (matchWinProb) {
                matchWinProb.textContent = `${(winProb * 100).toFixed(1)}%`;
            }
            
            if (matchConfidence) {
                matchConfidence.textContent = `${(confidence * 100).toFixed(0)}%`;
            }
            
            console.log(`Next match prediction: ${resultText} (${(winProb * 100).toFixed(1)}% win probability)`);
        }
        
        // Update battle strategy recommendations
        if (matchRecommendations && prediction.battle_tactics && prediction.battle_tactics.length > 0) {
            let html = '';
            prediction.battle_tactics.slice(0, 5).forEach(tactic => {
                html += `<div class="flex items-start space-x-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div class="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                    <span class="text-sm text-gray-700 dark:text-gray-300">${tactic}</span>
                </div>`;
            });
            matchRecommendations.innerHTML = html;
            console.log('Battle strategy recommendations updated');
        } else if (matchRecommendations) {
            matchRecommendations.innerHTML = '<p class="text-gray-500 dark:text-gray-400">No specific recommendations available</p>';
        }
        
        // Scroll to the next match section
        const nextMatchSection = document.getElementById('nextMatchSection');
        if (nextMatchSection) {
            nextMatchSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        
        console.log('✅ Next match prediction UI updated');
    }
    
    // Confidence Color Coding Methods
    getConfidenceColor(value) {
        if (value < 0.3) return '#ff7b00'; // Low confidence - orange
        if (value < 0.7) return '#ffd700'; // Medium confidence - gold
        return '#00c853'; // High confidence - green
    }
    
    getConfidenceClass(value) {
        if (value < 0.3) return 'text-orange-500';
        if (value < 0.7) return 'text-yellow-500';
        return 'text-green-500';
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ClashRoyaleApp();
});
