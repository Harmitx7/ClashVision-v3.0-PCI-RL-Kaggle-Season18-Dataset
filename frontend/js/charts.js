// Chart management and visualization
let winProbabilityChart = null;
let liveChart = null;
let liveChartData = [];

function initializeCharts() {
    initializeWinProbabilityGauge();
    initializeLiveChart();
}

function initializeWinProbabilityGauge() {
    const ctx = document.getElementById('winProbabilityChart').getContext('2d');
    
    winProbabilityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: [
                    'rgba(126, 211, 33, 0.8)',  // Green for win probability
                    'rgba(255, 255, 255, 0.1)'  // Light background
                ],
                borderWidth: 0,
                cutout: '75%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            },
            animation: {
                animateRotate: true,
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
    
    // Store reference globally
    window.winProbabilityChart = winProbabilityChart;
}

function initializeLiveChart() {
    const ctx = document.getElementById('liveChart').getContext('2d');
    
    // Initialize with empty data
    liveChartData = Array(20).fill(null);
    
    liveChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(20).fill('').map((_, i) => `${i * 5}s`),
            datasets: [{
                label: 'Win Probability',
                data: liveChartData,
                borderColor: 'rgba(102, 126, 234, 1)',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `Win Probability: ${(context.parsed.y * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: {
                            size: 12
                        }
                    }
                },
                y: {
                    display: true,
                    min: 0,
                    max: 1,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: {
                            size: 12
                        },
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            animation: {
                duration: 750,
                easing: 'easeOutQuart'
            }
        }
    });
    
    // Store reference globally
    window.liveChart = liveChart;
}

function updateLiveChart(winProbability) {
    if (!liveChart) return;
    
    // Add new data point
    liveChartData.push(winProbability);
    
    // Remove oldest data point if we have too many
    if (liveChartData.length > 20) {
        liveChartData.shift();
    }
    
    // Update chart data
    liveChart.data.datasets[0].data = [...liveChartData];
    
    // Update labels
    const currentTime = Date.now();
    liveChart.data.labels = liveChartData.map((_, i) => {
        const timeAgo = (liveChartData.length - 1 - i) * 5;
        return timeAgo === 0 ? 'Now' : `-${timeAgo}s`;
    });
    
    // Animate the update
    liveChart.update('active');
    
    // Add visual effects for significant changes
    if (liveChartData.length >= 2) {
        const previousValue = liveChartData[liveChartData.length - 2];
        const currentValue = winProbability;
        const change = Math.abs(currentValue - previousValue);
        
        if (change > 0.1) { // Significant change (>10%)
            addChartEffect(currentValue > previousValue ? 'positive' : 'negative');
        }
    }
}

function addChartEffect(type) {
    const chartContainer = document.getElementById('liveChart').parentElement;
    
    // Add glow effect
    chartContainer.classList.add('glow');
    
    // Create floating indicator
    const indicator = document.createElement('div');
    indicator.className = `absolute top-2 right-2 px-2 py-1 rounded text-sm font-bold ${
        type === 'positive' ? 'bg-clash-green text-white' : 'bg-clash-red text-white'
    }`;
    indicator.textContent = type === 'positive' ? '↗ Rising' : '↘ Falling';
    
    chartContainer.style.position = 'relative';
    chartContainer.appendChild(indicator);
    
    // Animate indicator
    anime({
        targets: indicator,
        opacity: [0, 1, 1, 0],
        translateY: [10, 0, 0, -10],
        duration: 2000,
        easing: 'easeOutQuart',
        complete: () => {
            if (chartContainer.contains(indicator)) {
                chartContainer.removeChild(indicator);
            }
        }
    });
    
    // Remove glow after animation
    setTimeout(() => {
        chartContainer.classList.remove('glow');
    }, 2000);
}

function createInfluencingFactorsChart(factors) {
    // Create a radar chart for influencing factors
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 300;
    
    const ctx = canvas.getContext('2d');
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: [
                'Deck Synergy',
                'Elixir Efficiency', 
                'Counter Potential',
                'Player Skill',
                'Recent Form'
            ],
            datasets: [{
                label: 'Impact',
                data: [
                    factors.deck_synergy || 0,
                    factors.elixir_efficiency || 0,
                    factors.opponent_counter || 0,
                    factors.player_skill || 0,
                    factors.recent_performance || 0
                ],
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.2)'
                    },
                    angleLines: {
                        color: 'rgba(255, 255, 255, 0.2)'
                    },
                    pointLabels: {
                        color: 'rgba(255, 255, 255, 0.8)',
                        font: {
                            size: 12
                        }
                    },
                    ticks: {
                        display: false
                    }
                }
            }
        }
    });
    
    return canvas;
}

function animateChartUpdate(chart, newData) {
    if (!chart) return;
    
    // Store original data
    const originalData = [...chart.data.datasets[0].data];
    
    // Animate to new data
    anime({
        targets: chart.data.datasets[0].data,
        ...newData.reduce((acc, value, index) => {
            acc[index] = value;
            return acc;
        }, {}),
        duration: 1000,
        easing: 'easeOutQuart',
        update: () => {
            chart.update('none');
        }
    });
}

function createBattleTimelineChart(battleData) {
    // Create a timeline chart showing battle progression
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: battleData.timeline || [],
            datasets: [
                {
                    label: 'Your Towers',
                    data: battleData.playerTowers || [],
                    borderColor: 'rgba(126, 211, 33, 1)',
                    backgroundColor: 'rgba(126, 211, 33, 0.1)',
                    borderWidth: 3,
                    fill: false
                },
                {
                    label: 'Enemy Towers',
                    data: battleData.opponentTowers || [],
                    borderColor: 'rgba(208, 2, 27, 1)',
                    backgroundColor: 'rgba(208, 2, 27, 0.1)',
                    borderWidth: 3,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: 'rgba(255, 255, 255, 0.8)'
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                y: {
                    min: 0,
                    max: 3,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        stepSize: 1
                    }
                }
            }
        }
    });
    
    return canvas;
}

// Export functions for global access
window.initializeCharts = initializeCharts;
window.updateLiveChart = updateLiveChart;
window.createInfluencingFactorsChart = createInfluencingFactorsChart;
window.animateChartUpdate = animateChartUpdate;
window.createBattleTimelineChart = createBattleTimelineChart;
