// Animation utilities and effects
class AnimationManager {
    constructor() {
        this.activeAnimations = new Map();
        this.init();
    }
    
    init() {
        this.setupScrollAnimations();
        this.setupHoverEffects();
        this.setupLoadingAnimations();
    }
    
    setupScrollAnimations() {
        // Animate elements when they come into view
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateElementIn(entry.target);
                }
            });
        }, observerOptions);
        
        // Observe all cards and sections
        document.querySelectorAll('.clash-card').forEach(card => {
            observer.observe(card);
        });
    }
    
    setupHoverEffects() {
        // Add hover animations to interactive elements
        document.querySelectorAll('.card-animation').forEach(element => {
            element.addEventListener('mouseenter', () => {
                this.animateHover(element, true);
            });
            
            element.addEventListener('mouseleave', () => {
                this.animateHover(element, false);
            });
        });
    }
    
    setupLoadingAnimations() {
        // Create loading spinner animations
        const spinners = document.querySelectorAll('.animate-spin');
        spinners.forEach(spinner => {
            this.createSpinAnimation(spinner);
        });
    }
    
    animateElementIn(element) {
        if (element.classList.contains('animated')) return;
        
        element.classList.add('animated');
        
        anime({
            targets: element,
            opacity: [0, 1],
            translateY: [30, 0],
            scale: [0.95, 1],
            duration: 800,
            easing: 'easeOutCubic',
            delay: Math.random() * 200
        });
    }
    
    animateHover(element, isHovering) {
        const animationId = `hover_${element.id || Math.random()}`;
        
        // Cancel existing animation
        if (this.activeAnimations.has(animationId)) {
            this.activeAnimations.get(animationId).pause();
        }
        
        const animation = anime({
            targets: element,
            scale: isHovering ? 1.05 : 1,
            translateY: isHovering ? -5 : 0,
            boxShadow: isHovering ? 
                '0 10px 30px rgba(0, 0, 0, 0.3)' : 
                '0 4px 15px rgba(0, 0, 0, 0.1)',
            duration: 300,
            easing: 'easeOutCubic'
        });
        
        this.activeAnimations.set(animationId, animation);
    }
    
    createSpinAnimation(element) {
        anime({
            targets: element,
            rotate: 360,
            duration: 1000,
            easing: 'linear',
            loop: true
        });
    }
    
    animateWinProbabilityChange(newValue, oldValue) {
        const element = document.getElementById('winProbabilityText');
        if (!element) return;
        
        const isIncrease = newValue > oldValue;
        const color = isIncrease ? '#7ED321' : '#D0021B';
        
        // Create floating indicator
        const indicator = document.createElement('div');
        indicator.textContent = isIncrease ? '+' : '-';
        indicator.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: ${color};
            font-size: 24px;
            font-weight: bold;
            pointer-events: none;
            z-index: 10;
        `;
        
        element.parentElement.style.position = 'relative';
        element.parentElement.appendChild(indicator);
        
        // Animate the main value
        anime({
            targets: element,
            scale: [1, 1.2, 1],
            color: [element.style.color, color, element.style.color],
            duration: 600,
            easing: 'easeOutBack'
        });
        
        // Animate the indicator
        anime({
            targets: indicator,
            opacity: [0, 1, 0],
            translateY: [0, -20, -40],
            scale: [0.5, 1, 0.5],
            duration: 1000,
            easing: 'easeOutCubic',
            complete: () => {
                if (element.parentElement.contains(indicator)) {
                    element.parentElement.removeChild(indicator);
                }
            }
        });
    }
    
    animateCardDeployment(cardName, position) {
        // Create card deployment animation
        const card = document.createElement('div');
        card.className = 'card-deployment-animation';
        card.textContent = cardName;
        card.style.cssText = `
            position: fixed;
            top: ${position.y}px;
            left: ${position.x}px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        `;
        
        document.body.appendChild(card);
        
        anime({
            targets: card,
            scale: [0, 1.2, 1],
            opacity: [0, 1, 1, 0],
            translateY: [0, -50, -100],
            rotate: [0, 360],
            duration: 2000,
            easing: 'easeOutCubic',
            complete: () => {
                if (document.body.contains(card)) {
                    document.body.removeChild(card);
                }
            }
        });
    }
    
    animateElixirDrop(amount, position) {
        // Create elixir drop animation
        const drop = document.createElement('div');
        drop.className = 'elixir-drop-animation';
        drop.textContent = `+${amount}`;
        drop.style.cssText = `
            position: fixed;
            top: ${position.y}px;
            left: ${position.x}px;
            color: #9B59B6;
            font-weight: bold;
            font-size: 18px;
            z-index: 1000;
            pointer-events: none;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        `;
        
        document.body.appendChild(drop);
        
        anime({
            targets: drop,
            opacity: [0, 1, 0],
            translateY: [0, -30],
            scale: [0.5, 1, 0.5],
            duration: 1500,
            easing: 'easeOutCubic',
            complete: () => {
                if (document.body.contains(drop)) {
                    document.body.removeChild(drop);
                }
            }
        });
    }
    
    animateTowerDamage(damage, towerElement) {
        if (!towerElement) return;
        
        // Shake animation for tower damage
        anime({
            targets: towerElement,
            translateX: [0, -10, 10, -5, 5, 0],
            duration: 500,
            easing: 'easeOutCubic'
        });
        
        // Create damage number
        const damageText = document.createElement('div');
        damageText.textContent = `-${damage}`;
        damageText.style.cssText = `
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            color: #D0021B;
            font-weight: bold;
            font-size: 16px;
            z-index: 10;
            pointer-events: none;
        `;
        
        towerElement.style.position = 'relative';
        towerElement.appendChild(damageText);
        
        anime({
            targets: damageText,
            opacity: [0, 1, 0],
            translateY: [0, -30],
            duration: 1000,
            easing: 'easeOutCubic',
            complete: () => {
                if (towerElement.contains(damageText)) {
                    towerElement.removeChild(damageText);
                }
            }
        });
    }
    
    animateVictory() {
        // Create victory celebration animation
        const celebration = document.createElement('div');
        celebration.className = 'victory-celebration';
        celebration.innerHTML = `
            <div class="victory-text">VICTORY!</div>
            <div class="victory-crown">👑</div>
        `;
        celebration.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            z-index: 2000;
            pointer-events: none;
        `;
        
        const style = document.createElement('style');
        style.textContent = `
            .victory-text {
                font-size: 48px;
                font-weight: bold;
                color: #FFD700;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                margin-bottom: 20px;
            }
            .victory-crown {
                font-size: 64px;
            }
        `;
        document.head.appendChild(style);
        document.body.appendChild(celebration);
        
        // Animate celebration
        anime({
            targets: celebration,
            scale: [0, 1.2, 1],
            opacity: [0, 1, 1, 0],
            rotate: [0, 10, -10, 0],
            duration: 3000,
            easing: 'easeOutElastic(1, .8)',
            complete: () => {
                if (document.body.contains(celebration)) {
                    document.body.removeChild(celebration);
                }
                if (document.head.contains(style)) {
                    document.head.removeChild(style);
                }
            }
        });
        
        // Add confetti effect
        this.createConfetti();
    }
    
    animateDefeat() {
        // Create defeat animation
        const defeat = document.createElement('div');
        defeat.className = 'defeat-animation';
        defeat.innerHTML = `
            <div class="defeat-text">DEFEAT</div>
            <div class="defeat-icon">💀</div>
        `;
        defeat.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            z-index: 2000;
            pointer-events: none;
        `;
        
        const style = document.createElement('style');
        style.textContent = `
            .defeat-text {
                font-size: 48px;
                font-weight: bold;
                color: #D0021B;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                margin-bottom: 20px;
            }
            .defeat-icon {
                font-size: 64px;
            }
        `;
        document.head.appendChild(style);
        document.body.appendChild(defeat);
        
        anime({
            targets: defeat,
            scale: [0, 1],
            opacity: [0, 1, 1, 0],
            duration: 2500,
            easing: 'easeOutCubic',
            complete: () => {
                if (document.body.contains(defeat)) {
                    document.body.removeChild(defeat);
                }
                if (document.head.contains(style)) {
                    document.head.removeChild(style);
                }
            }
        });
    }
    
    createConfetti() {
        const colors = ['#FFD700', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'];
        const confettiCount = 50;
        
        for (let i = 0; i < confettiCount; i++) {
            const confetti = document.createElement('div');
            confetti.style.cssText = `
                position: fixed;
                top: -10px;
                left: ${Math.random() * 100}vw;
                width: 10px;
                height: 10px;
                background: ${colors[Math.floor(Math.random() * colors.length)]};
                z-index: 1999;
                pointer-events: none;
            `;
            
            document.body.appendChild(confetti);
            
            anime({
                targets: confetti,
                translateY: '100vh',
                translateX: `${(Math.random() - 0.5) * 200}px`,
                rotate: 720,
                opacity: [1, 0],
                duration: 3000 + Math.random() * 2000,
                easing: 'easeInCubic',
                complete: () => {
                    if (document.body.contains(confetti)) {
                        document.body.removeChild(confetti);
                    }
                }
            });
        }
    }
    
    pulseElement(element, color = '#667eea') {
        anime({
            targets: element,
            boxShadow: [
                `0 0 0 0 ${color}`,
                `0 0 0 20px transparent`
            ],
            duration: 1000,
            easing: 'easeOutCubic',
            loop: 3
        });
    }
    
    cleanup() {
        // Clean up all active animations
        this.activeAnimations.forEach(animation => {
            animation.pause();
        });
        this.activeAnimations.clear();
    }
}

// Create global animation manager
window.animationManager = new AnimationManager();

// Clean up animations when page unloads
window.addEventListener('beforeunload', () => {
    if (window.animationManager) {
        window.animationManager.cleanup();
    }
});
