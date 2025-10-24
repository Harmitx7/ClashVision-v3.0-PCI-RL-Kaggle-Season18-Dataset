// 3D Tilted Card Effects for ClashVision
class TiltedCardEffect {
    constructor() {
        this.cards = [];
        this.init();
    }

    init() {
        // Initialize all tilted cards
        this.setupCards();
        
        // Re-initialize when new content is added
        document.addEventListener('DOMContentLoaded', () => this.setupCards());
    }

    setupCards() {
        const cardSelectors = [
            '.bg-white.dark\\:bg-gray-800', // All main cards
            '[class*="rounded-xl"]', // Rounded cards
            '.clash-card' // Custom clash cards if any
        ];

        cardSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                if (!element.classList.contains('tilted-card-initialized')) {
                    this.initializeCard(element);
                }
            });
        });
    }

    initializeCard(element) {
        // Skip if already initialized or if it's a button/input
        if (element.classList.contains('tilted-card-initialized') || 
            element.tagName === 'BUTTON' || 
            element.tagName === 'INPUT' ||
            element.closest('button') ||
            element.closest('input')) {
            return;
        }

        // Add tilted card classes
        element.classList.add('tilted-card', 'tilted-card-initialized');
        
        // Don't wrap content - apply tilt directly to the card element
        // Just add the classes, keep original content structure

        // Add event listeners
        this.addEventListeners(element);
        
        // Store reference
        this.cards.push(element);
    }

    addEventListeners(card) {
        let isHovering = false;

        card.addEventListener('mouseenter', () => {
            isHovering = true;
            card.style.transition = 'transform 0.1s ease-out';
        });

        card.addEventListener('mouseleave', () => {
            isHovering = false;
            // Reset transform smoothly
            card.style.transition = 'transform 0.3s ease-out';
            card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale(1)';
        });

        card.addEventListener('mousemove', (e) => {
            if (!isHovering) return;

            const rect = card.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            
            const mouseX = e.clientX - centerX;
            const mouseY = e.clientY - centerY;
            
            const rotateX = (mouseY / (rect.height / 2)) * -8; // Max 8 degrees
            const rotateY = (mouseX / (rect.width / 2)) * 8;   // Max 8 degrees
            
            // Apply transform directly to the entire card
            card.style.transition = 'none'; // Remove transition during mouse move for smooth tracking
            card.style.transform = `
                perspective(1000px) 
                rotateX(${rotateX}deg) 
                rotateY(${rotateY}deg) 
                scale(1.03)
            `;
        });

        // Touch events for mobile (simplified)
        card.addEventListener('touchstart', (e) => {
            e.preventDefault();
            card.style.transform = 'perspective(1000px) scale(1.02)';
        });

        card.addEventListener('touchend', () => {
            card.style.transform = 'perspective(1000px) scale(1)';
        });
    }

    // Method to reinitialize cards (useful after dynamic content updates)
    refresh() {
        this.setupCards();
    }

    // Method to add tilt effect to specific element
    addToElement(element) {
        this.initializeCard(element);
    }
}

// Initialize the tilted card system
const tiltedCards = new TiltedCardEffect();

// Make it globally available for dynamic content
window.TiltedCards = tiltedCards;
