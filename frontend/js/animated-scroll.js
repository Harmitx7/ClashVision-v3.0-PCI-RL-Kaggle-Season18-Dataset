// Animated Scroll Effects for ClashVision
class AnimatedScrollEffect {
    constructor() {
        this.observers = [];
        this.scrollContainers = [];
        this.init();
    }

    init() {
        // Initialize intersection observer for scroll animations
        this.setupIntersectionObserver();
        
        // Setup existing scroll containers
        this.setupScrollContainers();
        
        // Re-initialize when new content is added
        document.addEventListener('DOMContentLoaded', () => this.setupScrollContainers());
    }

    setupIntersectionObserver() {
        // Create intersection observer for items coming into view
        this.itemObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('in-view');
                } else {
                    entry.target.classList.remove('in-view');
                }
            });
        }, {
            threshold: 0.1, // Trigger when only 10% of item is visible (more sensitive)
            rootMargin: '100px 0px -20px 0px' // Start animation earlier and continue longer
        });
    }

    setupScrollContainers() {
        // Find containers that should have animated scroll effects
        const containers = document.querySelectorAll('#recentBattles, #recommendations, #playerStats, #playerInfo');
        
        containers.forEach(container => {
            if (!container.classList.contains('animated-scroll-initialized')) {
                this.initializeScrollContainer(container);
            }
        });
    }

    initializeScrollContainer(container) {
        // Skip if already initialized
        if (container.classList.contains('animated-scroll-initialized')) {
            return;
        }

        // Add container classes
        container.classList.add('animated-scroll-container', 'animated-scroll-initialized');
        
        // Wrap content in scroll list if needed
        if (!container.querySelector('.animated-scroll-list')) {
            const content = container.innerHTML;
            container.innerHTML = `
                <div class="animated-scroll-list">
                    ${content}
                </div>
                <div class="scroll-gradient-top"></div>
                <div class="scroll-gradient-bottom"></div>
            `;
        }

        const scrollList = container.querySelector('.animated-scroll-list');
        const topGradient = container.querySelector('.scroll-gradient-top');
        const bottomGradient = container.querySelector('.scroll-gradient-bottom');

        // Add scroll event listener for gradient effects
        if (scrollList && topGradient && bottomGradient) {
            scrollList.addEventListener('scroll', (e) => {
                this.handleScroll(e, topGradient, bottomGradient);
            });
        }

        // Initialize items in this container
        this.initializeScrollItems(container);
        
        // Store reference
        this.scrollContainers.push(container);
    }

    initializeScrollItems(container) {
        // Find all items that should be animated
        const items = container.querySelectorAll('div[class*="flex items-center"], div[class*="space-y"], div[class*="p-3"], div[class*="rounded-lg"]:not(.scroll-gradient-top):not(.scroll-gradient-bottom)');
        
        items.forEach((item, index) => {
            if (!item.classList.contains('animated-scroll-item')) {
                item.classList.add('animated-scroll-item');
                
                // Add staggered delay - more dramatic
                item.style.transitionDelay = `${index * 0.1}s`;
                
                // Observe this item for intersection
                this.itemObserver.observe(item);
            }
        });
    }

    handleScroll(event, topGradient, bottomGradient) {
        const { scrollTop, scrollHeight, clientHeight } = event.target;
        
        // Calculate gradient opacities
        const topOpacity = Math.min(scrollTop / 50, 1);
        const bottomDistance = scrollHeight - (scrollTop + clientHeight);
        const bottomOpacity = scrollHeight <= clientHeight ? 0 : Math.min(bottomDistance / 50, 1);
        
        // Apply gradient opacities
        if (topGradient) {
            topGradient.style.opacity = topOpacity;
        }
        if (bottomGradient) {
            bottomGradient.style.opacity = bottomOpacity;
        }
    }

    // Method to add animated scroll to specific container
    addToContainer(container) {
        this.initializeScrollContainer(container);
    }

    // Method to refresh all scroll containers
    refresh() {
        this.setupScrollContainers();
    }

    // Method to animate items in a specific container
    animateItems(container) {
        const items = container.querySelectorAll('.animated-scroll-item');
        
        items.forEach((item, index) => {
            // Remove in-view class temporarily
            item.classList.remove('in-view');
            
            // Add it back with a delay for staggered animation
            setTimeout(() => {
                item.classList.add('in-view');
            }, index * 50);
        });
    }

    // Method to add selection effects
    selectItem(container, index) {
        const items = container.querySelectorAll('.animated-scroll-item');
        
        // Remove selection from all items
        items.forEach(item => item.classList.remove('selected'));
        
        // Add selection to specific item
        if (items[index]) {
            items[index].classList.add('selected');
        }
    }

    // Clean up observers
    destroy() {
        if (this.itemObserver) {
            this.itemObserver.disconnect();
        }
        this.observers.forEach(observer => observer.disconnect());
        this.observers = [];
        this.scrollContainers = [];
    }
}

// Initialize the animated scroll system
const animatedScroll = new AnimatedScrollEffect();

// Make it globally available
window.AnimatedScroll = animatedScroll;
