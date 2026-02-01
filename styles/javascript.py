"""JavaScript code for the Streamlit app"""

HIDE_COLLAPSE_BUTTON_JS = """
<script>
    // Force hide sidebar collapse button with JavaScript
    function hideCollapseButton() {
        const selectors = [
            'button[data-testid="stSidebarCollapseButton"]',
            '[data-testid="collapsedControl"]',
            '[data-testid="stSidebarCollapsedControl"]',
            'button[aria-label*="Close"]',
            'button[aria-label*="close"]',
            'button[aria-label*="Collapse"]',
            'button[aria-label*="collapse"]',
            'button[kind="header"]',
            'section[data-testid="stSidebar"] > div:first-child > button',
            'section[data-testid="stSidebar"] > div:first-child > button:first-child',
            'header button',
            'header > div > button',
            '[data-testid="stHeader"] button',
            'button[data-baseweb="button"][aria-label]',
            'button:has(svg[viewBox*="0 0 24 24"])',
            'button svg[viewBox*="0 0 24 24"]'
        ];
        
        selectors.forEach(selector => {
            try {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    const ariaLabel = el.getAttribute('aria-label') || '';
                    const testId = el.getAttribute('data-testid') || '';
                    const hasCollapseIcon = el.querySelector('svg[viewBox*="0 0 24 24"]');
                    
                    if (testId.includes('Collapse') || 
                        testId.includes('collapsed') ||
                        ariaLabel.toLowerCase().includes('collapse') ||
                        ariaLabel.toLowerCase().includes('close') ||
                        hasCollapseIcon ||
                        el.getAttribute('kind') === 'header') {
                        el.style.display = 'none';
                        el.style.visibility = 'hidden';
                        el.style.opacity = '0';
                        el.style.width = '0';
                        el.style.height = '0';
                        el.style.padding = '0';
                        el.style.margin = '0';
                        el.style.pointerEvents = 'none';
                        el.style.position = 'absolute';
                        el.style.left = '-9999px';
                        el.style.zIndex = '-9999';
                        el.remove();
                    }
                });
            } catch(e) {}
        });
        
        try {
            const headerButtons = document.querySelectorAll('header button, [data-testid="stHeader"] button');
            headerButtons.forEach(btn => {
                const svg = btn.querySelector('svg');
                if (svg && svg.getAttribute('viewBox') && svg.getAttribute('viewBox').includes('24')) {
                    btn.style.display = 'none';
                    btn.remove();
                }
            });
        } catch(e) {}
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hideCollapseButton);
    } else {
        hideCollapseButton();
    }
    setInterval(hideCollapseButton, 50);
    new MutationObserver(hideCollapseButton).observe(document.body, { childList: true, subtree: true, attributes: true });
</script>
"""


