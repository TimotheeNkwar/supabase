document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    const filterButtons = document.querySelectorAll('.filter-btn');
    const sortSelect = document.getElementById('sort-select');
    const postsGrid = document.getElementById('posts-grid');
    const loadMoreButton = document.getElementById('load-more');
    let posts = Array.from(document.querySelectorAll('.post-card'));
    let visiblePostsCount = 3; // Start with first 3 in Featured Insights

    // Initialize: Show all posts
    function initializePosts() {
        posts.forEach((post) => {
            post.classList.remove('hidden');
        });
        loadMoreButton.style.display = 'none'; // Hide load more button
    }

    // Search functionality
    searchInput.addEventListener('input', () => {
        const query = searchInput.value.toLowerCase();
        filterAndSortPosts(query, getActiveFilter(), sortSelect.value);
    });

    // Filter functionality
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            filterButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            const filter = button.getAttribute('onclick').match(/'([^']+)'/)[1];
            filterAndSortPosts(searchInput.value.toLowerCase(), filter, sortSelect.value);
        });
    });

    // Sort functionality
    sortSelect.addEventListener('change', () => {
        filterAndSortPosts(searchInput.value.toLowerCase(), getActiveFilter(), sortSelect.value);
    });

    // Get active filter
    function getActiveFilter() {
        const activeButton = document.querySelector('.filter-btn.active');
        return activeButton.getAttribute('onclick').match(/'([^']+)'/)[1];
    }

    // Filter and sort posts
    function filterAndSortPosts(query, filter, sort) {
        let filteredPosts = posts.filter(post => {
            const title = post.querySelector('h3').textContent.toLowerCase();
            const description = post.querySelector('p').textContent.toLowerCase();
            const category = post.getAttribute('data-category');
            return (title.includes(query) || description.includes(query)) &&
                   (filter === 'all' || category === filter);
        });

        // Sort posts using ISO from data-initial for reliability
        filteredPosts.sort((a, b) => {
            const aIso = a.querySelector('.timestamp-text')?.getAttribute('data-initial') || '';
            const bIso = b.querySelector('.timestamp-text')?.getAttribute('data-initial') || '';
            const aTimestamp = new Date(aIso);
            const bTimestamp = new Date(bIso);
            const aViews = parseInt(a.querySelector('.view-count-text').textContent.replace(/[^0-9]/g, '')) || 0;
            const bViews = parseInt(b.querySelector('.view-count-text').textContent.replace(/[^0-9]/g, '')) || 0;
            const now = new Date('2025-08-02T16:37:00+03:00');
            const aAgeDays = (now - aTimestamp) / (1000 * 60 * 60 * 24);
            const bAgeDays = (now - bTimestamp) / (1000 * 60 * 60 * 24);

            if (sort === 'newest') {
                return bTimestamp - aTimestamp;
            } else if (sort === 'popular') {
                return bViews - aViews;
            } else if (sort === 'trending') {
                const aTrendScore = aViews / (aAgeDays || 1);
                const bTrendScore = bViews / (bAgeDays || 1);
                return bTrendScore - aTrendScore;
            }
        });

        // Update grid - show all filtered posts
        postsGrid.innerHTML = '';
        const allPosts = [...document.querySelectorAll('#featured-posts .post-card'), ...filteredPosts];
        allPosts.forEach((post, index) => {
            post.classList.remove('hidden');
            if (index >= 3) {
                postsGrid.appendChild(post);
            }
        });
        
        // Hide load more button since we're showing all posts
        loadMoreButton.style.display = 'none';
    }

    // Update Load More button visibility (not used but kept for compatibility)
    function updateLoadMoreButton() {
        loadMoreButton.style.display = 'none';
    }

    // Mobile menu toggle
    window.toggleMobileMenu = function () {
        const mobileMenu = document.getElementById('mobile-menu');
        mobileMenu.classList.toggle('hidden');
    };

    // Initialize
    initializePosts();

    // Expose global filter function for inline onclick handlers
    window.filterPosts = function (category) {
        try {
            // Update active button styling
            filterButtons.forEach(btn => btn.classList.remove('active'));
            const matching = Array.from(filterButtons).find(btn => {
                const attr = btn.getAttribute('onclick') || '';
                return attr.includes(`'${category}'`);
            });
            if (matching) matching.classList.add('active');

            // Apply filter using existing logic
            const query = (searchInput.value || '').toLowerCase();
            filterAndSortPosts(query, category, sortSelect.value);
        } catch (e) {
            console.warn('filterPosts failed', e);
        }
    };

    // Optimistic view tracking on "Read More" clicks with beacon to avoid double-count
    function trackViewBeacon(articleId) {
        if (!articleId) return;
        const key = `viewTracked:${articleId}`;
        try {
            // Mark as tracked to prevent article page from re-posting
            localStorage.setItem(key, '1');
        } catch (_) {}
        const url = `/api/track-view/${articleId}`;
        try {
            if (navigator.sendBeacon) {
                const blob = new Blob([], { type: 'application/json' });
                navigator.sendBeacon(url, blob);
            } else {
                fetch(url, { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' } }).catch(() => {});
            }
        } catch (_) {}
    }

    function incrementDomViewCount(articleId) {
        const el = document.querySelector(`.view-count[data-article-id="${articleId}"] .view-count-text`);
        if (!el) return;
        const current = parseInt((el.textContent || '').replace(/[^0-9]/g, '')) || 0;
        const next = current + 1;
        el.textContent = next >= 1000 ? `${(next / 1000).toFixed(1)}K views` : `${next} view${next !== 1 ? 's' : ''}`;
    }

    // Delegate clicks on Read More links to send beacon and optimistically update UI
    document.addEventListener('click', (e) => {
        const link = e.target.closest('a');
        if (!link) return;
        // Matches both featured and recent post links
        const postCard = link.closest('.post-card');
        if (!postCard) return;
        const articleId = postCard.getAttribute('data-article-id');
        if (!articleId) return;
        trackViewBeacon(articleId);
        incrementDomViewCount(articleId);
    });
});