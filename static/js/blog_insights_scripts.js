document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    const filterButtons = document.querySelectorAll('.filter-btn');
    const sortSelect = document.getElementById('sort-select');
    const postsGrid = document.getElementById('posts-grid');
    const loadMoreButton = document.getElementById('load-more');
    let posts = Array.from(document.querySelectorAll('.post-card'));
    let visiblePostsCount = 3; // Start with first 3 in Featured Insights

    // Initialize: Show first 3 posts in Recent Articles
    function initializePosts() {
        posts.forEach((post, index) => {
            post.classList.add('hidden');
            if (index >= 3 && index < visiblePostsCount + 3) {
                post.classList.remove('hidden');
            }
        });
        updateLoadMoreButton();
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

    // Load More functionality
    loadMoreButton.addEventListener('click', () => {
        visiblePostsCount += 3;
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

        // Sort posts
        filteredPosts.sort((a, b) => {
            const aTimestamp = new Date(a.querySelector('.timestamp-text').textContent);
            const bTimestamp = new Date(b.querySelector('.timestamp-text').textContent);
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

        // Update grid
        postsGrid.innerHTML = '';
        const allPosts = [...document.querySelectorAll('#featured-posts .post-card'), ...filteredPosts];
        allPosts.forEach((post, index) => {
            if (index < 3 || index < visiblePostsCount + 3) {
                post.classList.remove('hidden');
            } else {
                post.classList.add('hidden');
            }
            if (index >= 3) {
                postsGrid.appendChild(post);
            }
        });

        updateLoadMoreButton();
    }

    // Update Load More button visibility
    function updateLoadMoreButton() {
        const totalPosts = posts.filter(post => {
            const title = post.querySelector('h3').textContent.toLowerCase();
            const description = post.querySelector('p').textContent.toLowerCase();
            const category = post.getAttribute('data-category');
            const query = searchInput.value.toLowerCase();
            const filter = getActiveFilter();
            return (title.includes(query) || description.includes(query)) &&
                   (filter === 'all' || category === filter);
        }).length;
        loadMoreButton.style.display = visiblePostsCount + 3 >= totalPosts ? 'none' : 'inline-flex';
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
});