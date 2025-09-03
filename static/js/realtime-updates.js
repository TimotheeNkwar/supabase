const socket = io();

function formatViews(views) {
    if (views >= 1000) {
        return `${(views / 1000).toFixed(1)}K views`;
    }
    return `${views} view${views === 1 ? '' : 's'}`;
}

function updateTimestamp(articleId, timestamp) {
    const timestampElement = document.querySelector(`.timestamp[data-article-id="${articleId}"] .timestamp-text`);
    if (timestampElement && timestamp) {
        const date = new Date(timestamp.replace('Z', '+00:00'));
        const options = {
            year: 'numeric',
            month: 'long',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
            timeZone: 'Asia/Nicosia'
        };
        timestampElement.textContent = date.toLocaleString('en-US', options);
    }
}

// Initialiser les horodatages au chargement
document.querySelectorAll('.timestamp').forEach(timestamp => {
    const articleId = timestamp.getAttribute('data-article-id');
    const timestampValue = timestamp.getAttribute('data-timestamp');
    if (timestampValue) {
        updateTimestamp(articleId, timestampValue);
    }
});

// Gérer les connexions WebSocket
socket.on('connect', () => {
    console.log('Connected to WebSocket server');
    // Émettre track_view pour chaque article visible
    document.querySelectorAll('.post-card').forEach(card => {
        const articleId = card.getAttribute('data-article-id');
        const observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                socket.emit('track_view', { article_id: articleId });
                observer.disconnect();
            }
        }, { threshold: 0.5 });
        observer.observe(card);
    });
});

socket.on('connect_error', (error) => {
    console.error('WebSocket connection error:', error);
});

socket.on('disconnect', () => {
    console.log('Disconnected from WebSocket server');
});

socket.on('article_update', (data) => {
    console.log('Received article_update:', data);
    const articleId = data.id;
    const views = data.views;
    const timestamp = data.timestamp;

    const articleCard = document.querySelector(`.post-card[data-article-id="${articleId}"]`);
    if (articleCard) {
        const viewCountElement = articleCard.querySelector('.view-count-text');
        if (viewCountElement) {
            viewCountElement.textContent = formatViews(views);
        }
        if (timestamp) {
            updateTimestamp(articleId, timestamp);
        }
    } else {
        console.warn(`Article card not found for ID: ${articleId}`);
    }
});