const socket = io();

socket.on('connect', () => {
    console.log('Connected to WebSocket server');
});

socket.on('disconnect', () => {
    console.log('Disconnected from WebSocket server');
});

socket.on('article_update', (data) => {
    const articleId = data.id;
    const timestamp = data.timestamp;
    const views = data.views;

    // Update timestamp
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

    // Update view count
    const viewCountElement = document.querySelector(`.view-count[data-article-id="${articleId}"] .view-count-text`);
    if (viewCountElement && typeof views !== 'undefined') {
        viewCountElement.textContent = formatViews(views);
    }
});

// Format views (e.g., 1000 -> "1K views")
function formatViews(views) {
    if (views >= 1000) {
        return `${(views / 1000).toFixed(1)}K views`;
    }
    return `${views} view${views === 1 ? '' : 's'}`;
}

function updateTimestamp(articleId) {
    const timestampElement = document.querySelector(`.timestamp[data-article-id="${articleId}"] .timestamp-text`);
    if (timestampElement) {
        const now = new Date().toLocaleString('en-US', { timeZone: 'Asia/Nicosia', hour12: true });
        timestampElement.textContent = now;
    }
}

// Optionnel : mettre à jour l'heure locale toutes les secondes
document.querySelectorAll('.timestamp').forEach(timestamp => {
    const articleId = timestamp.getAttribute('data-article-id');
    updateTimestamp(articleId);
    setInterval(() => updateTimestamp(articleId), 1000);
});