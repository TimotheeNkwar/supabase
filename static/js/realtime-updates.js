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

    // Find article card by data-article-id
    const articleCard = document.querySelector(`.post-card[data-article-id="${articleId}"]`);
    if (articleCard) {
        // Update timestamp
        const timestampElement = articleCard.querySelector('.timestamp-text');
        if (timestampElement) {
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
        const viewCountElement = articleCard.querySelector('.view-count-text');
        if (viewCountElement) {
            viewCountElement.textContent = formatViews(views);
        }
    }
});

// Format views (e.g., 1000 -> "1K views")
function formatViews(views) {
    if (views >= 1000) {
        return `${(views / 1000).toFixed(1)}K views`;
    }
    return `${views} view${views === 1 ? '' : 's'}`;
}
const socket = io.connect('http://' + document.domain + ':' + location.port);

function updateTimestamp(articleId) {
    const timestampElement = document.querySelector(`.timestamp[data-article-id="${articleId}"] .timestamp-text`);
    if (timestampElement) {
        const now = new Date().toLocaleString('en-US', { timeZone: 'Asia/Nicosia', hour12: true });
        timestampElement.textContent = now;
    }
}

socket.on('article_update', function(data) {
    if (data.id) {
        updateTimestamp(data.id);
        const viewCountElement = document.querySelector(`.view-count[data-article-id="${data.id}"] .view-count-text`);
        if (viewCountElement && data.views) {
            viewCountElement.textContent = data.views >= 1000 ? `${(data.views / 1000).toFixed(1)}K views` : `${data.views} view${data.views !== 1 ? 's' : ''}`;
        }
    }
});

document.querySelectorAll('.timestamp').forEach(timestamp => {
    const articleId = timestamp.getAttribute('data-article-id');
    updateTimestamp(articleId);
    setInterval(() => updateTimestamp(articleId), 1000);
});