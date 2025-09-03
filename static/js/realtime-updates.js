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

// Initialiser les horodatages au chargement de la page
document.querySelectorAll('.timestamp').forEach(timestamp => {
    const articleId = timestamp.getAttribute('data-article-id');
    const timestampValue = timestamp.getAttribute('data-timestamp'); // Supposons que l'horodatage est passé via un attribut data
    if (timestampValue) {
        updateTimestamp(articleId, timestampValue);
    }
});

// Mettre à jour l'horodatage lors de la réception de article_update
socket.on('article_update', (data) => {
    const articleId = data.id;
    const views = data.views;
    const timestamp = data.timestamp; // Seulement si le serveur envoie timestamp

    const articleCard = document.querySelector(`.post-card[data-article-id="${articleId}"]`);
    if (articleCard) {
        // Mettre à jour les vues
        const viewCountElement = articleCard.querySelector('.view-count-text');
        if (viewCountElement) {
            viewCountElement.textContent = formatViews(views);
        }
        // Mettre à jour l'horodatage si fourni
        if (timestamp) {
            updateTimestamp(articleId, timestamp);
        }
    }
});