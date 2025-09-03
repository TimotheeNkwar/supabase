import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.45.4/+esm';

const supabaseUrl = 'https://gtinadlpbreniysssjai.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd0aW5hZGxwYnJlbml5c3NzamFpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMTE3MjcsImV4cCI6MjA2OTg4NzcyN30.LLrCSXgAF30gFq5BrHZhc_KEiasF8LfyZTEExbfwjUk';
const supabase = createClient(supabaseUrl, supabaseKey);
// Fonction pour formater les vues
function formatViews(views) {
    if (views >= 1000) {
        return `${(views / 1000).toFixed(1)}K views`;
    }
    return `${views} view${views === 1 ? '' : 's'}`;
}

// Fonction pour mettre à jour l'horodatage
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

// Fonction pour initialiser les vues des articles
async function initializeArticles() {
    try {
        let { data: articles, error } = await supabase
            .from('articles')
            .select('uuid, views, timestamp')
            .eq('hidden', false) // Ne récupérer que les articles visibles
            .order('timestamp', { ascending: false });

        if (error) {
            console.error('Error fetching articles:', error);
            return;
        }

        console.log('Fetched articles:', articles);
        articles.forEach(article => {
            const articleCard = document.querySelector(`.post-card[data-article-id="${article.uuid}"]`);
            if (articleCard) {
                const viewCountElement = articleCard.querySelector('.view-count-text');
                if (viewCountElement) {
                    viewCountElement.textContent = formatViews(article.views);
                }
                if (article.timestamp) {
                    updateTimestamp(article.uuid, article.timestamp);
                }
            }
        });
    } catch (error) {
        console.error('Error initializing articles:', error);
    }
}

// S'abonner aux modifications en temps réel
supabase
    .channel('public:articles')
    .on('postgres_changes', { event: 'UPDATE', schema: 'public', table: 'articles' }, (payload) => {
        const articleId = payload.new.uuid;
        const views = payload.new.views;
        const timestamp = payload.new.timestamp;

        console.log(`Article ${articleId} updated: ${views} views`);
        const articleCard = document.querySelector(`.post-card[data-article-id="${articleId}"]`);
        if (articleCard) {
            const viewCountElement = articleCard.querySelector('.view-count-text');
            if (viewCountElement) {
                viewCountElement.textContent = formatViews(views);
            }
            if (timestamp) {
                updateTimestamp(articleId, timestamp);
            }
        }
    })
    .subscribe((status) => {
        console.log('Supabase Realtime subscription status:', status);
    });

// Déclencher une vue lorsque l'article est visible
async function trackView(articleId) {
    try {
        const response = await fetch(`/api/track-view/${articleId}`, { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            console.log(`Updated views for article ${articleId}: ${data.views}`);
        } else {
            console.error(`Error tracking view for article ${articleId}: ${data.error}`);
        }
    } catch (error) {
        console.error(`Error tracking view for article ${articleId}:`, error);
    }
}

// Initialiser les horodatages et les vues au chargement
document.addEventListener('DOMContentLoaded', () => {
    // Initialiser les horodatages à partir des attributs data-timestamp
    document.querySelectorAll('.timestamp').forEach(timestamp => {
        const articleId = timestamp.getAttribute('data-article-id');
        const timestampValue = timestamp.getAttribute('data-timestamp');
        if (timestampValue) {
            updateTimestamp(articleId, timestampValue);
        }
    });

    // Récupérer les vues initiales des articles
    initializeArticles();

    // Déclencher track_view pour les articles visibles
    document.querySelectorAll('.post-card').forEach(card => {
        const articleId = card.getAttribute('data-article-id');
        const observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                trackView(articleId);
                observer.disconnect();
            }
        }, { threshold: 0.5 });
        observer.observe(card);
    });

    // Pour les pages d'article unique (article.html)
    const articleId = document.querySelector('.post-card')?.getAttribute('data-article-id');
    if (articleId) {
        trackView(articleId);
    }
});
