// Assure-toi d'avoir importé supabase-js dans ta page HTML
// <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js"></script>

// Configuration Supabase
const supabaseUrl = 'https://gtinadlpbreniysssjai.supabase.co'; // Ton URL Supabase
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd0aW5hZGxwYnJlbml5c3NzamFpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMTE3MjcsImV4cCI6MjA2OTg4NzcyN30.LLrCSXgAF30gFq5BrHZhc_KEiasF8LfyZTEExbfwjUk'; // Clé publique (anon key)
const supabaseClient = (window.supabase && window.supabase.createClient)
  ? window.supabase.createClient(supabaseUrl, supabaseKey)
  : null;

if (!supabaseClient) {
  console.error('[Realtime] Supabase library not loaded before realtime-updates.js');
}

// Abonnement aux changements en temps réel sur la table articles
supabaseClient && supabaseClient
  .channel('public:articles')
  .on(
    'postgres_changes',
    { event: 'UPDATE', schema: 'public', table: 'articles' },
    (payload) => {
      console.debug('[Realtime] Update received', payload);
      const article = payload.new;
      const articleId = article.id || article.uuid;
      const timestamp = article.timestamp;
      const views = article.views;

      // Met à jour le timestamp dans le DOM
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

      // Met à jour le nombre de vues dans le DOM
      const viewCountElement = document.querySelector(`.view-count[data-article-id="${articleId}"] .view-count-text`);
      if (viewCountElement && typeof views !== 'undefined') {
        viewCountElement.textContent = formatViews(views);
      }
    }
  )
  .subscribe();

// Formate le nombre de vues (ex: 1000 -> "1K views")
function formatViews(views) {
  if (views >= 1000) {
    return `${(views / 1000).toFixed(1)}K views`;
  }
  return `${views} view${views === 1 ? '' : 's'}`;
}

// Met à jour l'heure locale pour chaque article (optionnel)
function updateTimestamp(articleId) {
  const timestampElement = document.querySelector(`.timestamp[data-article-id="${articleId}"] .timestamp-text`);
  if (timestampElement) {
    const now = new Date().toLocaleString('en-US', { timeZone: 'Asia/Nicosia', hour12: true });
    timestampElement.textContent = now;
  }
}

// Met à jour l'heure locale toutes les secondes (optionnel)
document.querySelectorAll('.timestamp').forEach(timestamp => {
  const articleId = timestamp.getAttribute('data-article-id');
  updateTimestamp(articleId);
  setInterval(() => updateTimestamp(articleId), 1000);
});