

// Assure-toi d'avoir importé supabase-js dans ta page HTML
// <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js"></script>

const supabaseUrl = 'https://gtinadlpbreniysssjai.supabase.co'; // Remplace par ton URL
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd0aW5hZGxwYnJlbml5c3NzamFpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDMxMTcyNywiZXhwIjoyMDY5ODg3NzI3fQ.nIVXdQrwgNLNmITrn3iYAGngzW-l6V86fYO62gJyp24
SUPABASE_URL=https:'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd0aW5hZGxwYnJlbml5c3NzamFpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMTE3MjcsImV4cCI6MjA2OTg4NzcyN30.LLrCSXgAF30gFq5BrHZhc_KEiasF8LfyZTEExbfwjUk'; // Remplace par ta clé publique (anon key)
const supabase = window.supabase || supabase.createClient(supabaseUrl, supabaseKey);

supabase
  .channel('public:articles')
  .on(
    'postgres_changes',
    { event: 'UPDATE', schema: 'public', table: 'articles' },
    (payload) => {
      const article = payload.new;
      const articleId = article.uuid || article.id;
      const timestamp = article.timestamp;
      const views = article.views;

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
    }
  )
  .subscribe();

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