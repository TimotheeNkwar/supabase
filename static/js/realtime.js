async function initializeArticles() {
    try {
        let { data: articles, error } = await supabase
            .from('articles')
            .select('uuid, views, clicks, timestamp')
            .eq('hidden', false)
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
                    viewCountElement.textContent = formatCount(article.views, 'view');
                }
                const clickCountElement = articleCard.querySelector('.click-count-text');
                if (clickCountElement) {
                    clickCountElement.textContent = formatCount(article.clicks, 'click');
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

supabase
    .channel('public:articles')
    .on('postgres_changes', { event: 'UPDATE', schema: 'public', table: 'articles' }, (payload) => {
        const articleId = payload.new.uuid;
        const views = payload.new.views;
        const clicks = payload.new.clicks;
        const timestamp = payload.new.timestamp;

        console.log(`Article ${articleId} updated: ${views} views, ${clicks} clicks`);
        const articleCard = document.querySelector(`.post-card[data-article-id="${articleId}"]`);
        if (articleCard) {
            const viewCountElement = articleCard.querySelector('.view-count-text');
            if (viewCountElement && views !== undefined) {
                viewCountElement.textContent = formatCount(views, 'view');
            }
            const clickCountElement = articleCard.querySelector('.click-count-text');
            if (clickCountElement && clicks !== undefined) {
                clickCountElement.textContent = formatCount(clicks, 'click');
            }
            if (timestamp) {
                updateTimestamp(articleId, timestamp);
            }
        }
    })
    .subscribe((status) => {
        console.log('Supabase Realtime subscription status:', status);
    });