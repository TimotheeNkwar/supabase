/**
 * Fetches data with retry logic to handle rate limiting and transient errors.
 * @param {string} url - The API endpoint to fetch from.
 * @param {object} options - Fetch options (method, headers, body, etc.).
 * @param {number} [retries=5] - Number of retry attempts.
 * @param {number} [delay=3000] - Initial delay between retries (ms).
 * @returns {Promise<object>} - The JSON response from the server.
 * @throws {Error} - If all retries fail or an unhandled error occurs.
 */
async function fetchWithRetrySentiment(url, options, retries = 5, delay = 3000) {
    for (let i = 0; i < retries; i++) {
        try {
            console.log(`[Sentiment] Attempt ${i + 1} to fetch ${url}`);
            const response = await fetch(url, options);
            console.log(`[Sentiment] Response status: ${response.status}`);
            if (response.status === 429) {
                throw new Error('Rate limit exceeded');
            }
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`[Sentiment] Fetch attempt ${i + 1} failed:`, error);
            if (i === retries - 1) {
                throw error;
            }
            const backoffDelay = delay * Math.pow(2, i);
            await new Promise(resolve => setTimeout(resolve, backoffDelay));
        }
    }
}

/**
 * Analyzes sentiment of the input text and displays results.
 */
async function analyzeSentiment() {
    // Get DOM elements
    const elements = {
        text: document.getElementById('sentiment-text'),
        language: document.getElementById('language-select'),
        result: document.getElementById('sentiment-result'),
        errorMessage: document.getElementById('error-message'),
        label: document.getElementById('sentiment-label'),
        positiveScore: document.getElementById('positive-score'),
        positiveBar: document.getElementById('positive-bar'),
        neutralScore: document.getElementById('neutral-score'),
        neutralBar: document.getElementById('neutral-bar'),
        negativeScore: document.getElementById('negative-score'),
        negativeBar: document.getElementById('negative-bar'),
        insights: document.getElementById('sentiment-insights'),
        button: document.querySelector('button.btn-accent'),
        buttonText: document.getElementById('button-text'),
        loadingSpinner: document.getElementById('loading-spinner')
    };

    // Validate DOM elements
    for (const [key, element] of Object.entries(elements)) {
        if (!element) {
            console.error(`[Sentiment] Missing HTML element: ${key}`);
            elements.errorMessage.textContent = 'Erreur : Interface non initialisée correctement.';
            elements.errorMessage.classList.remove('hidden');
            return;
        }
    }

    const { text, language, result, errorMessage, label, positiveScore, positiveBar, neutralScore, neutralBar, negativeScore, negativeBar, insights, button, buttonText, loadingSpinner } = elements;

    // Clear previous state
    errorMessage.classList.add('hidden');
    result.classList.add('hidden');
    label.textContent = '';
    positiveScore.textContent = '';
    neutralScore.textContent = '';
    negativeScore.textContent = '';
    positiveBar.style.width = '0%';
    neutralBar.style.width = '0%';
    negativeBar.style.width = '0%';
    insights.innerHTML = '';

    // Validate input
    const textValue = text.value.trim();
    if (!textValue) {
        errorMessage.textContent = 'Veuillez entrer un texte à analyser.';
        errorMessage.classList.remove('hidden');
        return;
    }
    if (textValue.length > 10000) {
        errorMessage.textContent = 'Le texte est trop long (max 10 000 caractères).';
        errorMessage.classList.remove('hidden');
        return;
    }

    // Disable inputs during request
    button.disabled = true;
    button.setAttribute('aria-disabled', 'true');
    text.disabled = true;
    language.disabled = true;
    buttonText.textContent = 'Analyse en cours...';
    loadingSpinner.classList.remove('hidden');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        console.log('[Sentiment] Envoi de la requête avec le texte :', { text: textValue, language: language.value });
        const response = await fetchWithRetrySentiment('https://datacraft-815efe7282ee.herokuapp.com/api/analyze-sentiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textValue, language: language.value }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        console.log('[Sentiment] Réponse reçue :', response);

        // Handle server errors
        if (response.error || response.positive == null || response.neutral == null || response.negative == null || !response.insights) {
            const errorMap = {
                'Text is required': 'Veuillez entrer un texte à analyser.',
                'Request must be JSON': 'Erreur de format de la requête. Contactez le support.',
                'Input text is too long (max 10,000 characters)': 'Le texte est trop long (max 10 000 caractères).',
                'Input text exceeds token limit': 'Le texte dépasse la limite de tokens.',
                'Empty response from Gemini API': 'Aucune réponse du serveur d’analyse. Réessayez.',
                'Invalid JSON response from Gemini API': 'Erreur de traitement du serveur. Réessayez.',
                'Incomplete response from Gemini API': 'Réponse incomplète du serveur. Réessayez.',
                'Invalid score values from Gemini API': 'Données invalides du serveur. Réessayez.',
                'Scores do not sum to 100': 'Erreur dans les données du serveur. Réessayez.',
                'Invalid sentiment value': 'Valeur de sentiment invalide. Réessayez.',
                'Internal server error': 'Erreur serveur. Veuillez réessayer plus tard.'
            };
            errorMessage.textContent = errorMap[response.error] || `Erreur : ${response.error || 'Réponse incomplète du serveur'}`;
            errorMessage.classList.remove('hidden');
            result.classList.remove('hidden');
            return;
        }

        // Render results
        window.requestIdleCallback(() => {
            result.classList.remove('hidden');
            label.textContent = response.sentiment;
            label.className = `px-3 py-1 rounded-full text-sm font-medium ${getSentimentClass(response.sentiment)}`;

            positiveScore.textContent = `${response.positive}%`;
            positiveBar.style.width = `${response.positive}%`;
            positiveBar.setAttribute('aria-valuenow', response.positive);
            positiveBar.setAttribute('aria-valuetext', `Positive: ${response.positive}%`);

            neutralScore.textContent = `${response.neutral}%`;
            neutralBar.style.width = `${response.neutral}%`;
            neutralBar.setAttribute('aria-valuenow', response.neutral);
            neutralBar.setAttribute('aria-valuetext', `Neutral: ${response.neutral}%`);

            negativeScore.textContent = `${response.negative}%`;
            negativeBar.style.width = `${response.negative}%`;
            negativeBar.setAttribute('aria-valuenow', response.negative);
            negativeBar.setAttribute('aria-valuetext', `Negative: ${response.negative}%`);

            insights.innerHTML = `<strong>Observations clés :</strong> ${response.insights || 'Aucune observation fournie.'}`;
        }, { timeout: 1000 });

    } catch (error) {
        console.error('[Sentiment] Erreur lors de l\'analyse du sentiment :', {
            message: error.message,
            name: error.name
        });
        let errorText;
        if (error.name === 'AbortError') {
            errorText = 'La requête a expiré. Veuillez réessayer.';
        } else if (error.name === 'TypeError') {
            errorText = 'Erreur réseau : Vérifiez votre connexion ou la disponibilité du serveur.';
        } else if (error.message.includes('429')) {
            errorText = 'Trop de requêtes. Veuillez attendre et réessayer.';
        } else {
            errorText = 'Erreur lors de l\'analyse du sentiment. Veuillez réessayer.';
        }
        errorMessage.textContent = errorText;
        errorMessage.classList.remove('hidden');
        result.classList.remove('hidden');
    } finally {
        button.disabled = false;
        button.setAttribute('aria-disabled', 'false');
        text.disabled = false;
        language.disabled = false;
        buttonText.textContent = 'Analyze Sentiment';
        loadingSpinner.classList.add('hidden');
    }
}

/**
 * Determines CSS classes for sentiment label based on sentiment value.
 * @param {string} sentiment - The sentiment value ("Positive", "Negative", "Neutral").
 * @returns {string} - The CSS classes for the sentiment label.
 */
function getSentimentClass(sentiment) {
    switch (sentiment) {
        case 'Positive':
            return 'bg-success-100 text-success';
        case 'Negative':
            return 'bg-error-100 text-error';
        default:
            return 'bg-yellow-100 text-yellow-600';
    }
}

/**
 * Initializes the sentiment analysis button event listener.
 */
function initializeSentimentAnalysis() {
    const button = document.querySelector('button.btn-accent');
    if (!button) {
        console.error('[Sentiment] Button not found for sentiment analysis');
        return;
    }
    button.addEventListener('click', analyzeSentiment);
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initializeSentimentAnalysis);