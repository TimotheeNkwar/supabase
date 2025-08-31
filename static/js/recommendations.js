


async function fetchWithRetry(url, options, retries = 5, delay = 2000) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            if (response.status === 429) {
                throw new Error('Rate limit exceeded');
            }
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            if (i === retries - 1) {
                throw error;
            }
            const backoffDelay = delay * Math.pow(2, i);
            await new Promise(resolve => setTimeout(resolve, backoffDelay));
        }
    }
}

/**
 * Generates movie recommendations based on user input and displays them.
 * @param {Event} event - The form submission event.
 */
async function generateRecommendations(event) {
    event.preventDefault();

    // Get DOM elements
    const elements = {
        promptText: document.getElementById('prompt-text'),
        budget: document.getElementById('budget-range'),
        recommendationsList: document.getElementById('rec-list'),
        recommendationsResult: document.getElementById('rec-result'),
        errorMessage: document.getElementById('rec-error'),
        button: document.querySelector('button.btn-recommend'),
        buttonText: document.getElementById('rec-button-text'),
        loadingSpinner: document.getElementById('rec-loading-spinner')
    };

    // Validate DOM elements
    for (const [key, element] of Object.entries(elements)) {
        if (!element) {
            console.error(`Missing HTML element: ${key}`);
            elements.errorMessage.textContent = 'Erreur : Interface non initialisée correctement.';
            elements.errorMessage.classList.remove('hidden');
            return;
        }
    }

    const { promptText, budget, recommendationsList, recommendationsResult, errorMessage, button, buttonText, loadingSpinner } = elements;

    // Clear previous state
    errorMessage.classList.add('hidden');
    recommendationsResult.classList.add('hidden');
    recommendationsList.innerHTML = '';

    // Validate input
    const promptValue = promptText.value.trim();
    if (!promptValue) {
        errorMessage.textContent = 'Veuillez entrer une description de style ou de genre de film.';
        errorMessage.classList.remove('hidden');
        return;
    }
    if (promptValue.length > 10000) {
        errorMessage.textContent = 'La description est trop longue (max 10 000 caractères).';
        errorMessage.classList.remove('hidden');
        return;
    }
    if (!['budget', 'mid-range', 'premium'].includes(budget.value)) {
        errorMessage.textContent = 'Veuillez sélectionner une gamme de budget valide.';
        errorMessage.classList.remove('hidden');
        return;
    }

    // Disable inputs during request
    button.disabled = true;
    button.setAttribute('aria-disabled', 'true');
    promptText.disabled = true;
    budget.disabled = true;
    buttonText.textContent = 'Génération en cours...';
    loadingSpinner.classList.remove('hidden');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        console.log('Envoi de la requête avec le prompt :', { prompt: promptValue, budget: budget.value });
        const response = await fetchWithRetry('https://datacraft-815efe7282ee.herokuapp.com/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: promptValue, budget: budget.value }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        console.log('Réponse reçue :', response);

        // Handle server errors
        if (response.error || !response.recommendations) {
            const errorMap = {
                'Prompt is required': 'Veuillez entrer une description de style ou de genre de film.',
                'Request must be JSON': 'Erreur de format de la requête. Contactez le support.',
                'Input prompt is too long (max 10,000 characters)': 'La description est trop longue (max 10 000 caractères).',
                'Invalid budget range': 'Gamme de budget invalide. Veuillez sélectionner une option valide.',
                'No recommendations found': 'Aucun film trouvé pour cette description.',
                'Internal server error': 'Erreur serveur. Veuillez réessayer plus tard.'
            };
            errorMessage.textContent = errorMap[response.error] || `Erreur : ${response.error || 'Réponse incomplète du serveur'}`;
            errorMessage.classList.remove('hidden');
            recommendationsResult.classList.remove('hidden');
            return;
        }

        // Render recommendations
        window.requestIdleCallback(() => {
            recommendationsList.innerHTML = '';
            if (response.recommendations.length === 0) {
                recommendationsList.innerHTML = '<p class="text-text-secondary">Aucun film trouvé pour cette description.</p>';
            } else {
                response.recommendations.forEach(movie => {
                    const movieItem = document.createElement('div');
                    movieItem.className = 'text-text-secondary border-b border-surface-200 pb-2';
                    movieItem.innerHTML = `
                        <strong class="text-gray-800">${movie.name}</strong><br>
                        <span class="text-sm">Genres: ${movie.genres.join(', ')}</span><br>
                        <span class="text-sm font-medium">Prix: $${movie.price}M</span>
                    `;
                    recommendationsList.appendChild(movieItem);
                });
            }
            recommendationsResult.classList.remove('hidden');
        }, { timeout: 1000 });

    } catch (error) {
        console.error('Erreur lors de la génération des recommandations :', {
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
            errorText = 'Erreur lors de la génération des recommandations. Veuillez réessayer.';
        }
        errorMessage.textContent = errorText;
        errorMessage.classList.remove('hidden');
        recommendationsResult.classList.remove('hidden');
    } finally {
        // Re-enable inputs
        button.disabled = false;
        button.setAttribute('aria-disabled', 'false');
        promptText.disabled = false;
        budget.disabled = false;
        buttonText.textContent = 'Générer les recommandations';
        loadingSpinner.classList.add('hidden');
    }
}

/**
 * Initializes the recommendation form event listener.
 */
function initializeForm() {
    const form = document.getElementById('recommendation-form');
    if (!form) {
        console.error('Recommendation form not found');
        return;
    }
    form.addEventListener('submit', generateRecommendations);
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initializeForm);
