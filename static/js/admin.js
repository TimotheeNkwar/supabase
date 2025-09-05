// admin.js — gestion articles pour /admin avec modal
(() => {
  const API_CONFIG = { perPage: 10, baseUrl: '/api/articles' };

  let currentPage = 1;
  let socket = null;
  let isSubmitting = false;

  class ArticleManager {
    constructor() {
      this.perPage = API_CONFIG.perPage;
      this.initSocket();
      this.bindElements();
      this.initDarkMode();
      this.attachEvents();
      this.loadCategories();
      this.loadArticles(1);
    }

    initDarkMode() {
      // Check for saved user preference, if any, on load
      if (localStorage.getItem('darkMode') === 'true' || 
          (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
      
      // Update the toggle button state
      this.updateDarkModeToggle();
    }
    
    toggleDarkMode() {
      const isDark = document.documentElement.classList.toggle('dark');
      localStorage.setItem('darkMode', isDark);
      this.updateDarkModeToggle();
    }
    
    updateDarkModeToggle() {
      const darkModeToggle = document.getElementById('dark-mode-toggle');
      if (!darkModeToggle) return;
      
      const isDark = document.documentElement.classList.contains('dark');
      const moonIcon = darkModeToggle.querySelector('.fa-moon');
      const sunIcon = darkModeToggle.querySelector('.fa-sun');
      
      if (isDark) {
        moonIcon.classList.add('hidden');
        sunIcon.classList.remove('hidden');
      } else {
        moonIcon.classList.remove('hidden');
        sunIcon.classList.add('hidden');
      }
    }

    initSocket() {
      try {
        if (typeof io === 'undefined') {
          console.warn('socket.io not available, falling back to fetch-only mode');
          return;
        }
        socket = io();
        socket.on('connect', () => console.log('Socket connected'));
        socket.on('article_updated', () => {
          console.log('Article updated event received');
          this.loadArticles(currentPage);
        });
        socket.on('article_deleted', () => {
          console.log('Article deleted event received');
          this.loadArticles(currentPage);
        });
        socket.on('article_visibility_changed', () => {
          console.log('Article visibility changed event received');
          this.loadArticles(currentPage);
        });
      } catch (e) {
        console.warn('Error initializing socket.io:', e);
        socket = null;
      }
    }

    bindElements() {
      this.tableBody = document.getElementById('articles-body');
      this.infoEl = document.getElementById('articles-info');
      this.prevBtn = document.getElementById('prev-page');
      this.nextBtn = document.getElementById('next-page');
      this.pageInfo = document.getElementById('page-info');
      this.filterCategory = document.getElementById('category-filter');
      this.filterStatus = document.getElementById('status-filter');
      this.searchInput = document.getElementById('search-input');
      this.applyFiltersBtn = document.getElementById('apply-filters');
      this.newArticleBtn = document.getElementById('new-article-btn');
      this.editModal = document.getElementById('edit-modal');
      this.modalTitle = document.getElementById('modal-title');
      this.editForm = document.getElementById('edit-form');
      this.cancelEditBtn = document.getElementById('cancel-edit');
      this.closeModalBtn = document.getElementById('close-modal');
      this.submitBtn = this.editForm?.querySelector('button[type="submit"]');
      this.fieldId = document.getElementById('edit-article-id');
      this.fieldTitle = document.getElementById('edit-title');
      this.fieldCategory = document.getElementById('edit-category');
      this.fieldDescription = document.getElementById('edit-description');
      this.fieldContent = document.getElementById('edit-content');
      this.fieldTags = document.getElementById('edit-tags');
      this.fieldImage = document.getElementById('edit-image');
      this.fieldReadTime = document.getElementById('edit-read-time');
      this.toastContainer = document.getElementById('notifications');

      // Validate DOM elements
      const requiredElements = [
        this.tableBody, this.infoEl, this.prevBtn, this.nextBtn, this.pageInfo,
        this.filterCategory, this.filterStatus, this.searchInput, this.applyFiltersBtn,
        this.newArticleBtn, this.editModal, this.modalTitle, this.editForm,
        this.cancelEditBtn, this.closeModalBtn, this.submitBtn, this.fieldId,
        this.fieldTitle, this.fieldCategory, this.fieldDescription, this.fieldContent,
        this.fieldTags, this.fieldImage, this.fieldReadTime, this.toastContainer
      ];
      requiredElements.forEach((el, index) => {
        if (!el) console.error(`Missing DOM element at index ${index}`);
      });
    }

    attachEvents() {
      // Dark mode toggle
      const darkModeToggle = document.getElementById('dark-mode-toggle');
      if (darkModeToggle) {
        darkModeToggle.addEventListener('click', () => this.toggleDarkMode());
      }
      
      // Pagination
      if (this.prevBtn) this.prevBtn.addEventListener('click', () => {
        if (currentPage > 1) this.loadArticles(currentPage - 1);
      });
      if (this.nextBtn) this.nextBtn.addEventListener('click', () => {
        this.loadArticles(currentPage + 1);
      });
      if (this.applyFiltersBtn) this.applyFiltersBtn.addEventListener('click', () => this.loadArticles(1));
      if (this.newArticleBtn) this.newArticleBtn.addEventListener('click', () => this.openEditModal({}));
      if (this.cancelEditBtn) this.cancelEditBtn.addEventListener('click', () => this.closeEditModal());
      if (this.closeModalBtn) this.closeModalBtn.addEventListener('click', () => this.closeEditModal());
      if (this.editForm) this.editForm.addEventListener('submit', (e) => {
        e.preventDefault();
        this.handleFormSubmit();
      });

      // Debounced search input
      let debounceTimer;
      if (this.searchInput) this.searchInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => this.loadArticles(1), 300);
      });
    }

    async loadCategories() {
      try {
        const res = await fetch('/api/categories', {
          headers: { 'Accept': 'application/json' },
          credentials: 'include'
        });
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`Erreur HTTP ${res.status}: ${errorText}`);
        }
        const categories = await res.json();
        if (this.filterCategory) {
          this.filterCategory.innerHTML = '<option value="">All Categories</option>';
          categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category.charAt(0).toUpperCase() + category.slice(1);
            this.filterCategory.appendChild(option);
          });
        }
      } catch (e) {
        console.error('Error loading categories:', e);
        this.showToast('Impossible de charger les catégories', { type: 'error' });
      }
    }

    async loadArticles(page = 1) {
      const category = this.filterCategory?.value || '';
      const status = this.filterStatus?.value || 'all';
      const search = this.searchInput?.value || '';

      const qs = new URLSearchParams({
        page,
        per_page: this.perPage,
        category,
        status,
        search
      });

      try {
        this.showToast('Loading articles...', { autoHide: true, timeout: 800 });
        const res = await fetch(`${API_CONFIG.baseUrl}?${qs.toString()}`, {
          headers: {
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': 'application/json'
          },
          credentials: 'include'
        });
        if (res.status === 401) {
          this.showToast('Veuillez vous connecter', { type: 'error' });
          window.location.href = '/login';
          return;
        }
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`Erreur HTTP ${res.status}: ${errorText}`);
        }
        const data = await res.json();
        this.renderArticles(data.articles || []);
        this.updatePagination(data, page);
        currentPage = page;
        this.showToast('Loaded items', { type: 'success', autoHide: true, timeout: 1000 });
      } catch (err) {
        console.error('Error loading articles:', err);
        this.showToast(`Unable to load items: ${err.message}`, { type: 'error' });
      }
    }

    renderArticles(list) {
      if (!this.tableBody) {
        console.error('Table body element missing');
        return;
      }
      this.tableBody.innerHTML = '';
      if (!Array.isArray(list) || list.length === 0) {
        this.tableBody.innerHTML = `
          <tr><td colspan="4" class="text-center py-4 text-muted">Aucun article trouvé</td></tr>
        `;
        return;
      }

      list.forEach(article => {
        const tr = document.createElement('tr');
        tr.className = 'border-b hover:bg-gray-50';
        tr.innerHTML = `
          <td class="p-3">${this.escapeHtml(article.title || '—')}</td>
          <td class="p-3">${this.escapeHtml(article.category || '—')}</td>
          <td class="p-3">
            <span class="px-2 py-1 rounded-full text-sm ${article.hidden ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}">
              ${article.hidden ? 'Hide' : 'Visible'}
            </span>
          </td>
          <td class="p-3">
            <div class="flex gap-2">
              <button class="btn btn-sm btn-primary" data-action="edit" data-id="${this.escapeHtml(article.id)}">Edit</button>
              <button class="btn btn-sm btn-outline" data-action="toggle" data-id="${this.escapeHtml(article.id)}" data-hidden="${!!article.hidden}">
                ${article.hidden ? 'Show' : 'Hide'}
              </button>
              <button class="btn btn-sm btn-danger" data-action="delete" data-id="${this.escapeHtml(article.id)}">Delete</button>
            </div>
          </td>
        `;
        tr.querySelectorAll('button[data-action]').forEach(btn => {
          btn.addEventListener('click', (e) => {
            const action = btn.getAttribute('data-action');
            const id = btn.getAttribute('data-id');
            if (action === 'edit') this.editArticle(id);
            if (action === 'toggle') this.toggleVisibility(id, btn.getAttribute('data-hidden') === 'true');
            if (action === 'delete') this.deleteArticle(id);
          });
        });
        this.tableBody.appendChild(tr);
      });
    }

    updatePagination(data, page) {
      if (!this.infoEl || !this.pageInfo || !this.prevBtn || !this.nextBtn) {
        console.error('Pagination elements missing');
        return;
      }
      const total = Number(data.total || 0);
      const pages = Number(data.pages || 1);
      const start = total === 0 ? 0 : (page - 1) * this.perPage + 1;
      const end = Math.min(page * this.perPage, total);

      this.infoEl.textContent = `Affichage de ${start} à ${end} sur ${total} articles`;
      this.prevBtn.disabled = page <= 1;
      this.nextBtn.disabled = page >= pages;
      this.pageInfo.textContent = `Page ${page} sur ${pages}`;
    }

    async editArticle(id) {
      if (!id) {
        this.showToast('Identifiant d\'article manquant', { type: 'error' });
        return;
      }
      try {
        this.showToast('Chargement de l\'article...', { autoHide: true, timeout: 800 });
        const res = await fetch(`${API_CONFIG.baseUrl}/${id}`, {
          headers: { 'Accept': 'application/json' },
          credentials: 'include'
        });
        if (res.status === 401) {
          this.showToast('Veuillez vous connecter', { type: 'error' });
          window.location.href = '/login';
          return;
        }
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`Erreur HTTP ${res.status}: ${errorText}`);
        }
        const article = await res.json();
        this.openEditModal(article);
        this.showToast('Article chargé', { type: 'success', autoHide: true, timeout: 1000 });
      } catch (e) {
        console.error('Error loading article:', e);
        this.showToast(`Impossible de charger l'article: ${e.message}`, { type: 'error' });
      }
    }

    openEditModal(article = {}) {
      if (!this.modalTitle || !this.fieldId || !this.editModal || !this.submitBtn) {
        console.error('Edit modal elements missing');
        this.showToast('Erreur: éléments du formulaire manquants', { type: 'error' });
        return;
      }
      this.modalTitle.textContent = article.id ? "Modifier l'article" : "Nouvel article";
      this.fieldId.value = article.id || '';
      this.fieldTitle.value = article.title || '';
      this.fieldCategory.value = article.category || 'technologie';
      this.fieldDescription.value = article.description || '';
      this.fieldContent.value = article.content || '';
      this.fieldTags.value = Array.isArray(article.tags) ? article.tags.join(', ') : (article.tags || '');
      this.fieldImage.value = article.image || '';
      this.fieldReadTime.value = article.read_time || 5;
      this.editModal.classList.remove('hidden');
      this.submitBtn.disabled = false;
      this.submitBtn.textContent = article.id ? 'Mettre à jour' : 'Créer';
    }

    closeEditModal() {
      if (!this.editModal || !this.editForm) {
        console.error('Edit modal or form missing');
        return;
      }
      this.editModal.classList.add('hidden');
      try {
        this.editForm.reset();
        if (this.submitBtn) this.submitBtn.disabled = false;
      } catch (e) {
        console.warn('Error resetting form:', e);
      }
    }

    async handleFormSubmit() {
      if (!this.fieldId || !this.fieldTitle || !this.submitBtn) {
        console.error('Form field ID or title missing');
        this.showToast('Error: Missing form elements', { type: 'error' });
        return;
      }
      if (isSubmitting) {
        this.showToast('Submission in progress, please wait', { type: 'info' });
        return;
      }

      const id = this.fieldId.value;
      const payload = {
        title: this.fieldTitle.value?.trim() || '',
        category: this.fieldCategory.value?.trim() || 'technology',
        description: this.fieldDescription.value || '',
        content: this.fieldContent.value || '',
        tags: (this.fieldTags.value || '').split(',').map(t => t.trim()).filter(Boolean),
        image: this.fieldImage.value || '',
        read_time: parseInt(this.fieldReadTime.value || '5', 10)
      };

      if (!payload.title) {
        this.showToast('Title is required', { type: 'error' });
        return;
      }
      if (payload.read_time <= 0) {
        this.showToast('Reading time should be positive', { type: 'error' });
        return;
      }

      isSubmitting = true;
      this.submitBtn.disabled = true;
      this.submitBtn.textContent = id ? 'Update...' : 'Creation...';

      try {
        const url = id ? `${API_CONFIG.baseUrl}/${id}` : API_CONFIG.baseUrl;
        const method = id ? 'PUT' : 'POST';
        this.showToast(id ? 'Updated the article...' : 'Creation of the article...', { autoHide: true, timeout: 800 });
        const res = await fetch(url, {
          method,
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          body: JSON.stringify(payload),
          credentials: 'include'
        });
        if (res.status === 401) {
          this.showToast('Please log in', { type: 'error' });
          window.location.href = '/login';
          return;
        }
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`Erreur HTTP ${res.status}: ${errorText}`);
        }
        const result = await res.json();
        if (socket) socket.emit('article_updated', { article: result.article || result });
        this.closeEditModal();
        this.showToast(id ? 'Article updated' : 'Article created', { type: 'success', autoHide: true, timeout: 1000 });
        await this.loadArticles(currentPage);
      } catch (e) {
        console.error('Error saving article:', e);
        this.showToast(`Backup failed: ${e.message}`, { type: 'error' });
      } finally {
        isSubmitting = false;
        if (this.submitBtn) {
          this.submitBtn.disabled = false;
          this.submitBtn.textContent = id ? 'Mettre à jour' : 'Créer';
        }
      }
    }

    async toggleVisibility(articleId, currentlyHidden) {
      if (!articleId) {
        this.showToast('Missing item ID', { type: 'error' });
        return;
      }
      try {
        this.showToast('Visibility update...', { autoHide: true, timeout: 800 });
        const res = await fetch(`${API_CONFIG.baseUrl}/${articleId}/toggle-visibility`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          body: JSON.stringify({ hidden: !currentlyHidden }),
          credentials: 'include'
        });
        if (res.status === 401) {
          this.showToast('Please log in', { type: 'error' });
          window.location.href = '/login';
          return;
        }
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`Erreur HTTP ${res.status}: ${errorText}`);
        }
        if (socket) socket.emit('article_visibility_changed', { articleId });
        this.showToast('Visibility updated', { type: 'success', autoHide: true, timeout: 1000 });
        await this.loadArticles(currentPage);
      } catch (e) {
        console.error('Error toggling visibility:', e);
        this.showToast(`Impossible de changer la visibilité: ${e.message}`, { type: 'error' });
      }
    }

    async deleteArticle(articleId) {
      if (!articleId) {
        this.showToast('Identifiant d\'article manquant', { type: 'error' });
        return;
      }
      if (!confirm('Êtes-vous sûr de vouloir supprimer cet article ?')) return;
      try {
        this.showToast('Suppression de l\'article...', { autoHide: true, timeout: 800 });
        const res = await fetch(`${API_CONFIG.baseUrl}/${articleId}`, {
          method: 'DELETE',
          headers: { 'Accept': 'application/json' },
          credentials: 'include'
        });
        if (res.status === 401) {
          this.showToast('Veuillez vous connecter', { type: 'error' });
          window.location.href = '/login';
          return;
        }
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`Erreur HTTP ${res.status}: ${errorText}`);
        }
        if (socket) socket.emit('article_deleted', { articleId });
        this.showToast('Article supprimé', { type: 'success', autoHide: true, timeout: 1000 });
        await this.loadArticles(currentPage);
      } catch (e) {
        console.error('Error deleting article:', e);
        this.showToast(`Impossible de supprimer l'article: ${e.message}`, { type: 'error' });
      }
    }

    escapeHtml(unsafe = '') {
      return String(unsafe)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    showToast(message, opts = {}) {
      if (!this.toastContainer) {
        console.error('Toast container missing');
        return;
      }
      const { type = 'info', timeout = 3000, autoHide = false } = opts;
      const div = document.createElement('div');
      div.className = `toast ${type === 'error' ? 'bg-red-100 text-red-800' : type === 'success' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'} p-3 rounded-lg shadow-md`;
      div.textContent = message;
      this.toastContainer.appendChild(div);
      if (autoHide || timeout) setTimeout(() => div.remove(), timeout);
    }
  }

  // Instantiate globally
  try {
    window.articleManager = new ArticleManager();
  } catch (e) {
    console.error('Failed to initialize ArticleManager:', e);
    document.body.insertAdjacentHTML('beforeend', '<div class="bg-red-100 text-red-800 p-4 rounded-lg">Erreur d\'initialisation, veuillez vérifier la console.</div>');
  }
})();