// script.js - OPTIMIZAT PENTRU CĂUTARE ÎMBUNĂTĂȚITĂ - v3.0.0
// =================================================================
// Acest script a fost consolidat din 5 fișiere separate.
// Am eliminat duplicatele și am reordonat funcțiile logic
// pentru a asigura funcționarea corectă a aplicației.
// =================================================================

// -----------------------------------------------------------------
// SECȚIUNEA 1: Configurare și Variabile Globale
// -----------------------------------------------------------------

// Configurare API - Adresa serverului backend
const API_BASE_URL = 'http://localhost:8070';

// Variabile globale care stochează starea aplicației
let currentCollection = null; // Colecția selectată curent
let documentsList = []; // Lista de documente din colecția curentă
let searchHistory = []; // Istoricul căutărilor efectuate
let queryAnalytics = { // Statistici despre interogări
    totalQueries: 0,
    averageResponseTime: 0,
    successRate: 0
};

// Cache pentru rezultatele căutărilor, pentru a oferi răspunsuri rapide
const resultCache = new Map();
const MAX_CACHE_SIZE = 100; // Numărul maxim de rezultate stocate în cache
const CACHE_TTL = 10 * 60 * 1000; // Timpul de viață al cache-ului (10 minute)

// Configurări optimizate pentru performanță și comportament
const CONFIG = {
    DEBOUNCE_DELAY: 300, // Întârziere pentru funcții de debouncing (ex: căutare live)
    MAX_RETRIES: 3, // Numărul maxim de reîncercări pentru cererile API eșuate
    RETRY_DELAY: 1000, // Timpul de așteptare inițial înainte de reîncercare
    DEFAULT_TOP_K: 10, // Numărul implicit de documente relevante returnate
    DEFAULT_TEMPERATURE: 0.3, // "Creativitatea" răspunsului AI (valori mici = mai precis)
    DEFAULT_SIMILARITY_THRESHOLD: 0.15, // Pragul de similaritate pentru căutare
    ENABLE_HYBRID_SEARCH: true, // Activează/dezactivează căutarea hibridă (semantică + keyword)
    AUTO_SAVE_INTERVAL: 30000, // Intervalul pentru salvarea automată a stării (30 secunde)
    PERFORMANCE_MONITORING: true // Activează/dezactivează logarea metricilor de performanță
};


// -----------------------------------------------------------------
// SECȚIUNEA 2: Funcții Utilitare Generale
// -----------------------------------------------------------------

/**
 * Funcție de "debouncing". Împiedică executarea repetată a unei funcții
 * la fiecare eveniment, așteptând un interval de timp de la ultimul apel.
 * Util pentru input-uri de căutare, pentru a nu trimite cereri la fiecare literă tastată.
 * @param {Function} func - Funcția de executat.
 * @param {number} wait - Timpul de așteptare în milisecunde.
 * @param {boolean} immediate - Dacă funcția trebuie executată imediat la primul apel.
 */
function debounce(func, wait, immediate = false) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

/**
 * Sistem de reîncercare automată pentru funcții asincrone (ex: cereri API).
 * Utilizează un "backoff exponențial" - timpul de așteptare crește după fiecare eșec.
 * @param {Function} fn - Funcția asincronă de executat.
 * @param {number} maxRetries - Numărul maxim de reîncercări.
 */
async function withRetry(fn, maxRetries = CONFIG.MAX_RETRIES) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            if (attempt === maxRetries) break;
            
            const delay = CONFIG.RETRY_DELAY * Math.pow(2, attempt - 1);
            console.warn(`Attempt ${attempt} failed, retrying in ${delay}ms:`, error.message);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    
    throw lastError;
}

/**
 * Afișează un indicator de încărcare (spinner) peste un element specificat.
 * @param {string} selector - Selectorul CSS pentru element (ex: '.sidebar').
 * @param {string} message - Mesajul afișat sub spinner.
 */
function showLoading(selector, message = 'Se procesează...') {
    const element = document.querySelector(selector);
    if (!element) return;
    
    element.classList.add('loading');
    
    let overlay = element.querySelector('.loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        element.appendChild(overlay);
    }
    
    overlay.innerHTML = `
        <div class="spinner-container">
            <div class="spinner"></div>
            <div class="loading-text">${message}</div>
            <div class="loading-progress">
                <div class="progress-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    
    overlay.style.display = 'flex';
}

/**
 * Ascunde indicatorul de încărcare de pe un element.
 * @param {string} selector - Selectorul CSS pentru element.
 */
function hideLoading(selector) {
    const element = document.querySelector(selector);
    if (!element) return;
    
    element.classList.remove('loading');
    const overlay = element.querySelector('.loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

/**
 * Funcție de siguranță pentru a "curăța" textul înainte de a-l insera în HTML.
 * Previne atacurile de tip Cross-Site Scripting (XSS).
 * @param {string} text - Textul de curățat.
 * @returns {string} Textul curățat.
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Funcție pentru a "curăța" un string pentru a fi folosit în expresii regulate.
 * @param {string} string - Textul de curățat.
 * @returns {string} Textul curățat.
 */
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Formatează dimensiunea unui fișier din bytes în unități mai lizibile (KB, MB, GB).
 * @param {number} bytes - Dimensiunea în bytes.
 * @returns {string} Dimensiunea formatată.
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}


// -----------------------------------------------------------------
// SECȚIUNEA 3: Sisteme Avansate (Notificări, Cache, Analiză Query)
// -----------------------------------------------------------------

/**
 * Clasă pentru un sistem de notificări avansat.
 * Gestionează afișarea, coada de așteptare, tipurile și acțiunile notificărilor.
 */
class AdvancedNotificationSystem {
    constructor() {
        this.container = this.createContainer();
        this.notifications = new Map();
        this.queue = [];
        this.maxVisible = 5;
        this.analytics = { shown: 0, dismissed: 0, clicked: 0 };
    }

    createContainer() {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.className = 'notification-container';
        document.body.appendChild(container);
        return container;
    }

    show(message, type = 'success', options = {}) {
        const notification = this.createNotification(message, type, options);
        const id = this.generateId();
        
        this.notifications.set(id, notification);
        this.analytics.shown++;
        
        if (this.notifications.size > this.maxVisible) {
            this.queue.push({ message, type, options });
            return id;
        }
        
        this.container.appendChild(notification.element);
        
        requestAnimationFrame(() => {
            notification.element.classList.add('show');
        });

        if (options.duration !== false) {
            const duration = options.duration || this.getDefaultDuration(type);
            setTimeout(() => this.remove(id), duration);
        }

        return id;
    }

    createNotification(message, type, options) {
        const element = document.createElement('div');
        element.className = `notification notification-${type}`;
        if (options.important) element.classList.add('notification-important');
        
        const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️', loading: '⏳', search: '🔍', upload: '📤', delete: '🗑️' };
        const icon = icons[type] || icons.info;
        
        let actionsHTML = '';
        if (options.actions) {
            actionsHTML = `<div class="notification-actions">${options.actions.map(action => `<button class="notification-action" data-action="${action.id}">${action.label}</button>`).join('')}</div>`;
        }
        
        element.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">${icon}</div>
                <div class="notification-message">
                    <div class="notification-title">${options.title || type.charAt(0).toUpperCase() + type.slice(1)}</div>
                    <div class="notification-text">${message}</div>
                </div>
                ${actionsHTML}
                <button class="notification-close" aria-label="Close">×</button>
            </div>
            ${options.progress ? '<div class="notification-progress"><div class="progress-bar"></div></div>' : ''}
        `;

        if (options.actions) {
            element.querySelectorAll('.notification-action').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const actionId = e.target.getAttribute('data-action');
                    const action = options.actions.find(a => a.id === actionId);
                    if (action && action.callback) {
                        action.callback();
                        this.analytics.clicked++;
                    }
                });
            });
        }

        element.querySelector('.notification-close').addEventListener('click', () => this.removeByElement(element));
        return { element, type, options };
    }

    updateProgress(id, progress) {
        const notification = this.notifications.get(id);
        if (notification) {
            const progressBar = notification.element.querySelector('.progress-bar');
            if (progressBar) progressBar.style.width = `${progress}%`;
        }
    }

    remove(id) {
        const notification = this.notifications.get(id);
        if (notification) this.removeByElement(notification.element);
    }

    removeByElement(element) {
        element.classList.add('hide');
        this.analytics.dismissed++;
        
        setTimeout(() => {
            if (element.parentNode) element.parentNode.removeChild(element);
            for (const [id, notification] of this.notifications.entries()) {
                if (notification.element === element) {
                    this.notifications.delete(id);
                    break;
                }
            }
            this.processQueue();
        }, 300);
    }

    processQueue() {
        if (this.queue.length > 0 && this.notifications.size < this.maxVisible) {
            const next = this.queue.shift();
            this.show(next.message, next.type, next.options);
        }
    }

    getDefaultDuration(type) {
        const durations = { success: 4000, error: 8000, warning: 6000, info: 5000, loading: false, search: 3000 };
        return durations[type] || 5000;
    }

    generateId() { return Date.now() + Math.random().toString(36).substr(2, 9); }
    clear() { this.notifications.forEach((_, id) => this.remove(id)); }
    getAnalytics() { return { ...this.analytics }; }
}
const notifications = new AdvancedNotificationSystem();

/**
 * Wrapper pentru funcția `fetch` a browser-ului.
 * Adaugă automat gestionarea erorilor, timeout-uri, headere custom și notificări.
 * @param {string} url - URL-ul către care se face cererea.
 * @param {object} options - Opțiunile pentru `fetch`.
 */
async function fetchWithErrorHandling(url, options = {}) {
    const startTime = performance.now();
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        const requestOptions = { ...options };
        if (!(options.body instanceof FormData)) {
             requestOptions.headers = { 'Content-Type': 'application/json', ...options.headers };
        }
        requestOptions.headers = { 'X-Request-ID': generateRequestId(), 'X-Client-Version': '3.0.0', ...requestOptions.headers };

        const response = await fetch(url, { ...requestOptions, signal: controller.signal });
        clearTimeout(timeoutId);
        
        const endTime = performance.now();
        const duration = endTime - startTime;
        
        if (CONFIG.PERFORMANCE_MONITORING) console.log(`API Call: ${url} - ${duration.toFixed(2)}ms`);

        if (!response.ok) {
            let errorMessage = 'A apărut o eroare în comunicația cu serverul.';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                errorMessage = `Eroare ${response.status}: ${response.statusText}`;
            }
            notifications.show(errorMessage, 'error', { title: `Eroare ${response.status}`, duration: 8000, actions: response.status >= 500 ? [{ id: 'retry', label: 'Reîncearcă', callback: () => fetchWithErrorHandling(url, options) }] : null });
            throw new Error(errorMessage);
        }

        queryAnalytics.averageResponseTime = (queryAnalytics.averageResponseTime * queryAnalytics.totalQueries + duration) / (queryAnalytics.totalQueries + 1);
        return await response.json();
        
    } catch (error) {
        console.error('Eroare de rețea:', error);
        if (error.name === 'AbortError') {
            notifications.show('Cererea a expirat. Verificați conexiunea la internet.', 'warning', { title: 'Timeout', actions: [{ id: 'retry', label: 'Reîncearcă', callback: () => fetchWithErrorHandling(url, options) }] });
            throw new Error('Cererea a expirat. Încercați din nou.');
        }
        if (!navigator.onLine) {
            notifications.show('Nu există conexiune la internet.', 'error', { title: 'Conexiune', duration: false });
        }
        throw error;
    }
}
function generateRequestId() { return 'req_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9); }

/**
 * Clasă pentru gestionarea avansată a cache-ului.
 * Implementează un cache cu limită de dimensiune, TTL (Time To Live),
 * și o strategie de eliminare LRU (Least Recently Used).
 */
class AdvancedCacheManager {
    constructor(maxSize = MAX_CACHE_SIZE, ttl = CACHE_TTL) {
        this.cache = new Map();
        this.accessTimes = new Map();
        this.maxSize = maxSize;
        this.ttl = ttl;
        this.hits = 0;
        this.misses = 0;
        setInterval(() => this.cleanup(), 60000);
    }

    set(key, value) {
        if (this.cache.size >= this.maxSize) this.evictLRU();
        const item = { value: this.compress(value), timestamp: Date.now(), accessCount: 0, size: JSON.stringify(value).length };
        this.cache.set(key, item);
        this.accessTimes.set(key, Date.now());
    }

    get(key) {
        const item = this.cache.get(key);
        if (!item) { this.misses++; return null; }
        if (Date.now() - item.timestamp > this.ttl) { this.cache.delete(key); this.accessTimes.delete(key); this.misses++; return null; }
        item.accessCount++;
        this.accessTimes.set(key, Date.now());
        this.hits++;
        return this.decompress(item.value);
    }

    compress(data) { return JSON.stringify(data); }
    decompress(data) { return JSON.parse(data); }

    evictLRU() {
        let oldestTime = Date.now(), oldestKey = null;
        for (const [key, time] of this.accessTimes.entries()) {
            if (time < oldestTime) { oldestTime = time; oldestKey = key; }
        }
        if (oldestKey) { this.cache.delete(oldestKey); this.accessTimes.delete(oldestKey); }
    }

    cleanup() {
        const now = Date.now(), expiredKeys = [];
        for (const [key, item] of this.cache.entries()) {
            if (now - item.timestamp > this.ttl) expiredKeys.push(key);
        }
        expiredKeys.forEach(key => { this.cache.delete(key); this.accessTimes.delete(key); });
        if (expiredKeys.length > 0) console.log(`Cache cleanup: removed ${expiredKeys.length} expired items`);
    }

    getStats() {
        const totalRequests = this.hits + this.misses;
        const hitRate = totalRequests > 0 ? (this.hits / totalRequests) * 100 : 0;
        const totalSize = Array.from(this.cache.values()).reduce((sum, item) => sum + item.size, 0);
        return { size: this.cache.size, maxSize: this.maxSize, hitRate: hitRate.toFixed(2) + '%', hits: this.hits, misses: this.misses, totalSize, averageItemSize: this.cache.size > 0 ? Math.round(totalSize / this.cache.size) : 0 };
    }

    clear() { this.cache.clear(); this.accessTimes.clear(); this.hits = 0; this.misses = 0; }
}
const cache = new AdvancedCacheManager();

/**
 * Clasă pentru analiza interogărilor (query-urilor).
 * Detectează limba, complexitatea, intenția și oferă sugestii
 * și optimizări automate pentru parametrii de căutare.
 */
class QueryAnalyzer {
    constructor() {
        this.patterns = new Map();
        this.suggestions = new Map();
    }

    analyzeQuery(query) {
        const analysis = { complexity: this.calculateComplexity(query), language: this.detectLanguage(query), intent: this.detectIntent(query), suggestions: this.getSuggestions(query), optimizations: this.getOptimizations(query) };
        this.learnPattern(query, analysis);
        return analysis;
    }

    calculateComplexity(query) {
        const words = query.split(/\s+/), wordCount = words.length, avgWordLength = words.reduce((sum, word) => sum + word.length, 0) / wordCount;
        const questionWords = ['cum', 'ce', 'de ce', 'când', 'unde', 'cine', 'care', 'how', 'what', 'why', 'when', 'where', 'who', 'which'];
        const hasQuestionWords = questionWords.some(word => query.toLowerCase().includes(word));
        let score = 0;
        if (wordCount > 10) score += 2;
        if (avgWordLength > 6) score += 1;
        if (hasQuestionWords) score += 1;
        if (query.includes('?')) score += 1;
        if (score >= 4) return 'high';
        if (score >= 2) return 'medium';
        return 'low';
    }

    detectLanguage(query) {
        const romanianWords = ['cum', 'este', 'sunt', 'pentru', 'acest', 'această', 'și', 'în', 'de', 'la'], englishWords = ['how', 'is', 'are', 'for', 'this', 'and', 'in', 'of', 'to', 'the'];
        const roCount = romanianWords.filter(word => query.toLowerCase().includes(word)).length, enCount = englishWords.filter(word => query.toLowerCase().includes(word)).length;
        if (roCount > enCount) return 'romanian';
        if (enCount > roCount) return 'english';
        return 'mixed';
    }

    detectIntent(query) {
        const patterns = { howTo: /\b(cum să|cum pot|how to|how can)\b/i, definition: /\b(ce este|what is|define|definition)\b/i, comparison: /\b(diferența|difference|compare|vs|versus)\b/i, list: /\b(listă|list|enumerate|care sunt)\b/i, troubleshooting: /\b(problemă|eroare|nu funcționează|error|problem|fix)\b/i };
        for (const [intent, pattern] of Object.entries(patterns)) { if (pattern.test(query)) return intent; }
        return 'general';
    }
    
    getSuggestions(query) {
        const suggestions = [], words = query.toLowerCase().split(/\s+/);
        if (words.length < 3) suggestions.push('Încercați să adăugați mai multe cuvinte pentru rezultate mai precise');
        if (!query.includes('?') && this.detectIntent(query) !== 'general') suggestions.push('Reformulați ca întrebare pentru rezultate mai bune');
        const technicalTerms = ['api', 'configuration', 'setup', 'install'], hasTechnical = technicalTerms.some(term => words.includes(term));
        if (hasTechnical) suggestions.push('Pentru întrebări tehnice, includeți contextul specific (versiune, sistem de operare, etc.)');
        return suggestions;
    }

    getOptimizations(query) {
        const analysis = { recommendedTopK: CONFIG.DEFAULT_TOP_K, recommendedTemperature: CONFIG.DEFAULT_TEMPERATURE, recommendedThreshold: CONFIG.DEFAULT_SIMILARITY_THRESHOLD, enableHybridSearch: CONFIG.ENABLE_HYBRID_SEARCH };
        const complexity = this.calculateComplexity(query);
        if (complexity === 'high') { analysis.recommendedTopK = 15; analysis.recommendedTemperature = 0.4; analysis.recommendedThreshold = 0.1; }
        else if (complexity === 'low') { analysis.recommendedTopK = 8; analysis.recommendedTemperature = 0.2; analysis.recommendedThreshold = 0.2; }
        return analysis;
    }

    learnPattern(query, analysis) {
        const pattern = `${analysis.complexity}_${analysis.language}_${analysis.intent}`;
        if (!this.patterns.has(pattern)) this.patterns.set(pattern, []);
        this.patterns.get(pattern).push({ query: query.substring(0, 50), timestamp: Date.now(), analysis });
        const patterns = this.patterns.get(pattern);
        if (patterns.length > 100) patterns.splice(0, patterns.length - 100);
    }
    
    getInsights() {
        const insights = { totalQueries: 0, complexityDistribution: { low: 0, medium: 0, high: 0 }, languageDistribution: { romanian: 0, english: 0, mixed: 0 }, intentDistribution: {}, topPatterns: [] };
        
        for (const [pattern, queries] of this.patterns.entries()) {
            insights.totalQueries += queries.length;
            const [complexity, language, intent] = pattern.split('_');
            insights.complexityDistribution[complexity]++;
            insights.languageDistribution[language]++;
            insights.intentDistribution[intent] = (insights.intentDistribution[intent] || 0) + 1;
        }
        
        const sortedPatterns = Array.from(this.patterns.entries()).sort((a, b) => b[1].length - a[1].length).slice(0, 5);
        insights.topPatterns = sortedPatterns.map(([pattern, queries]) => ({ pattern, count: queries.length, lastUsed: new Date(Math.max(...queries.map(q => q.timestamp))) }));
        
        return insights;
    }
}
const queryAnalyzer = new QueryAnalyzer();


// -----------------------------------------------------------------
// SECȚIUNEA 4: Managementul Colecțiilor (Creare, Afișare, Ștergere)
// -----------------------------------------------------------------

/**
 * Încarcă lista de colecții de la server și o afișează în interfață.
 * Folosește cache-ul pentru a evita cereri repetate.
 */
async function loadCollections() {
    try {
        showLoading('.sidebar', 'Încărcare colecții...');
        const cacheKey = 'collections_list_v3';
        let collections = cache.get(cacheKey);
        
        if (!collections) {
            collections = await withRetry(() => fetchWithErrorHandling(`${API_BASE_URL}/collections`));
            cache.set(cacheKey, collections);
        }
        
        populateCollectionsList(collections);
        notifications.show(`Încărcate ${collections.length} colecții`, 'success', { title: 'Colecții', duration: 2000 });
        
    } catch (error) {
        console.error('Eroare la încărcarea colecțiilor:', error);
        populateCollectionsList([]);
        notifications.show('Nu s-au putut încărca colecțiile', 'error', { title: 'Eroare', actions: [{ id: 'retry', label: 'Reîncearcă', callback: loadCollections }] });
    } finally {
        hideLoading('.sidebar');
    }
}

/**
 * Construiește și afișează lista de colecții în sidebar.
 * @param {Array<string>} collections - Lista cu numele colecțiilor.
 */
function populateCollectionsList(collections) {
    const collectionsList = document.getElementById('collectionsList');
    collectionsList.innerHTML = '';
    
    if (collections.length === 0) {
        collectionsList.innerHTML = `
            <li class="list-group-item text-center text-muted empty-state">
                <div class="empty-icon">📁</div>
                <div class="empty-title">Nu există colecții</div>
                <div class="empty-description">Creați prima colecție pentru a începe</div>
                <button class="btn btn-primary btn-sm mt-2" onclick="document.getElementById('createCollectionBtn').click()">
                    <i class="bi bi-plus-circle"></i> Creare colecție
                </button>
            </li>`;
        return;
    }
    
    collections.forEach(collection => {
        const li = document.createElement('li');
        li.className = 'list-group-item collection-item d-flex justify-content-between align-items-center';
        const isActive = currentCollection === collection;
        if (isActive) li.classList.add('active');
        
        li.innerHTML = `
            <span class="collection-name" data-collection="${collection}" role="button">
                <div class="collection-info">
                    <i class="bi bi-folder${isActive ? '-open' : ''}"></i>
                    <span class="collection-title">${escapeHtml(collection)}</span>
                </div>
                <small class="collection-meta text-muted">Click pentru a selecta</small>
            </span>
            <div class="collection-actions">
                <button class="btn btn-sm btn-outline-info me-1 collection-stats" data-collection="${collection}" title="Statistici colecție" aria-label="Vezi statistici pentru ${collection}"><i class="bi bi-graph-up"></i></button>
                <button class="btn btn-sm btn-outline-danger delete-collection" data-collection="${collection}" title="Șterge colecția" aria-label="Șterge colecția ${collection}"><i class="bi bi-trash"></i></button>
            </div>`;
        
        li.querySelector('.collection-name').addEventListener('click', () => selectCollection(collection));
        li.querySelector('.collection-stats').addEventListener('click', (e) => { e.stopPropagation(); showCollectionStats(collection); });
        li.querySelector('.delete-collection').addEventListener('click', (e) => { e.stopPropagation(); showDeleteConfirmation('collection', collection); });
        
        collectionsList.appendChild(li);
    });
}

/**
 * Afișează statisticile pentru o anumită colecție într-un modal.
 * @param {string} collectionName - Numele colecției.
 */
async function showCollectionStats(collectionName) {
    try {
        showLoading('.main-content', 'Încărcare statistici...');
        const stats = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${collectionName}/stats`);
        
        const modal = createStatsModal(collectionName, stats);
        document.body.appendChild(modal);
        
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
        
    } catch (error) {
        notifications.show(`Eroare la încărcarea statisticilor pentru ${collectionName}`, 'error');
    } finally {
        hideLoading('.main-content');
    }
}

/**
 * Creează elementul DOM pentru modalul de statistici.
 * @param {string} collectionName - Numele colecției.
 * @param {object} stats - Obiectul cu statistici.
 * @returns {HTMLElement} Elementul modal.
 */
function createStatsModal(collectionName, stats) {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.tabIndex = -1;
    
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="bi bi-graph-up text-primary"></i>
                        Statistici colecție: ${escapeHtml(collectionName)}
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    ${generateStatsHTML(stats)}
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Închide</button>
                    <button type="button" class="btn btn-primary" onclick="exportStats('${collectionName}', ${JSON.stringify(stats).replace(/"/g, '&quot;')})">
                        <i class="bi bi-download"></i> Export statistici
                    </button>
                </div>
            </div>
        </div>`;
    
    return modal;
}

/**
 * Generează conținutul HTML pentru afișarea statisticilor.
 * @param {object} stats - Obiectul cu statistici.
 * @returns {string} Stringul HTML.
 */
function generateStatsHTML(stats) {
    const summary = stats.summary || {};
    const sources = stats.sources || {};
    const topKeywords = stats.top_keywords || [];
    
    return `
        <div class="row">
            <div class="col-md-6">
                <div class="stats-card">
                    <h6 class="stats-title">📊 Informații generale</h6>
                    <div class="stats-grid">
                        <div class="stat-item"><span class="stat-label">Total chunk-uri:</span><span class="stat-value">${summary.total_chunks || 0}</span></div>
                        <div class="stat-item"><span class="stat-label">Total cuvinte:</span><span class="stat-value">${(summary.total_words || 0).toLocaleString()}</span></div>
                        <div class="stat-item"><span class="stat-label">Surse unice:</span><span class="stat-value">${summary.unique_sources || 0}</span></div>
                        <div class="stat-item"><span class="stat-label">Lungime medie chunk:</span><span class="stat-value">${summary.average_chunk_length || 0} caractere</span></div>
                        <div class="stat-item"><span class="stat-label">Keywords unice:</span><span class="stat-value">${summary.unique_keywords || 0}</span></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="stats-card">
                    <h6 class="stats-title">🏷️ Top Keywords</h6>
                    <div class="keywords-list">
                        ${topKeywords.slice(0, 10).map(([keyword, count]) => `<div class="keyword-item"><span class="keyword">${escapeHtml(keyword)}</span><span class="keyword-count badge bg-primary">${count}</span></div>`).join('')}
                    </div>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-12">
                <div class="stats-card">
                    <h6 class="stats-title">📁 Surse de documente</h6>
                    <div class="sources-table">
                        <table class="table table-sm">
                            <thead><tr><th>Sursă</th><th>Chunk-uri</th><th>Cuvinte</th><th>Caractere</th></tr></thead>
                            <tbody>
                                ${Object.entries(sources).map(([source, data]) => `
                                    <tr>
                                        <td title="${escapeHtml(source)}">${escapeHtml(source.length > 30 ? source.substring(0, 30) + '...' : source)}</td>
                                        <td>${data.chunks || 0}</td>
                                        <td>${(data.words || 0).toLocaleString()}</td>
                                        <td>${(data.chars || 0).toLocaleString()}</td>
                                    </tr>`).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>`;
}

/**
 * Creează o nouă colecție.
 * @param {string} name - Numele noii colecții.
 */
async function createCollection(name) {
    try {
        if (!name || !/^[a-zA-Z0-9_]{1,50}$/.test(name)) {
            notifications.show('Numele colecției trebuie să conțină doar litere, cifre și underscore (_), maximum 50 caractere.', 'error', { title: 'Validare' });
            return false;
        }
        
        const collections = await fetchWithErrorHandling(`${API_BASE_URL}/collections`);
        if (collections.includes(name)) {
            notifications.show(`Colecția "${name}" există deja.`, 'warning', { title: 'Colecție existentă' });
            return false;
        }
        
        showLoading('.main-content', 'Creare colecție...');
        await fetchWithErrorHandling(`${API_BASE_URL}/collections/${name}`, {
            method: 'POST',
            body: JSON.stringify({ name: name, description: `Colecție creată la ${new Date().toLocaleString()}`, enable_hybrid_search: true, default_language: 'auto' })
        });
        
        notifications.show(`Colecția "${name}" a fost creată cu succes.`, 'success', { title: 'Colecție creată', actions: [{ id: 'select', label: 'Selectează', callback: () => selectCollection(name) }] });
        
        cache.clear();
        await loadCollections();
        selectCollection(name);
        
        return true;
    } catch (error) {
        console.error('Eroare la crearea colecției:', error);
        return false;
    } finally {
        hideLoading('.main-content');
    }
}

/**
 * Șterge o colecție existentă.
 * @param {string} name - Numele colecției de șters.
 */
async function deleteCollection(name) {
    try {
        showLoading('.main-content', 'Ștergere colecție...');
        await fetchWithErrorHandling(`${API_BASE_URL}/collections/${name}`, { method: 'DELETE' });
        notifications.show(`Colecția "${name}" a fost ștearsă cu succes.`, 'success', { title: 'Colecție ștearsă' });
        
        if (currentCollection === name) {
            currentCollection = null;
            updateUI();
        }
        
        cache.clear();
        await loadCollections();
        return true;
    } catch (error) {
        console.error('Eroare la ștergerea colecției:', error);
        return false;
    } finally {
        hideLoading('.main-content');
    }
}

/**
 * Selectează o colecție ca fiind cea activă.
 * @param {string} name - Numele colecției de selectat.
 */
function selectCollection(name) {
    currentCollection = name;
    updateUI();
    
    document.querySelectorAll('#collectionsList .collection-item').forEach(item => {
        item.classList.remove('active');
        const collectionSpan = item.querySelector('.collection-name');
        if (collectionSpan && collectionSpan.getAttribute('data-collection') === name) {
            item.classList.add('active');
            const icon = item.querySelector('.bi');
            if (icon) icon.className = 'bi bi-folder-open';
        } else {
            const icon = item.querySelector('.bi-folder-open');
            if(icon) icon.className = 'bi bi-folder';
        }
    });
    
    localStorage.setItem('rag_current_collection', name);
    loadDocuments(name);
    notifications.show(`Colecția "${name}" a fost selectată.`, 'info', { title: 'Colecție selectată', duration: 2000 });
}

/**
 * Actualizează interfața grafică în funcție de starea curentă (ex: dacă e selectată o colecție).
 */
function updateUI() {
    const currentCollectionEl = document.getElementById('currentCollection');
    const uploadBtn = document.getElementById('uploadDocumentBtn');
    const queryBtn = document.getElementById('runQueryBtn');
    
    if (currentCollection) {
        currentCollectionEl.innerHTML = `
            <i class="bi bi-folder-open text-primary"></i> 
            Colecția: <strong>${currentCollection}</strong>
            <small class="text-muted ms-2"><span id="collectionStatus">Încărcare...</span></small>`;
        uploadBtn.disabled = false;
        queryBtn.disabled = false;
        updateCollectionStatus();
    } else {
        currentCollectionEl.innerHTML = `
            <i class="bi bi-folder text-muted"></i>
            Selectați o colecție
            <small class="text-muted d-block">Pentru a începe să lucrați cu documente JSON</small>`;
        uploadBtn.disabled = true;
        queryBtn.disabled = true;
        document.getElementById('documentsList').innerHTML = '<tr><td colspan="4" class="text-center text-muted p-5">Vă rugăm selectați o colecție.</td></tr>';
    }
}

/**
 * Obține și afișează asincron statusul colecției curente (număr de chunk-uri și cuvinte).
 */
async function updateCollectionStatus() {
    try {
        const stats = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${currentCollection}/stats`);
        const statusEl = document.getElementById('collectionStatus');
        if (statusEl && stats.summary) {
            statusEl.innerHTML = `${stats.summary.total_chunks || 0} chunk-uri, ${(stats.summary.total_words || 0).toLocaleString()} cuvinte`;
        }
    } catch (error) {
        const statusEl = document.getElementById('collectionStatus');
        if (statusEl) statusEl.textContent = 'Status indisponibil';
    }
}


// -----------------------------------------------------------------
// SECȚIUNEA 5: Managementul Documentelor (Afișare, Upload, Ștergere)
// -----------------------------------------------------------------

/**
 * Încarcă și afișează documentele pentru colecția specificată.
 * @param {string} collectionName - Numele colecției.
 */
async function loadDocuments(collectionName) {
    try {
        showLoading('.main-content', 'Încărcare documente...');
        const cacheKey = `documents_${collectionName}_v3`;
        let documents = cache.get(cacheKey);
        
        if (!documents) {
            documents = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${collectionName}/documents`);
            cache.set(cacheKey, documents);
        }
        
        displayDocuments(documents);
        window.documentsList = documents;
        notifications.show(`Încărcate ${documents.length} documente din ${collectionName}`, 'success', { title: 'Documente', duration: 2000 });
        
    } catch (error) {
        console.error('Eroare la încărcarea documentelor:', error);
        displayDocumentsError();
    } finally {
        hideLoading('.main-content');
    }
}

/**
 * Construiește și afișează tabelul cu documente.
 * @param {Array<object>} documents - Lista de obiecte document.
 */
function displayDocuments(documents) {
    const documentsListEl = document.getElementById('documentsList');
    
    if (documents.length === 0) {
        documentsListEl.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-5">
                    <div class="empty-state">
                        <div class="empty-icon">📄</div>
                        <div class="empty-title">Nu există fișiere JSON în această colecție</div>
                        <div class="empty-description">Încărcați fișiere JSON chunkizate pentru a începe</div>
                        <button class="btn btn-success mt-3" onclick="document.getElementById('uploadDocumentBtn').click()"><i class="bi bi-upload"></i> Încarcă primul fișier JSON</button>
                    </div>
                </td>
            </tr>`;
        return;
    }
    
    let html = '';
    documents.forEach((file, index) => {
        const chunkCount = file.doc_count || 0;
        const wordCount = file.word_count || 0;
        const qualityScore = file.quality_score || 0;
        
        let badgeClass = 'bg-secondary', statusText = 'Mic';
        if (chunkCount > 50) { badgeClass = 'bg-success'; statusText = 'Mare'; }
        else if (chunkCount > 20) { badgeClass = 'bg-warning'; statusText = 'Mediu'; }
        else if (chunkCount > 10) { badgeClass = 'bg-info'; statusText = 'Modest'; }
        
        let qualityBadge = 'bg-secondary', qualityText = 'N/A';
        if (qualityScore > 0) {
            if (qualityScore > 0.7) { qualityBadge = 'bg-success'; qualityText = 'Excelent'; }
            else if (qualityScore > 0.5) { qualityBadge = 'bg-warning'; qualityText = 'Bun'; }
            else { qualityBadge = 'bg-danger'; qualityText = 'Scăzut'; }
        }
        
        html += `
        <tr class="document-row" data-index="${index}">
            <td>
                <div class="d-flex align-items-center">
                    <i class="bi bi-file-earmark-code text-primary me-2 fs-5"></i> 
                    <div class="document-info">
                        <div class="document-name">${escapeHtml(file.source || 'Necunoscut')}</div>
                        <small class="text-muted">Fișier JSON chunkizat ${file.language_detected ? `• ${file.language_detected}` : ''} ${file.processing_version ? `• v${file.processing_version}` : ''}</small>
                    </div>
                </div>
            </td>
            <td>
                <div class="chunk-info">
                    <span class="badge ${badgeClass} me-1">${chunkCount} chunk-uri</span>
                    <span class="badge bg-light text-dark">${statusText}</span>
                </div>
                <small class="text-muted d-block mt-1">${wordCount > 0 ? `${wordCount.toLocaleString()} cuvinte` : 'Indexate pentru căutare'}</small>
            </td>
            <td>
                <div class="date-info"><i class="bi bi-calendar3"></i> ${escapeHtml(file.created_at || '-')}</div>
                <small class="text-muted">Data procesării</small>
                ${qualityScore > 0 ? `<div class="mt-1"><span class="badge ${qualityBadge}">${qualityText}</span><small class="text-muted ms-1">${(qualityScore * 100).toFixed(1)}%</small></div>` : ''}
            </td>
            <td>
                <div class="btn-group" role="group">
                    <button class="btn btn-sm btn-outline-info preview-document" data-source="${escapeHtml(file.source)}" title="Previzualizare document"><i class="bi bi-eye"></i></button>
                    <button class="btn btn-sm btn-outline-danger delete-document" data-source="${escapeHtml(file.source)}" title="Șterge fișierul JSON"><i class="bi bi-trash"></i></button>
                </div>
            </td>
        </tr>`;
    });
    
    documentsListEl.innerHTML = html;
    
    document.querySelectorAll('.delete-document').forEach(button => button.addEventListener('click', () => showDeleteConfirmation('document', button.getAttribute('data-source'))));
    document.querySelectorAll('.preview-document').forEach(button => button.addEventListener('click', () => showDocumentPreview(button.getAttribute('data-source'))));
}

/**
 * Afișează un mesaj de eroare în tabelul de documente.
 */
function displayDocumentsError() {
    document.getElementById('documentsList').innerHTML = `
        <tr>
            <td colspan="4" class="text-center py-5">
                <div class="error-state">
                    <div class="error-icon">⚠️</div>
                    <div class="error-title">Eroare la încărcarea documentelor</div>
                    <div class="error-description">Verificați conexiunea la server și încercați din nou</div>
                    <button class="btn btn-primary mt-3" onclick="loadDocuments(currentCollection)"><i class="bi bi-arrow-clockwise"></i> Reîncearcă</button>
                </div>
            </td>
        </tr>`;
}

/**
 * Previzualizează conținutul unui document (primele chunk-uri) într-un modal.
 * @param {string} source - Numele fișierului sursă.
 */
async function showDocumentPreview(source) {
    try {
        showLoading('.main-content', 'Încărcare previzualizare...');
        const searchResults = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${currentCollection}/search?query=source:${encodeURIComponent(source)}&top_k=5`);
        
        const modal = createPreviewModal(source, searchResults.results || []);
        document.body.appendChild(modal);
        
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        modal.addEventListener('hidden.bs.modal', () => document.body.removeChild(modal));
        
    } catch (error) {
        notifications.show(`Eroare la previzualizarea documentului: ${source}`, 'error');
    } finally {
        hideLoading('.main-content');
    }
}

/**
 * Creează elementul DOM pentru modalul de previzualizare a documentului.
 * @param {string} source - Numele sursei.
 * @param {Array<object>} chunks - Lista de chunk-uri.
 * @returns {HTMLElement} Elementul modal.
 */
function createPreviewModal(source, chunks) {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.tabIndex = -1;
    
    modal.innerHTML = `
        <div class="modal-dialog modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title"><i class="bi bi-eye text-primary"></i> Previzualizare: ${escapeHtml(source)}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" style="max-height: 600px; overflow-y: auto;">${generatePreviewHTML(chunks)}</div>
                <div class="modal-footer">
                    <div class="me-auto"><small class="text-muted">Afișate ${chunks.length} chunk-uri din document</small></div>
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Închide</button>
                </div>
            </div>
        </div>`;
    
    return modal;
}

/**
 * Generează conținutul HTML pentru previzualizarea chunk-urilor.
 * @param {Array<object>} chunks - Lista de chunk-uri.
 * @returns {string} Stringul HTML.
 */
function generatePreviewHTML(chunks) {
    if (!chunks || chunks.length === 0) {
        return `<div class="alert alert-info"><i class="bi bi-info-circle"></i> Nu s-au găsit chunk-uri pentru acest document.</div>`;
    }
    
    return chunks.map((chunk, index) => {
        const content = chunk.content || '', meta = chunk.meta || {};
        return `
            <div class="chunk-preview card mb-3">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h6 class="mb-0"><i class="bi bi-file-text text-info"></i> ${meta.chunk_id || `Chunk ${index + 1}`}</h6>
                    <div class="chunk-meta">
                        ${meta.word_count ? `<span class="badge bg-secondary">${meta.word_count} cuvinte</span>` : ''}
                        ${chunk.score ? `<span class="badge bg-primary ms-1">${(chunk.score * 100).toFixed(1)}%</span>` : ''}
                    </div>
                </div>
                <div class="card-body">
                    <div class="chunk-content">${formatContentForPreview(content)}</div>
                    ${meta.keywords ? `<div class="mt-3"><small class="text-muted">Keywords:</small><div class="keywords-tags mt-1">${meta.keywords.split(',').slice(0, 5).map(keyword => `<span class="badge bg-light text-dark me-1">${keyword.trim()}</span>`).join('')}</div></div>` : ''}
                </div>
            </div>`;
    }).join('');
}

/**
 * Formatează conținutul unui chunk pentru a fi afișat în previzualizare.
 * @param {string} content - Conținutul text.
 * @returns {string} HTML formatat.
 */
function formatContentForPreview(content) {
    if (!content) return '<em class="text-muted">Conținut gol</em>';
    let preview = content.length > 500 ? content.substring(0, 500) + '...' : content;
    preview = escapeHtml(preview).replace(/\n/g, '<br>');
    return preview;
}

/**
 * Validează un fișier JSON selectat de utilizator înainte de upload.
 * Verifică extensia, dimensiunea și structura internă a chunk-urilor.
 * @param {File} file - Fișierul de validat.
 * @returns {Promise<object>} O promisiune care se rezolvă cu un obiect de validare.
 */
async function validateJsonFile(file) {
    return new Promise((resolve, reject) => {
        if (!file.name.toLowerCase().endsWith('.json')) return reject(new Error('Fișierul trebuie să aibă extensia .json'));
        if (file.size > 100 * 1024 * 1024) return reject(new Error('Fișierul este prea mare. Maxim 100MB.'));
        if (file.size < 50) return reject(new Error('Fișierul este prea mic pentru a fi un JSON valid.'));
        
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const jsonData = JSON.parse(e.target.result);
                if (typeof jsonData !== 'object' || Array.isArray(jsonData)) return reject(new Error('JSON-ul trebuie să fie un obiect, nu o listă.'));
                
                const chunkKeys = Object.keys(jsonData).filter(key => /^chunk_\d+$/.test(key));
                if (chunkKeys.length === 0) return reject(new Error('Nu s-au găsit chunk-uri în formatul așteptat (chunk_0, chunk_1, etc.).'));
                
                let validChunks = 0, totalContentLength = 0;
                for (const key of chunkKeys) {
                    const chunk = jsonData[key];
                    if (typeof chunk === 'object' && chunk.metadata && chunk.chunk && typeof chunk.metadata === 'string' && typeof chunk.chunk === 'string' && chunk.chunk.trim().length >= 10) {
                        validChunks++;
                        totalContentLength += chunk.chunk.trim().length;
                    }
                }
                
                if (validChunks === 0) return reject(new Error('Nu s-au găsit chunk-uri valide cu structura corectă.'));
                
                const avgChunkLength = totalContentLength / validChunks;
                resolve({ isValid: true, totalChunks: chunkKeys.length, validChunks, averageChunkLength: Math.round(avgChunkLength), recommendations: generateFileRecommendations(chunkKeys.length, validChunks, avgChunkLength) });
                
            } catch (error) { reject(new Error(`JSON invalid: ${error.message}`)); }
        };
        reader.onerror = () => reject(new Error('Eroare la citirea fișierului.'));
        reader.readAsText(file);
    });
}

/**
 * Generează recomandări pe baza analizei unui fișier JSON.
 * @param {number} totalChunks - Numărul total de chunk-uri.
 * @param {number} validChunks - Numărul de chunk-uri valide.
 * @param {number} avgLength - Lungimea medie a chunk-urilor.
 * @returns {Array<string>} O listă de recomandări.
 */
function generateFileRecommendations(totalChunks, validChunks, avgLength) {
    const recommendations = [];
    if (validChunks < totalChunks) recommendations.push(`${totalChunks - validChunks} chunk-uri au fost ignorate (structură invalidă)`);
    if (totalChunks < 5) recommendations.push('Fișierul conține puține chunk-uri - considerați combinarea cu alte fișiere');
    if (avgLength < 100) recommendations.push('Chunk-urile sunt scurte - căutarea poate fi mai puțin precisă');
    if (recommendations.length === 0) recommendations.push('Fișierul este optimizat pentru sistemul RAG');
    return recommendations;
}

/**
 * Încarcă un document în colecția curentă.
 * @param {string} collectionName - Numele colecției.
 * @param {File} file - Fișierul de încărcat.
 */
async function uploadDocument(collectionName, file) {
    if (!file) return notifications.show('Selectați un fișier pentru încărcare.', 'error');
    
    try {
        const validation = await validateJsonFile(file);
        notifications.show(`Fișier valid: ${validation.totalChunks} chunk-uri detectate, ${validation.validChunks} valide.`, 'success', { title: 'Validare completă' });
        
        showLoading('.main-content', 'Încărcare și procesare fișier...');
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${collectionName}/upload`, { method: 'POST', body: formData });
        
        if (response) {
            notifications.show(`Fișierul "${file.name}" a fost procesat! ${response.chunks_count || ''} chunk-uri indexate.`, 'success', { title: 'Upload complet' });
            cache.clear();
            return true;
        }
        
    } catch (error) {
        notifications.show(`Eroare la încărcarea fișierului: ${error.message}`, 'error', { title: 'Eroare upload' });
        return false;
    } finally {
        hideLoading('.main-content');
    }
}

/**
 * Șterge un document (o sursă) dintr-o colecție.
 * @param {string} collectionName - Numele colecției.
 * @param {string} source - Numele sursei de șters.
 */
async function deleteDocument(collectionName, source) {
    try {
        showLoading('.main-content', 'Ștergere document...');
        await fetchWithErrorHandling(`${API_BASE_URL}/collections/${collectionName}/documents`, {
            method: 'DELETE',
            body: JSON.stringify({ source: source, clear_cache: true, force_delete: false })
        });
        
        notifications.show(`Fișierul "${source}" a fost șters cu succes.`, 'success', { title: 'Document șters' });
        cache.clear();
        await loadDocuments(collectionName);
        return true;
    } catch (error) {
        console.error('Eroare la ștergerea documentului:', error);
        return false;
    } finally {
        hideLoading('.main-content');
    }
}


// -----------------------------------------------------------------
// SECȚIUNEA 6: Interogare & Afișare Rezultate
// -----------------------------------------------------------------

/**
 * Funcția principală pentru gestionarea unei interogări.
 * Analizează textul, aplică optimizări, verifică cache-ul și trimite cererea la API.
 */
async function handleQuery() {
    const queryText = document.getElementById('queryInput').value.trim();
    if (!queryText) return notifications.show('Introduceți o întrebare.', 'error', { title: 'Query gol' });
    if (!currentCollection) return notifications.show('Selectați o colecție.', 'error');
    
    const queryAnalysis = queryAnalyzer.analyzeQuery(queryText);
    if (queryAnalysis.suggestions.length > 0) {
        notifications.show(`Sugestie: ${queryAnalysis.suggestions[0]}`, 'info', { title: 'Optimizare query' });
    }

    const optimizedParams = {
        topK: parseInt(document.getElementById('topK').value) || queryAnalysis.optimizations.recommendedTopK,
        temperature: parseFloat(document.getElementById('temperature').value) || queryAnalysis.optimizations.recommendedTemperature,
        threshold: parseFloat(document.getElementById('similarityThreshold')?.value) || queryAnalysis.optimizations.recommendedThreshold,
        hybridSearch: document.getElementById('useHybridSearch')?.checked ?? queryAnalysis.optimizations.enableHybridSearch
    };
    
    const cacheKey = `query_${currentCollection}_${queryText}_${optimizedParams.topK}_${optimizedParams.temperature}_${optimizedParams.threshold}_${optimizedParams.hybridSearch}`;
    let cachedResult = cache.get(cacheKey);
    
    if (cachedResult) {
        displayGeneratedResults(cachedResult, queryText, queryAnalysis);
        notifications.show('Rezultat din cache - răspuns instant!', 'success', { title: 'Cache hit', duration: 2000 });
        return;
    }
    
    const startTime = performance.now();
    showLoading('#query', 'Căutare și generare răspuns cu AI...');
    
    try {
        const requestBody = {
            query: queryText,
            temperature: optimizedParams.temperature,
            top_k_docs: optimizedParams.topK,
            use_hybrid_search: optimizedParams.hybridSearch,
            similarity_threshold: optimizedParams.threshold,
            enable_query_expansion: true,
            search_method: optimizedParams.hybridSearch ? 'hybrid' : 'semantic'
        };
        
        const response = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${currentCollection}/generate`, { method: 'POST', body: JSON.stringify(requestBody) });
        
        const endTime = performance.now();
        const clientProcessingTime = endTime - startTime;
        
        queryAnalytics.totalQueries++;
        
        const enhancedResponse = { ...response, clientProcessingTime, queryAnalysis, optimizedParams, cachedAt: Date.now() };
        cache.set(cacheKey, enhancedResponse);
        
        addToSearchHistory(queryText, response, queryAnalysis);
        displayGeneratedResults(enhancedResponse, queryText, queryAnalysis);
        
        notifications.show(`Răspuns generat în ${clientProcessingTime.toFixed(0)}ms folosind ${response.documents?.length || 0} documente`, 'success', { title: 'Query complet' });
        
    } catch (error) {
        console.error('Eroare la procesarea interogării:', error);
        document.getElementById('queryResults').innerHTML = `<div class="alert alert-danger"><h6><i class="bi bi-exclamation-triangle"></i> Eroare la procesarea interogării</h6><p>${error.message}</p></div>`;
    } finally {
        hideLoading('#query');
    }
}

/**
 * Adaugă o interogare în istoricul local.
 * @param {string} query - Textul interogării.
 * @param {object} response - Răspunsul de la server.
 * @param {object} analysis - Analiza interogării.
 */
function addToSearchHistory(query, response, analysis) {
    const historyItem = { id: Date.now(), query: query.substring(0, 100), collection: currentCollection, timestamp: new Date().toISOString(), documentsFound: response.documents?.length || 0, processingTime: response.query_time_ms || 0, complexity: analysis.complexity, language: analysis.language, intent: analysis.intent };
    searchHistory.unshift(historyItem);
    if (searchHistory.length > 50) searchHistory = searchHistory.slice(0, 50);
    saveToStorage('rag_search_history', searchHistory);
}

/**
 * Afișează răspunsul generat de AI și sursele utilizate.
 * @param {object} data - Datele complete ale răspunsului.
 * @param {string} queryText - Textul interogării originale.
 * @param {object} queryAnalysis - Analiza interogării.
 */
function displayGeneratedResults(data, queryText, queryAnalysis) {
    const resultsContainer = document.getElementById('queryResults');
    resultsContainer.innerHTML = '';
    
    if (!data || !data.answer) {
        resultsContainer.innerHTML = `<div class="alert alert-warning"><h6><i class="bi bi-exclamation-circle"></i> Nu s-a putut genera un răspuns</h6><p>Nu s-au găsit informații relevante.</p></div>`;
        if (data && data.documents && data.documents.length > 0) displayQueryResults(data.documents, true);
        return;
    }
    
    const answerContainer = document.createElement('div');
    answerContainer.className = 'generated-answer mb-4';
    answerContainer.innerHTML = `
        <div class="answer-header">
            <h5 class="question-title"><i class="bi bi-question-circle text-primary"></i> ${escapeHtml(queryText)}</h5>
        </div>
        <div class="answer-content">
            <div class="answer-header-section">
                <i class="bi bi-robot text-success fs-5 me-2"></i> 
                <strong>Răspuns generat de AI:</strong>
            </div>
            <div class="answer-text mt-3">${formatAnswer(data.answer)}</div>
        </div>
        <div class="answer-footer mt-3">
             <small class="text-muted"><i class="bi bi-clock"></i> ${data.query_time_ms || 0}ms server + ${Math.round(data.clientProcessingTime || 0)}ms client | <i class="bi bi-files"></i> ${data.documents?.length || 0} documente folosite</small>
        </div>`;
    resultsContainer.appendChild(answerContainer);
    
    if (data.documents && data.documents.length > 0) {
        const sourcesSection = document.createElement('div');
        sourcesSection.className = 'sources-section';
        sourcesSection.innerHTML = `<h5 class="sources-title mt-4 mb-3"><i class="bi bi-file-text text-info"></i> Surse utilizate pentru răspuns</h5>`;
        resultsContainer.appendChild(sourcesSection);
        displayQueryResults(data.documents, true);
    }
    
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Afișează chunk-urile relevante returnate de căutare.
 * @param {Array<object>} results - Lista de rezultate.
 * @param {boolean} isSourceDocuments - Indică dacă sunt surse pentru un răspuns generat.
 */
function displayQueryResults(results, isSourceDocuments = false) {
    const resultsContainer = document.getElementById('queryResults');
    
    if (!results || results.length === 0) {
        if (!isSourceDocuments) resultsContainer.innerHTML = `<div class="alert alert-info"><i class="bi bi-info-circle"></i> Nu s-au găsit chunk-uri relevante.</div>`;
        return;
    }
    
    results.forEach((result, index) => {
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        
        const score = result.score ? (result.score * 100).toFixed(1) : 'N/A';
        const source = result.meta?.original_source || result.meta?.source || 'Necunoscută';
        const chunkId = result.meta?.chunk_id || `chunk_${index}`;
        
        resultCard.innerHTML = `
            <div class="result-header">
                <div class="d-flex align-items-center"><i class="bi bi-file-earmark-code text-primary me-2"></i><span class="result-number fw-bold">${escapeHtml(chunkId)}</span></div>
                <div class="result-score-section"><span class="score-value">${score}%</span></div>
            </div>
            <div class="result-content"><div class="content-text">${formatContent(result.content)}</div></div>
            <div class="result-footer">
                <div class="source-info"><i class="bi bi-file-text text-muted me-1"></i><strong>Sursă:</strong> <span title="${escapeHtml(source)}">${escapeHtml(source.length > 50 ? source.substring(0, 50) + '...' : source)}</span></div>
                <div class="result-actions"><button class="btn btn-sm btn-outline-secondary" onclick="expandChunk(this, '${escapeHtml(result.content)}')"><i class="bi bi-arrows-expand"></i> Extinde</button></div>
            </div>`;
        resultsContainer.appendChild(resultCard);
    });
}


// -----------------------------------------------------------------
// SECȚIUNEA 7: Funcții Helper pentru UI și Interacțiune
// -----------------------------------------------------------------

/**
 * Extinde sau restrânge conținutul unui chunk afișat.
 * @param {HTMLElement} button - Butonul pe care s-a dat click.
 * @param {string} content - Conținutul complet al chunk-ului.
 */
function expandChunk(button, content) {
    const contentDiv = button.closest('.result-card').querySelector('.content-text');
    if (button.querySelector('i').classList.contains('bi-arrows-expand')) {
        contentDiv.innerHTML = formatContent(content, false); // Fără trunchiere
        button.innerHTML = '<i class="bi bi-arrows-collapse"></i> Restrânge';
    } else {
        contentDiv.innerHTML = formatContent(content, true); // Cu trunchiere
        button.innerHTML = '<i class="bi bi-arrows-expand"></i> Extinde';
    }
}

/**
 * Formatează textul răspunsului AI pentru afișare (ex: adaugă paragrafe).
 * @param {string} answer - Textul răspunsului.
 * @returns {string} HTML formatat.
 */
function formatAnswer(answer) {
    if (!answer) return '';
    return escapeHtml(answer).replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>').replace(/^/, '<p>').replace(/$/, '</p>').replace(/<p><\/p>/g, '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>');
}

/**
 * Formatează conținutul unui chunk, trunchindu-l și evidențiind termenii din query.
 * @param {string} content - Conținutul de formatat.
 * @param {boolean} truncate - Dacă se trunchiază conținutul.
 * @returns {string} HTML formatat.
 */
function formatContent(content, truncate = true) {
    if (!content) return '<em class="text-muted">Conținut gol</em>';
    let formatted = escapeHtml(content);
    if (truncate && formatted.length > 500) formatted = formatted.substring(0, 500) + '...';
    
    const currentQuery = document.getElementById('queryInput')?.value.toLowerCase() || '';
    if (currentQuery) {
        const queryWords = currentQuery.split(/\s+/).filter(word => word.length > 2);
        queryWords.forEach(word => {
            const regex = new RegExp(`(${escapeRegex(word)})`, 'gi');
            formatted = formatted.replace(regex, '<mark>$1</mark>');
        });
    }
    return formatted;
}

/**
 * Afișează un modal de confirmare înainte de o acțiune distructivă (ștergere).
 * @param {string} type - Tipul acțiunii ('collection' sau 'document').
 * @param {string} name - Numele elementului de șters.
 */
function showDeleteConfirmation(type, name) {
    const modal = new bootstrap.Modal(document.getElementById('deleteConfirmModal'));
    const confirmText = document.getElementById('deleteConfirmText');
    const confirmBtn = document.getElementById('confirmDeleteBtn');
    
    if (type === 'collection') {
        confirmText.innerHTML = `<p class="fw-bold">Ești sigur că vrei să ștergi colecția <span class="text-primary">"${escapeHtml(name)}"</span>?</p><p class="text-danger">Această acțiune este ireversibilă și va șterge toate documentele din colecție.</p>`;
        confirmBtn.onclick = async () => { modal.hide(); await deleteCollection(name); };
    } else if (type === 'document') {
        confirmText.innerHTML = `<p class="fw-bold">Ești sigur că vrei să ștergi fișierul <span class="text-primary">"${escapeHtml(name)}"</span>?</p><p class="text-danger">Această acțiune este ireversibilă.</p>`;
        confirmBtn.onclick = async () => { modal.hide(); await deleteDocument(currentCollection, name); };
    }
    
    modal.show();
}

/**
 * Caută live în lista de colecții pe măsură ce utilizatorul tastează.
 */
const searchCollections = debounce((searchTerm) => {
    const items = document.querySelectorAll('#collectionsList .collection-item');
    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(searchTerm.toLowerCase()) ? '' : 'none';
    });
}, CONFIG.DEBOUNCE_DELAY);


// -----------------------------------------------------------------
// SECȚIUNEA 8: Inițializare și Gestiune Evenimente
// -----------------------------------------------------------------

/**
 * Setează scurtăturile de la tastatură.
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'n' && !e.shiftKey) { e.preventDefault(); document.getElementById('createCollectionBtn')?.click(); }
        if (e.ctrlKey && e.key === 'u' && currentCollection) { e.preventDefault(); document.getElementById('uploadDocumentBtn')?.click(); }
        if (e.ctrlKey && e.key === 'q') { e.preventDefault(); document.getElementById('queryInput')?.focus(); }
        if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); if (document.activeElement.id === 'queryInput') handleQuery(); }
        if (e.ctrlKey && e.shiftKey && e.key === 'C') { e.preventDefault(); clearAllCache(); }
        if (e.key === 'Escape') { const activeModal = bootstrap.Modal.getInstance(document.querySelector('.modal.show')); if (activeModal) activeModal.hide(); }
        if (e.key === 'F1') { e.preventDefault(); showHelp(); }
    });
}

/**
 * Salvează preferințele utilizatorului (ex: parametrii de căutare) în localStorage.
 */
function saveUserPreferences() {
    const preferences = {
        topK: document.getElementById('topK')?.value,
        temperature: document.getElementById('temperature')?.value,
        similarityThreshold: document.getElementById('similarityThreshold')?.value,
        useHybridSearch: document.getElementById('useHybridSearch')?.checked,
    };
    saveToStorage('rag_user_preferences', preferences);
}

/**
 * Restaurează preferințele utilizatorului din localStorage la încărcarea paginii.
 */
function restoreUserPreferences() {
    const preferences = loadFromStorage('rag_user_preferences');
    if (!preferences) return;
    
    if(preferences.topK) document.getElementById('topK').value = preferences.topK;
    if(preferences.temperature) document.getElementById('temperature').value = preferences.temperature;
    if(preferences.similarityThreshold) document.getElementById('similarityThreshold').value = preferences.similarityThreshold;
    if(preferences.useHybridSearch !== undefined) document.getElementById('useHybridSearch').checked = preferences.useHybridSearch;
    
    const lastQuery = loadFromStorage('rag_last_query');
    if (lastQuery) document.getElementById('queryInput').value = lastQuery;
    
    const lastCollection = loadFromStorage('rag_current_collection');
    if (lastCollection) {
        setTimeout(() => {
            const collectionNames = Array.from(document.querySelectorAll('.collection-name')).map(el => el.getAttribute('data-collection'));
            if (collectionNames.includes(lastCollection)) selectCollection(lastCollection);
        }, 1000);
    }
}

/**
 * Salvează starea curentă a aplicației în localStorage.
 */
function saveApplicationState() {
    const state = {
        currentCollection,
        queryAnalytics,
        cacheStats: cache.getStats(),
        timestamp: new Date().toISOString(),
        version: '3.0.0'
    };
    saveToStorage('rag_app_state', state);
}

/**
 * Helper pentru a salva date în localStorage.
 * @param {string} key - Cheia sub care se salvează.
 * @param {any} data - Datele de salvat.
 */
function saveToStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
        console.warn(`Nu s-au putut salva datele pentru ${key}:`, error);
    }
}

/**
 * Helper pentru a încărca date din localStorage.
 * @param {string} key - Cheia de la care se încarcă.
 * @returns {any|null} Datele încărcate sau null.
 */
function loadFromStorage(key) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    } catch (error) {
        console.warn(`Nu s-au putut încărca datele pentru ${key}:`, error);
        return null;
    }
}

/**
 * Golește complet cache-ul aplicației.
 */
function clearAllCache() {
    cache.clear();
    if (currentCollection) {
        fetchWithErrorHandling(`${API_BASE_URL}/cache/clear`, { method: 'POST' }).catch(console.error);
    }
    notifications.show('Cache-ul a fost curățat complet', 'success', { title: 'Cache' });
}

/**
 * Setează funcționalitatea de drag & drop pentru upload-ul de fișiere.
 */
function setupDragAndDrop() {
    const uploadArea = document.querySelector('.tab-content');
    if (!uploadArea) return;
    
    const preventDefaults = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    const highlight = () => uploadArea.classList.add('drag-over');
    const unhighlight = () => uploadArea.classList.remove('drag-over');
    ['dragenter', 'dragover'].forEach(eventName => uploadArea.addEventListener(eventName, highlight, false));
    ['dragleave', 'drop'].forEach(eventName => uploadArea.addEventListener(eventName, unhighlight, false));
    
    uploadArea.addEventListener('drop', (e) => {
        if (!currentCollection) {
            notifications.show('Selectați o colecție înainte de a încărca fișiere', 'warning');
            return;
        }
        const file = e.dataTransfer.files[0];
        if (file && file.name.toLowerCase().endsWith('.json')) {
            const fileInput = document.getElementById('documentFile');
            if(fileInput) {
                const newFileList = new DataTransfer();
                newFileList.items.add(file);
                fileInput.files = newFileList.files;
                new bootstrap.Modal(document.getElementById('uploadDocumentModal')).show();
            }
        } else {
            notifications.show('Doar fișierele JSON sunt acceptate!', 'error');
        }
    }, false);
}

/**
 * Afișează modalul de ajutor.
 */
function showHelp() {
    const helpModalContent = `
        <p>Acesta este un sistem RAG (Retrieval-Augmented Generation) care permite interogarea unei baze de documente JSON folosind limbaj natural.</p>
        <h6>Funcționalități principale:</h6>
        <ul>
            <li><strong>Colecții:</strong> Organizați documentele în colecții separate.</li>
            <li><strong>Upload JSON:</strong> Încărcați documente formatate ca JSON-uri cu "chunk-uri".</li>
            <li><strong>Căutare Avansată:</strong> Folosește căutare hibridă pentru a găsi cele mai relevante informații.</li>
            <li><strong>Răspunsuri AI:</strong> Generează răspunsuri concise pe baza informațiilor găsite.</li>
        </ul>
        <h6>Scurtături Tastatură:</h6>
        <ul>
            <li><kbd>Ctrl+N</kbd> - Colecție nouă</li>
            <li><kbd>Ctrl+U</kbd> - Încarcă JSON</li>
            <li><kbd>Ctrl+Q</kbd> - Focus pe câmpul de interogare</li>
            <li><kbd>Ctrl+Enter</kbd> - Execută interogarea</li>
            <li><kbd>F1</kbd> - Afișează acest ajutor</li>
        </ul>
    `;
    const helpModal = new bootstrap.Modal(document.getElementById('helpModal'));
    document.getElementById('helpModalBody').innerHTML = helpModalContent;
    helpModal.show();
}

/**
 * Exportă statisticile unei colecții ca fișier JSON.
 * @param {string} collectionName - Numele colecției.
 * @param {object} stats - Obiectul cu statistici.
 */
function exportStats(collectionName, stats) {
    const dataStr = JSON.stringify(stats, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `stats_${collectionName}_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    notifications.show(`Statistici exportate pentru ${collectionName}`, 'success');
}


/**
 * Listener principal care se execută după ce întreaga pagină a fost încărcată.
 * Inițializează toate funcționalitățile și adaugă listener-ii de evenimente.
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Inițializare aplicație RAG optimizată v3.0.0');
    
    // Inițializări de bază
    loadCollections();
    setupKeyboardShortcuts();
    setupDragAndDrop();
    
    // Restaurare stare și preferințe
    restoreUserPreferences();
    const appState = loadFromStorage('rag_app_state');
    if (appState) queryAnalytics = { ...queryAnalytics, ...appState.queryAnalytics };
    const savedHistory = loadFromStorage('rag_search_history');
    if (savedHistory) searchHistory = savedHistory;

    // Salvare automată
    setInterval(saveApplicationState, CONFIG.AUTO_SAVE_INTERVAL);
    
    // Event listeners pentru butoane și input-uri
    document.getElementById('saveCollectionBtn')?.addEventListener('click', async () => {
        const collectionName = document.getElementById('collectionName')?.value.trim();
        if (collectionName) {
            const success = await createCollection(collectionName);
            if (success) bootstrap.Modal.getInstance(document.getElementById('createCollectionModal'))?.hide();
        }
    });
    
    document.getElementById('startUploadBtn')?.addEventListener('click', async () => {
        const fileInput = document.getElementById('documentFile');
        if (!fileInput?.files?.length) return notifications.show('Selectați un fișier.', 'error');
        if (!currentCollection) return notifications.show('Selectați o colecție.', 'error');
        
        const success = await uploadDocument(currentCollection, fileInput.files[0]);
        if (success) {
            bootstrap.Modal.getInstance(document.getElementById('uploadDocumentModal'))?.hide();
            fileInput.value = '';
            await loadDocuments(currentCollection);
        }
    });
    
    document.getElementById('runQueryBtn')?.addEventListener('click', handleQuery);
    
    const queryInput = document.getElementById('queryInput');
    queryInput?.addEventListener('keypress', (e) => { if (e.key === 'Enter' && !e.ctrlKey) handleQuery(); });
    queryInput?.addEventListener('input', debounce((e) => saveToStorage('rag_last_query', e.target.value), 2000));
    
    document.getElementById('searchCollections')?.addEventListener('input', (e) => searchCollections(e.target.value));

    document.querySelectorAll('#topK, #temperature, #similarityThreshold, #useHybridSearch').forEach(el => {
        el.addEventListener('change', saveUserPreferences);
    });

    // Mesaj de bun venit
    if (!loadFromStorage('rag_visited_before')) {
        setTimeout(() => {
            notifications.show('Bun venit! Apăsați F1 pentru ajutor sau creați o colecție pentru a începe.', 'info', { title: 'Bun venit!', duration: 8000 });
            saveToStorage('rag_visited_before', true);
        }, 1500);
    }
    
    console.log('✅ Aplicație RAG inițializată cu succes');
});

// Export funcții pentru debugging în consolă
window.RAG_DEBUG = {
    cache,
    notifications,
    queryAnalyzer,
    currentCollection: () => currentCollection,
    clearCache: clearAllCache,
    analytics: () => queryAnalytics,
    history: () => searchHistory,
    appState: () => loadFromStorage('rag_app_state'),
    version: '3.0.0'
};