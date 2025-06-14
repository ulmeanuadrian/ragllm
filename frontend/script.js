// script.js - OPTIMIZAT PENTRU JSON CHUNKIZAT

// Configurare API
const API_BASE_URL = 'http://localhost:8070';

// Variabile globale
let currentCollection = null;
let documentsList = [];

// Cache pentru rezultate
const resultCache = new Map();
const MAX_CACHE_SIZE = 50;

// Funcție de debouncing pentru optimizare căutări
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Funcții utilitare optimizate
function showLoading(selector) {
    const element = document.querySelector(selector);
    if (!element) return;
    
    element.classList.add('loading');
    
    let overlay = element.querySelector('.loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="spinner"></div>
            <div class="loading-text">Se procesează JSON...</div>
        `;
        element.appendChild(overlay);
    }
    
    overlay.style.display = 'flex';
}

function hideLoading(selector) {
    const element = document.querySelector(selector);
    if (!element) return;
    
    element.classList.remove('loading');
    const overlay = element.querySelector('.loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// Sistem de notificări optimizat
class NotificationSystem {
    constructor() {
        this.container = this.createContainer();
        this.notifications = new Set();
    }

    createContainer() {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            max-width: 400px;
        `;
        document.body.appendChild(container);
        return container;
    }

    show(message, type = 'success', duration = 5000) {
        const notification = this.createNotification(message, type);
        this.container.appendChild(notification);
        this.notifications.add(notification);

        // Animație de intrare
        requestAnimationFrame(() => {
            notification.style.transform = 'translateX(0)';
            notification.style.opacity = '1';
        });

        // Auto-remove
        setTimeout(() => {
            this.remove(notification);
        }, duration);

        return notification;
    }

    createNotification(message, type) {
        const notification = document.createElement('div');
        const icons = {
            success: 'bi-check-circle',
            error: 'bi-exclamation-triangle',
            warning: 'bi-exclamation-circle',
            info: 'bi-info-circle'
        };

        const colors = {
            success: '#28a745',
            error: '#dc3545',
            warning: '#ffc107',
            info: '#17a2b8'
        };

        notification.style.cssText = `
            background: white;
            border-left: 4px solid ${colors[type] || colors.info};
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-bottom: 10px;
            padding: 16px;
            transform: translateX(100%);
            opacity: 0;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: space-between;
        `;

        notification.innerHTML = `
            <div style="display: flex; align-items: center;">
                <i class="bi ${icons[type] || icons.info}" style="margin-right: 10px; color: ${colors[type] || colors.info};"></i>
                <span>${message}</span>
            </div>
            <button class="close-btn" style="background: none; border: none; font-size: 18px; cursor: pointer; color: #666;">×</button>
        `;

        notification.querySelector('.close-btn').onclick = () => this.remove(notification);
        return notification;
    }

    remove(notification) {
        if (!this.notifications.has(notification)) return;

        notification.style.transform = 'translateX(100%)';
        notification.style.opacity = '0';

        setTimeout(() => {
            if (this.container.contains(notification)) {
                this.container.removeChild(notification);
            }
            this.notifications.delete(notification);
        }, 300);
    }
}

const notifications = new NotificationSystem();

// Wrapper pentru fetch cu gestionare avansată a erorilor
async function fetchWithErrorHandling(url, options = {}) {
    try {
        // Timeout pentru cereri
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            let errorMessage = 'A apărut o eroare în comunicația cu serverul.';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                errorMessage = `Eroare ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }

        return await response.json();
    } catch (error) {
        console.error('Eroare de rețea:', error);
        
        if (error.name === 'AbortError') {
            throw new Error('Cererea a expirat. Încercați din nou.');
        }
        
        notifications.show(error.message || 'A apărut o eroare în comunicația cu serverul.', 'error');
        throw error;
    }
}

// Cache management optimizat
class CacheManager {
    constructor(maxSize = MAX_CACHE_SIZE) {
        this.cache = new Map();
        this.maxSize = maxSize;
    }

    set(key, value) {
        if (this.cache.size >= this.maxSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, {
            value,
            timestamp: Date.now()
        });
    }

    get(key) {
        const item = this.cache.get(key);
        if (!item) return null;

        // Cache expiry de 10 minute
        if (Date.now() - item.timestamp > 600000) {
            this.cache.delete(key);
            return null;
        }

        return item.value;
    }

    clear() {
        this.cache.clear();
    }
}

const cache = new CacheManager();

// Funcții pentru gestionarea colecțiilor
function populateCollectionsList(collections) {
    const collectionsList = document.getElementById('collectionsList');
    collectionsList.innerHTML = '';
    
    if (collections.length === 0) {
        collectionsList.innerHTML = `
            <li class="list-group-item text-center text-muted">
                <i class="bi bi-folder-x"></i> Nu există colecții. Creați una nouă.
            </li>
        `;
        return;
    }
    
    collections.forEach(collection => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center';
        li.innerHTML = `
            <span class="collection-name">
                <i class="bi bi-folder"></i> ${collection}
            </span>
            <div class="actions">
                <button class="btn btn-sm btn-danger delete-collection" 
                        data-collection="${collection}" 
                        title="Șterge colecția"
                        aria-label="Șterge colecția ${collection}">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        `;
        
        // Event listener pentru selectarea colecției
        li.querySelector('.collection-name').addEventListener('click', () => {
            selectCollection(collection);
        });
        
        // Event listener pentru ștergerea colecției
        li.querySelector('.delete-collection').addEventListener('click', (e) => {
            e.stopPropagation();
            showDeleteConfirmation('collection', collection);
        });
        
        collectionsList.appendChild(li);
    });
}

async function loadCollections() {
    try {
        showLoading('.sidebar');
        
        const cacheKey = 'collections_list';
        let collections = cache.get(cacheKey);
        
        if (!collections) {
            collections = await fetchWithErrorHandling(`${API_BASE_URL}/collections`);
            cache.set(cacheKey, collections);
        }
        
        populateCollectionsList(collections);
    } catch (error) {
        console.log('Serverul nu este disponibil momentan.');
        populateCollectionsList([]);
    } finally {
        hideLoading('.sidebar');
    }
}

async function createCollection(name) {
    try {
        // Validare avansată
        if (!name || !/^[a-zA-Z0-9_]{1,50}$/.test(name)) {
            notifications.show('Numele colecției trebuie să conțină doar litere, cifre și underscore (_), maximum 50 caractere.', 'error');
            return false;
        }
        
        // Verificăm dacă colecția există deja
        const collections = await fetchWithErrorHandling(`${API_BASE_URL}/collections`);
        
        if (collections.includes(name)) {
            notifications.show(`Colecția "${name}" există deja.`, 'error');
            return false;
        }
        
        // Creăm colecția
        await fetchWithErrorHandling(`${API_BASE_URL}/collections/${name}`, {
            method: 'POST'
        });
        
        notifications.show(`Colecția "${name}" a fost creată cu succes.`, 'success');
        
        // Invalidăm cache-ul și reîncărcăm
        cache.clear();
        await loadCollections();
        selectCollection(name);
        
        return true;
    } catch (error) {
        console.error('Eroare la crearea colecției:', error);
        return false;
    }
}

async function deleteCollection(name) {
    try {
        showLoading('.main-content');
        
        await fetchWithErrorHandling(`${API_BASE_URL}/collections/${name}`, {
            method: 'DELETE'
        });
        
        notifications.show(`Colecția "${name}" a fost ștearsă cu succes.`, 'success');
        
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

function selectCollection(name) {
    currentCollection = name;
    updateUI();
    
    // Marchează colecția selectată
    document.querySelectorAll('#collectionsList .list-group-item').forEach(item => {
        item.classList.remove('active');
        const collectionSpan = item.querySelector('.collection-name');
        if (collectionSpan && collectionSpan.textContent.includes(name)) {
            item.classList.add('active');
        }
    });
    
    loadDocuments(name);
}

function updateUI() {
    const currentCollectionEl = document.getElementById('currentCollection');
    const uploadBtn = document.getElementById('uploadDocumentBtn');
    const queryBtn = document.getElementById('runQueryBtn');
    
    if (currentCollection) {
        currentCollectionEl.innerHTML = `<i class="bi bi-folder-open"></i> Colecția: ${currentCollection}`;
        uploadBtn.disabled = false;
        queryBtn.disabled = false;
    } else {
        currentCollectionEl.textContent = 'Selectați o colecție';
        uploadBtn.disabled = true;
        queryBtn.disabled = true;
    }
}

// Funcții pentru gestionarea documentelor JSON
async function loadDocuments(collectionName) {
    try {
        showLoading('.main-content');
        
        const cacheKey = `documents_${collectionName}`;
        let documents = cache.get(cacheKey);
        
        if (!documents) {
            documents = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${collectionName}/documents`);
            cache.set(cacheKey, documents);
        }
        
        displayDocuments(documents);
        
        // Salvăm pentru uz global
        window.documentsList = documents;
        
    } catch (error) {
        console.error('Eroare la încărcarea documentelor:', error);
        displayDocumentsError();
    } finally {
        hideLoading('.main-content');
    }
}

function displayDocuments(documents) {
    const documentsList = document.getElementById('documentsList');
    
    if (documents.length === 0) {
        documentsList.innerHTML = `
            <tr>
                <td colspan="4" class="text-center text-muted py-4">
                    <i class="bi bi-file-earmark-x fs-1"></i>
                    <div class="mt-2">Nu există fișiere JSON în această colecție.</div>
                    <div><small>Încărcați fișiere JSON chunkizate pentru a începe.</small></div>
                </td>
            </tr>
        `;
        return;
    }
    
    let html = '';
    documents.forEach(file => {
        const chunkCount = file.doc_count || 0;
        const badgeClass = chunkCount > 10 ? 'bg-success' : chunkCount > 5 ? 'bg-warning' : 'bg-info';
        
        html += `
        <tr>
            <td>
                <div class="d-flex align-items-center">
                    <i class="bi bi-file-earmark-code text-primary me-2"></i> 
                    <div>
                        <strong>${escapeHtml(file.source || 'Necunoscut')}</strong>
                        <br><small class="text-muted">Fișier JSON chunkizat</small>
                    </div>
                </div>
            </td>
            <td>
                <span class="badge ${badgeClass}">${chunkCount} chunk-uri</span>
                <br><small class="text-muted">Indexate pentru căutare</small>
            </td>
            <td>
                <i class="bi bi-calendar3"></i> ${escapeHtml(file.created_at || '-')}
                <br><small class="text-muted">Data procesării</small>
            </td>
            <td>
                <button class="btn btn-sm btn-outline-danger delete-document" 
                        data-source="${escapeHtml(file.source)}" 
                        title="Șterge fișierul JSON"
                        aria-label="Șterge fișierul ${escapeHtml(file.source)}">
                    <i class="bi bi-trash"></i> Șterge
                </button>
            </td>
        </tr>
        `;
    });
    
    documentsList.innerHTML = html;
    
    // Adăugăm event listeners pentru ștergere
    document.querySelectorAll('.delete-document').forEach(button => {
        button.addEventListener('click', () => {
            const source = button.getAttribute('data-source');
            showDeleteConfirmation('document', source);
        });
    });
}

function displayDocumentsError() {
    const documentsList = document.getElementById('documentsList');
    documentsList.innerHTML = `
        <tr>
            <td colspan="4" class="text-center text-danger py-4">
                <i class="bi bi-exclamation-triangle fs-1"></i>
                <div class="mt-2">Eroare la încărcarea documentelor.</div>
                <div><small>Verificați conexiunea la server.</small></div>
            </td>
        </tr>
    `;
}

// Validare avansată fișiere JSON
async function validateJsonFile(file) {
    return new Promise((resolve, reject) => {
        // Validări preliminare
        if (!file.name.toLowerCase().endsWith('.json')) {
            reject(new Error('Fișierul trebuie să aibă extensia .json'));
            return;
        }
        
        if (file.size > 50 * 1024 * 1024) { // 50MB
            reject(new Error('Fișierul este prea mare. Dimensiunea maximă este 50MB.'));
            return;
        }
        
        if (file.size < 100) { // Minim 100 bytes
            reject(new Error('Fișierul este prea mic pentru a conține JSON valid.'));
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const jsonData = JSON.parse(e.target.result);
                
                if (typeof jsonData !== 'object' || Array.isArray(jsonData)) {
                    reject(new Error('JSON-ul trebuie să fie un obiect, nu o listă.'));
                    return;
                }
                
                const chunkKeys = Object.keys(jsonData).filter(key => key.startsWith('chunk_'));
                
                if (chunkKeys.length === 0) {
                    reject(new Error('Nu s-au găsit chunk-uri în formatul așteptat (chunk_0, chunk_1, etc.).'));
                    return;
                }
                
                // Validăm primele chunk-uri
                let validChunks = 0;
                for (const key of chunkKeys.slice(0, 5)) {
                    const chunk = jsonData[key];
                    if (typeof chunk === 'object' && chunk.metadata && chunk.chunk) {
                        if (typeof chunk.chunk === 'string' && chunk.chunk.trim().length >= 10) {
                            validChunks++;
                        }
                    }
                }
                
                if (validChunks === 0) {
                    reject(new Error('Nu s-au găsit chunk-uri valide cu structura corectă.'));
                    return;
                }
                
                resolve({
                    isValid: true,
                    chunksCount: chunkKeys.length,
                    validChunks: validChunks,
                    sampleChunk: jsonData[chunkKeys[0]]
                });
                
            } catch (error) {
                reject(new Error(`JSON invalid: ${error.message}`));
            }
        };
        
        reader.onerror = () => reject(new Error('Eroare la citirea fișierului.'));
        reader.readAsText(file);
    });
}

async function uploadDocument(collectionName, file) {
    try {
        if (!file) {
            notifications.show('Selectați un fișier pentru încărcare.', 'error');
            return false;
        }
        
        // Validare avansată
        const validation = await validateJsonFile(file);
        notifications.show(
            `Fișier valid: ${validation.chunksCount} chunk-uri detectate, ${validation.validChunks} valide.`, 
            'success'
        );
        
        showLoading('.main-content');
        
        // Progress tracking
        const progressBar = document.getElementById('uploadProgress');
        const progressText = document.getElementById('uploadProgressText');
        progressBar.classList.remove('d-none');
        
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 20;
            if (progress > 90) progress = 90;
            progressBar.querySelector('.progress-bar').style.width = `${progress}%`;
            progressText.textContent = `${Math.round(progress)}%`;
        }, 300);
        
        // Upload fișier
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/collections/${collectionName}/upload`, {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        progressBar.querySelector('.progress-bar').style.width = '100%';
        progressText.textContent = '100%';
        
        if (response.ok) {
            const result = await response.json();
            notifications.show(
                `Fișierul "${file.name}" a fost procesat cu succes! ${result.chunks_count || 'Multiple'} chunk-uri indexate.`,
                'success'
            );
            
            // Invalidăm cache-ul pentru această colecție
            cache.clear();
            
            return true;
        } else {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Eroare la încărcarea fișierului');
        }
        
    } catch (error) {
        console.error('Eroare la încărcarea documentului:', error);
        notifications.show(`Eroare la încărcarea fișierului: ${error.message}`, 'error');
        return false;
    } finally {
        hideLoading('.main-content');
        setTimeout(() => {
            document.getElementById('uploadProgress').classList.add('d-none');
        }, 1000);
    }
}

async function deleteDocument(collectionName, source) {
    try {
        showLoading('.main-content');
        
        await fetchWithErrorHandling(`${API_BASE_URL}/collections/${collectionName}/documents`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ source: source })
        });
        
        notifications.show(`Fișierul "${source}" a fost șters cu succes.`, 'success');
        
        // Invalidăm cache-ul și reîncărcăm documentele
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

// Funcții pentru interogare optimizată
async function handleQuery() {
    const queryText = document.getElementById('queryInput').value.trim();
    const topK = parseInt(document.getElementById('topK').value) || 5;
    const temperature = parseFloat(document.getElementById('temperature').value) || 0.2;
    
    // Validări
    if (!queryText) {
        notifications.show('Introduceți o întrebare pentru interogare.', 'error');
        return;
    }
    
    if (!currentCollection) {
        notifications.show('Selectați o colecție pentru interogare.', 'error');
        return;
    }
    
    if (topK < 1 || topK > 20) {
        notifications.show('Numărul de chunk-uri trebuie să fie între 1 și 20.', 'warning');
        return;
    }
    
    if (temperature < 0 || temperature > 1) {
        notifications.show('Temperatura trebuie să fie între 0 și 1.', 'warning');
        return;
    }
    
    // Verificăm cache-ul
    const cacheKey = `query_${currentCollection}_${queryText}_${topK}_${temperature}`;
    let cachedResult = cache.get(cacheKey);
    
    if (cachedResult) {
        displayGeneratedResults(cachedResult, queryText);
        notifications.show('Rezultat din cache - răspuns instant!', 'info');
        return;
    }
    
    showLoading('#query');
    
    try {
        const resultsContainer = document.getElementById('queryResults');
        resultsContainer.innerHTML = `
            <div class="alert alert-info">
                <div class="d-flex align-items-center">
                    <div class="spinner-border spinner-border-sm me-2" role="status"></div>
                    <div>
                        <strong>Se procesează interogarea...</strong><br>
                        <small>Căutare în chunk-urile JSON și generare răspuns cu AI</small>
                    </div>
                </div>
            </div>
        `;
        
        const response = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${currentCollection}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: queryText,
                temperature: temperature,
                top_k_docs: topK
            })
        });
        
        // Salvăm în cache
        cache.set(cacheKey, response);
        
        displayGeneratedResults(response, queryText);
        
    } catch (error) {
        console.error('Eroare la procesarea interogării:', error);
        const resultsContainer = document.getElementById('queryResults');
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> 
                <strong>Eroare la procesarea interogării</strong><br>
                <small>${error.message}</small>
            </div>
        `;
    } finally {
        hideLoading('#query');
    }
}

function displayGeneratedResults(data, queryText) {
    const resultsContainer = document.getElementById('queryResults');
    resultsContainer.innerHTML = '';
    
    if (!data || !data.answer) {
        resultsContainer.innerHTML = `
            <div class="alert alert-warning">
                <i class="bi bi-exclamation-circle"></i> 
                <strong>Nu s-a putut genera un răspuns</strong><br>
                <small>Nu s-au găsit informații relevante pentru această interogare.</small>
            </div>
        `;
        
        if (data && data.documents && data.documents.length > 0) {
            displayQueryResults(data.documents, true);
        }
        return;
    }
    
    // Container pentru răspunsul generat
    const answerContainer = document.createElement('div');
    answerContainer.className = 'generated-answer mb-4';
    answerContainer.innerHTML = `
        <div class="question-title">
            <i class="bi bi-question-circle text-primary"></i> 
            <strong>Întrebare:</strong> ${escapeHtml(queryText)}
        </div>
        <div class="answer-content mt-3">
            <div class="d-flex align-items-center mb-2">
                <i class="bi bi-robot text-success fs-5 me-2"></i> 
                <strong>Răspuns generat de AI:</strong>
            </div>
            <div class="answer-text">${formatAnswer(data.answer)}</div>
        </div>
    `;
    
    resultsContainer.appendChild(answerContainer);
    
    // Separator
    const separator = document.createElement('hr');
    separator.className = 'my-4';
    resultsContainer.appendChild(separator);
    
    // Chunk-uri sursă
    if (data.documents && data.documents.length > 0) {
        const sourcesTitle = document.createElement('h5');
        sourcesTitle.className = 'mt-3 mb-3';
        sourcesTitle.innerHTML = `
            <i class="bi bi-file-text text-info"></i> 
            Chunk-uri JSON utilizate pentru generarea răspunsului:
        `;
        resultsContainer.appendChild(sourcesTitle);
        
        displayQueryResults(data.documents, true);
    }
}

function displayQueryResults(results, isSourceDocuments = false) {
    const resultsContainer = document.getElementById('queryResults');
    
    if (!isSourceDocuments) {
        resultsContainer.innerHTML = '';
    }
    
    if (!results || results.length === 0) {
        const noResults = document.createElement('div');
        noResults.className = 'alert alert-info';
        noResults.innerHTML = `
            <i class="bi bi-info-circle"></i> 
            Nu s-au găsit chunk-uri relevante pentru această interogare.
        `;
        resultsContainer.appendChild(noResults);
        return;
    }
    
    const resultsArray = Array.isArray(results) ? results : [results];
    
    resultsArray.forEach((result, index) => {
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        
        const score = result.score ? (result.score * 100).toFixed(1) : 'N/A';
        const source = result.meta?.original_source || 'Necunoscută';
        const chunkId = result.meta?.chunk_id || `chunk_${index}`;
        const matchType = result.match_type || 'semantic';
        
        const matchTypeConfig = {
            exact: { icon: 'bi-bullseye', color: 'text-success', label: 'Potrivire exactă' },
            keyword: { icon: 'bi-key', color: 'text-warning', label: 'Cuvinte cheie' },
            semantic: { icon: 'bi-search', color: 'text-info', label: 'Semantică' }
        };
        
        const config = matchTypeConfig[matchType] || matchTypeConfig.semantic;
        
        resultCard.innerHTML = `
            <div class="result-header d-flex justify-content-between align-items-center">
                <div class="d-flex align-items-center">
                    <i class="bi bi-file-earmark-code text-primary me-2"></i>
                    <span class="result-number fw-bold">${escapeHtml(chunkId)}</span>
                    <span class="ms-2 badge bg-secondary">${config.label}</span>
                </div>
                <div class="d-flex align-items-center">
                    <i class="bi ${config.icon} ${config.color} me-1"></i>
                    <span class="result-score">Relevanță: ${score}%</span>
                </div>
            </div>
            <div class="result-content mt-3">
                <p class="mb-0">${formatContent(result.content)}</p>
            </div>
            <div class="result-meta mt-3 pt-2 border-top">
                <small class="text-muted">
                    <i class="bi bi-file-text"></i> <strong>Sursă:</strong> ${escapeHtml(source)}
                </small>
            </div>
        `;
        
        resultsContainer.appendChild(resultCard);
    });
}

// Funcții helper pentru formatare și siguranță
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

function formatAnswer(answer) {
    return escapeHtml(answer)
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

function formatContent(content) {
    const escaped = escapeHtml(content);
    if (escaped.length > 500) {
        return escaped.substring(0, 500) + '...';
    }
    return escaped;
}

// Funcție pentru confirmarea ștergerii
function showDeleteConfirmation(type, name) {
    const modal = new bootstrap.Modal(document.getElementById('deleteConfirmModal'));
    const confirmText = document.getElementById('deleteConfirmText');
    const confirmBtn = document.getElementById('confirmDeleteBtn');
    
    if (type === 'collection') {
        confirmText.innerHTML = `
            Ești sigur că vrei să ștergi colecția <strong>"${escapeHtml(name)}"</strong>?<br>
            <small class="text-muted">Toate fișierele JSON și chunk-urile din această colecție vor fi șterse definitiv.</small>
        `;
        confirmBtn.onclick = async () => {
            modal.hide();
            await deleteCollection(name);
        };
    } else if (type === 'document') {
        confirmText.innerHTML = `
            Ești sigur că vrei să ștergi fișierul JSON <strong>"${escapeHtml(name)}"</strong> din colecția <strong>"${escapeHtml(currentCollection)}"</strong>?<br>
            <small class="text-muted">Toate chunk-urile din acest fișier vor fi șterse definitiv.</small>
        `;
        confirmBtn.onclick = async () => {
            modal.hide();
            await deleteDocument(currentCollection, name);
        };
    }
    
    modal.show();
}

// Funcții pentru căutarea în colecții cu debouncing
const searchCollections = debounce((searchTerm) => {
    const items = document.querySelectorAll('#collectionsList .list-group-item');
    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        const isVisible = text.includes(searchTerm.toLowerCase());
        item.style.display = isVisible ? '' : 'none';
    });
}, 300);

// Funcții avansate pentru gestionarea fișierelor
function getFileInfo(file) {
    return {
        name: file.name,
        size: file.size,
        sizeFormatted: formatFileSize(file.size),
        type: file.type,
        lastModified: new Date(file.lastModified).toLocaleString('ro-RO')
    };
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Funcții pentru keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl+N pentru colecție nouă
        if (e.ctrlKey && e.key === 'n') {
            e.preventDefault();
            document.getElementById('createCollectionBtn').click();
        }
        
        // Ctrl+U pentru upload
        if (e.ctrlKey && e.key === 'u' && currentCollection) {
            e.preventDefault();
            document.getElementById('uploadDocumentBtn').click();
        }
        
        // Ctrl+Q pentru focus pe query input
        if (e.ctrlKey && e.key === 'q') {
            e.preventDefault();
            document.getElementById('queryInput').focus();
        }
        
        // Escape pentru închidere modale
        if (e.key === 'Escape') {
            const activeModal = document.querySelector('.modal.show');
            if (activeModal) {
                const modal = bootstrap.Modal.getInstance(activeModal);
                if (modal) modal.hide();
            }
        }
    });
}

// Funcții pentru auto-save
function setupAutoSave() {
    ['topK', 'temperature'].forEach(id => {
        const element = document.getElementById(id);
        element.addEventListener('change', () => {
            localStorage.setItem(`rag_${id}`, element.value);
        });
        
        // Restore din localStorage
        const savedValue = localStorage.getItem(`rag_${id}`);
        if (savedValue) {
            element.value = savedValue;
        }
    });
    
    // Auto-save pentru query input
    const queryInput = document.getElementById('queryInput');
    const saveQuery = debounce((value) => {
        if (value.trim()) {
            localStorage.setItem('rag_last_query', value);
        }
    }, 1000);
    
    queryInput.addEventListener('input', (e) => {
        saveQuery(e.target.value);
    });
    
    // Restore ultima interogare
    const lastQuery = localStorage.getItem('rag_last_query');
    if (lastQuery) {
        queryInput.value = lastQuery;
    }
}

// Funcții pentru performance monitoring
function setupPerformanceMonitoring() {
    if ('performance' in window) {
        window.addEventListener('load', () => {
            const perfData = performance.getEntriesByType('navigation')[0];
            const loadTime = perfData.loadEventEnd - perfData.loadEventStart;
            console.log(`Pagina s-a încărcat în ${loadTime.toFixed(2)}ms`);
            
            // Raportăm dacă încărcarea este lentă
            if (loadTime > 3000) {
                console.warn('Încărcare lentă detectată');
            }
        });
    }
}

// Funcții pentru drag & drop
function setupDragAndDrop() {
    const uploadArea = document.querySelector('.tab-content');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('drag-over');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('drag-over');
    }
    
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0 && currentCollection) {
            const file = files[0];
            if (file.name.toLowerCase().endsWith('.json')) {
                document.getElementById('documentFile').files = files;
                notifications.show(`Fișier detectat: ${file.name}`, 'info');
            } else {
                notifications.show('Doar fișierele JSON sunt acceptate!', 'error');
            }
        }
    }
}

// Event listeners optimizați
document.addEventListener('DOMContentLoaded', () => {
    // Inițializări
    loadCollections();
    setupKeyboardShortcuts();
    setupAutoSave();
    setupPerformanceMonitoring();
    setupDragAndDrop();
    
    // Crearea colecțiilor
    const saveCollectionBtn = document.getElementById('saveCollectionBtn');
    saveCollectionBtn.addEventListener('click', async () => {
        const collectionName = document.getElementById('collectionName').value.trim();
        const success = await createCollection(collectionName);
        if (success) {
            bootstrap.Modal.getInstance(document.getElementById('createCollectionModal')).hide();
            document.getElementById('collectionName').value = '';
        }
    });
    
    // Upload documente JSON
    const startUploadBtn = document.getElementById('startUploadBtn');
    startUploadBtn.addEventListener('click', async () => {
        const fileInput = document.getElementById('documentFile');
        if (!fileInput.files || fileInput.files.length === 0) {
            notifications.show('Selectați un fișier JSON pentru încărcare.', 'error');
            return;
        }
        
        if (!currentCollection) {
            notifications.show('Selectați o colecție pentru încărcare.', 'error');
            return;
        }
        
        const file = fileInput.files[0];
        
        try {
            await validateJsonFile(file);
        } catch (validationError) {
            notifications.show(`Format JSON invalid: ${validationError.message}`, 'error');
            return;
        }
        
        const success = await uploadDocument(currentCollection, file);
        if (success) {
            bootstrap.Modal.getInstance(document.getElementById('uploadDocumentModal')).hide();
            fileInput.value = '';
            await loadDocuments(currentCollection);
        }
    });
    
    // Interogare
    document.getElementById('runQueryBtn').addEventListener('click', handleQuery);
    
    // Căutare în colecții
    document.getElementById('searchCollections').addEventListener('input', (e) => {
        searchCollections(e.target.value);
    });

    // Enter pentru interogare (Ctrl+Enter)
    document.getElementById('queryInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleQuery();
        }
    });
    
    // Validare fișiere la selecție
    document.getElementById('documentFile').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && !file.name.toLowerCase().endsWith('.json')) {
            notifications.show('Selectați doar fișiere JSON cu extensia .json', 'error');
            e.target.value = '';
        } else if (file) {
            const fileInfo = getFileInfo(file);
            notifications.show(`Fișier selectat: ${fileInfo.name} (${fileInfo.sizeFormatted})`, 'info');
        }
    });
    
    // Validare în timp real pentru numele colecției
    document.getElementById('collectionName').addEventListener('input', (e) => {
        const value = e.target.value;
        const isValid = /^[a-zA-Z0-9_]*$/.test(value);
        const saveBtn = document.getElementById('saveCollectionBtn');
        
        if (value && !isValid) {
            e.target.classList.add('is-invalid');
            saveBtn.disabled = true;
        } else if (value.length > 0) {
            e.target.classList.remove('is-invalid');
            saveBtn.disabled = false;
        } else {
            e.target.classList.remove('is-invalid');
            saveBtn.disabled = true;
        }
    });
    
    // Cleanup la închiderea paginii
    window.addEventListener('beforeunload', () => {
        cache.clear();
    });
    
    // Tooltips pentru accesibilitate
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Service Worker pentru caching (opțional)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js').catch(() => {
            console.log('Service Worker nu este disponibil');
        });
    }
    
    // Detectare conexiune internet
    window.addEventListener('online', () => {
        notifications.show('Conexiunea la internet a fost restabilită.', 'success');
    });
    
    window.addEventListener('offline', () => {
        notifications.show('Conexiunea la internet s-a pierdut. Unele funcții pot fi indisponibile.', 'warning');
    });
    
    // Gestionare schimbare tab
    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', (e) => {
            const targetTab = e.target.getAttribute('data-bs-target');
            localStorage.setItem('rag_active_tab', targetTab);
        });
    });
    
    // Restore tab activ
    const activeTab = localStorage.getItem('rag_active_tab');
    if (activeTab) {
        const tabElement = document.querySelector(`[data-bs-target="${activeTab}"]`);
        if (tabElement) {
            const tab = new bootstrap.Tab(tabElement);
            tab.show();
        }
    }
});

// Export funcții pentru debugging
window.RAG_DEBUG = {
    cache,
    notifications,
    currentCollection: () => currentCollection,
    clearCache: () => cache.clear(),
    showNotification: (msg, type) => notifications.show(msg, type),
    getFileInfo,
    formatFileSize,
    escapeHtml,
    validateJsonFile
};