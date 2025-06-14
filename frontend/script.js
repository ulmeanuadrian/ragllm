// script.js - MODIFICAT PENTRU DOAR JSON CHUNKIZAT

// Configurare API
const API_BASE_URL = 'http://localhost:8070';

// Variabile globale
let currentCollection = null;
let documentsList = [];

// Funcții utilitare
function showLoading(selector) {
    const element = document.querySelector(selector);
    element.classList.add('loading');
    
    // Verificăm dacă overlay-ul de încărcare există deja
    if (!element.querySelector('.loading-overlay')) {
        // Creăm overlay-ul de încărcare
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        
        // Adăugăm spinner-ul
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        overlay.appendChild(spinner);
        
        // Adăugăm textul "Se încarcă..."
        const loadingText = document.createElement('div');
        loadingText.className = 'loading-text';
        loadingText.textContent = 'Se procesează JSON...';
        loadingText.style.marginLeft = '10px';
        loadingText.style.fontWeight = 'bold';
        overlay.appendChild(loadingText);
        
        // Adăugăm overlay-ul în element
        element.appendChild(overlay);
    } else {
        // Dacă există deja, îl facem vizibil
        element.querySelector('.loading-overlay').style.display = 'flex';
    }
}

function hideLoading(selector) {
    const element = document.querySelector(selector);
    element.classList.remove('loading');
    
    // Ascundem overlay-ul de încărcare dacă există
    const overlay = element.querySelector('.loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function showToast(message, type = 'success') {
    // Creăm un element de tip toast pentru notificări mai elegante
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    // Icon pentru tipul de toast
    let icon = 'bi-check-circle';
    if (type === 'error') icon = 'bi-exclamation-triangle';
    if (type === 'warning') icon = 'bi-exclamation-circle';
    
    toast.innerHTML = `
        <div class="toast-content">
            <div>
                <i class="bi ${icon}"></i>
                <span>${message}</span>
            </div>
            <button class="close-toast">&times;</button>
        </div>
    `;
    
    // Adăugăm toast-ul în pagina
    document.body.appendChild(toast);
    
    // Afișăm toast-ul
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    // Adăugăm event listener pentru butonul de închidere
    toast.querySelector('.close-toast').addEventListener('click', () => {
        toast.classList.remove('show');
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 300);
    });
    
    // Ascundem toast-ul automat după 5 secunde
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 300);
    }, 5000);
}

// Funcție pentru gestionarea erorilor de rețea
async function fetchWithErrorHandling(url, options = {}) {
    try {
        const response = await fetch(url, options);
        
        // Verificăm dacă răspunsul este ok (status 200-299)
        if (!response.ok) {
            // Încercăm să extragem mesajul de eroare din răspuns
            let errorMessage = 'A apărut o eroare în comunicația cu serverul.';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                // Dacă nu putem extrage JSON, folosim mesajul de stare HTTP
                errorMessage = `Eroare ${response.status}: ${response.statusText}`;
            }
            
            // Aruncăm o eroare cu mesajul extras
            throw new Error(errorMessage);
        }
        
        // Dacă răspunsul este ok, returnam datele
        return await response.json();
    } catch (error) {
        // Gestionăm erorile de rețea și alte erori
        console.error('Eroare de rețea:', error);
        
        // Afișăm o notificare de eroare utilizatorului
        showToast(error.message || 'A apărut o eroare în comunicația cu serverul. Încercați din nou.', 'error');
        
        // Propagăm eroarea pentru a fi gestionată în funcția apelantă
        throw error;
    }
}

// Funcții pentru gestionarea colecțiilor
function populateCollectionsList(collections) {
    const collectionsList = document.getElementById('collectionsList');
    collectionsList.innerHTML = '';
    
    if (collections.length === 0) {
        collectionsList.innerHTML = '<li class="list-group-item text-center text-muted"><i class="bi bi-folder-x"></i> Nu există colecții. Creați una nouă.</li>';
        return;
    }
    
    collections.forEach(collection => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center';
        li.innerHTML = `
            <span><i class="bi bi-folder"></i> ${collection}</span>
            <div class="actions">
                <button class="btn btn-sm btn-danger delete-collection" data-collection="${collection}" title="Șterge colecția">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        `;
        li.addEventListener('click', (e) => {
            // Nu selectăm colecția dacă s-a făcut click pe butonul de ștergere
            if (!e.target.closest('.delete-collection')) {
                selectCollection(collection);
            }
        });
        collectionsList.appendChild(li);
    });
    
    // Adaugă event listeners pentru butoanele de ștergere
    document.querySelectorAll('.delete-collection').forEach(button => {
        button.addEventListener('click', (e) => {
            e.stopPropagation(); // Previne selectarea colecției
            const collection = button.getAttribute('data-collection');
            showDeleteConfirmation('collection', collection);
        });
    });
}

async function loadCollections() {
    try {
        showLoading('.sidebar');
        
        // Folosim noua funcție de gestionare a erorilor
        const collections = await fetchWithErrorHandling(`${API_BASE_URL}/collections`);
        populateCollectionsList(collections);
    } catch (error) {
        // În caz de eroare, afișăm o listă goală
        console.log('Serverul nu este disponibil momentan. Se afișează interfața fără colecții.');
        populateCollectionsList([]);
    } finally {
        hideLoading('.sidebar');
    }
}

async function createCollection(name) {
    try {
        // Verificăm dacă numele colecției este valid
        if (!name || !/^[a-zA-Z0-9_]+$/.test(name)) {
            showToast('Numele colecției trebuie să conțină doar litere, cifre și underscore (_).', 'error');
            return false;
        }
        
        // Verificăm dacă colecția există deja folosind fetch
        const response = await fetch(`${API_BASE_URL}/collections`);
        let collections = [];
        
        if (response.ok) {
            collections = await response.json();
        } else {
            console.warn('Eroare HTTP:', response.status);
        }
        
        if (collections.includes(name)) {
            showToast(`Colecția "${name}" există deja.`, 'error');
            return false;
        }
        
        // Creăm colecția folosind noul endpoint
        const createResponse = await fetch(`${API_BASE_URL}/collections/${name}`, {
            method: 'POST'
        });
        
        if (createResponse.ok) {
            const result = await createResponse.json();
            showToast(`Colecția "${name}" a fost creată cu succes. Încărcați fișiere JSON pentru a o popula.`);
            await loadCollections();
            selectCollection(name);
        } else {
            showToast(`Eroare la crearea colecției "${name}".`, 'error');
            return false;
        }
        return true;
    } catch (error) {
        console.error('Eroare la crearea colecției:', error);
        showToast('Eroare la crearea colecției. Verificați consola pentru detalii.', 'error');
        return false;
    }
}

async function deleteCollection(name) {
    try {
        showLoading('.main-content');
        
        // Folosim fetch direct deoarece CORS este configurat corect în backend
        const response = await fetch(`${API_BASE_URL}/collections/${name}`, {
            method: 'DELETE'
        });
        
        const success = response.ok;
        
        if (success) {
            showToast(`Colecția "${name}" a fost ștearsă cu succes.`);
            if (currentCollection === name) {
                currentCollection = null;
                document.getElementById('currentCollection').textContent = 'Selectați o colecție';
                document.getElementById('uploadDocumentBtn').disabled = true;
                document.getElementById('runQueryBtn').disabled = true;
                document.getElementById('documentsList').innerHTML = '';
            }
            await loadCollections();
            return true;
        } else {
            throw new Error('Eroare la ștergerea colecției');
        }
    } catch (error) {
        console.error('Eroare la ștergerea colecției:', error);
        showToast(`Eroare la ștergerea colecției: ${error.message}`, 'error');
        return false;
    } finally {
        hideLoading('.main-content');
    }
}

function selectCollection(name) {
    currentCollection = name;
    document.getElementById('currentCollection').innerHTML = `<i class="bi bi-folder-open"></i> Colecția: ${name}`;
    document.getElementById('uploadDocumentBtn').disabled = false;
    document.getElementById('runQueryBtn').disabled = false;
    
    // Marchează colecția selectată în listă
    document.querySelectorAll('#collectionsList .list-group-item').forEach(item => {
        const collectionSpan = item.querySelector('span');
        if (collectionSpan && collectionSpan.textContent.trim().includes(name)) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Încarcă documentele din colecție
    loadDocuments(name);
}

// Funcții pentru gestionarea documentelor JSON
async function loadDocuments(collectionName) {
    try {
        showLoading('.main-content');
        
        // Folosim endpoint-ul actualizat care grupează documentele după fișierul sursă
        const documents = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${collectionName}/documents`);
        
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
        
        // Salvăm documentele în variabila globală
        window.documentsList = documents;
        
        // Construim tabelul cu fișierele originale
        let html = '';
        documents.forEach(file => {
            const chunkCount = file.doc_count || 0;
            const badgeClass = chunkCount > 10 ? 'bg-success' : chunkCount > 5 ? 'bg-warning' : 'bg-info';
            
            html += `
            <tr>
                <td>
                    <i class="bi bi-file-earmark-code text-primary"></i> 
                    <strong>${file.source || 'Necunoscut'}</strong>
                    <br><small class="text-muted">Fișier JSON chunkizat</small>
                </td>
                <td>
                    <span class="badge ${badgeClass}">${chunkCount} chunk-uri</span>
                    <br><small class="text-muted">Indexate pentru căutare</small>
                </td>
                <td>
                    <i class="bi bi-calendar3"></i> ${file.created_at || '-'}
                    <br><small class="text-muted">Data procesării</small>
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-danger delete-document" data-source="${file.source}" title="Șterge fișierul JSON">
                        <i class="bi bi-trash"></i> Șterge
                    </button>
                </td>
            </tr>
            `;
        });
        
        documentsList.innerHTML = html;
        
        // Adăugăm event listeners pentru butoanele de ștergere
        document.querySelectorAll('.delete-document').forEach(button => {
            button.addEventListener('click', () => {
                const source = button.getAttribute('data-source');
                showDeleteConfirmation('document', source);
            });
        });
    } catch (error) {
        console.error('Eroare la încărcarea documentelor:', error);
        document.getElementById('documentsList').innerHTML = 
            `<tr><td colspan="4" class="text-center text-danger py-4">
                <i class="bi bi-exclamation-triangle fs-1"></i>
                <div class="mt-2">Eroare la încărcarea documentelor.</div>
                <div><small>Verificați consola pentru detalii.</small></div>
            </td></tr>`;
    } finally {
        hideLoading('.main-content');
    }
}

async function uploadDocument(collectionName, file) {
    try {
        if (!file) {
            showToast('Selectați un fișier pentru încărcare.', 'error');
            return false;
        }
        
        // VERIFICARE STRICTĂ - DOAR JSON
        if (!file.name.toLowerCase().endsWith('.json')) {
            showToast('Sunt acceptate DOAR fișiere JSON cu extensia .json!', 'error');
            return false;
        }
        
        // Verificăm tipul MIME (opțional)
        if (file.type && !['application/json', 'text/json', ''].includes(file.type)) {
            showToast('Tipul fișierului trebuie să fie JSON. Verificați dacă fișierul este un JSON valid.', 'warning');
        }
        
        // Verificare dimensiune fișier (max 10MB pentru JSON)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            showToast('Fișierul JSON este prea mare. Dimensiunea maximă acceptată este 10MB.', 'error');
            return false;
        }
        
        showLoading('.main-content');
        
        // Afișăm progress bar
        const progressBar = document.getElementById('uploadProgress');
        const progressText = document.getElementById('uploadProgressText');
        progressBar.classList.remove('d-none');
        
        // Simulăm progresul pentru feedback utilizator
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 30;
            if (progress > 90) progress = 90;
            progressBar.querySelector('.progress-bar').style.width = `${progress}%`;
            progressText.textContent = `${Math.round(progress)}%`;
        }, 200);
        
        // Folosim fetch direct pentru upload de fișiere
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/collections/${collectionName}/upload`, {
            method: 'POST',
            body: formData
        });
        
        // Completăm progress bar-ul
        clearInterval(progressInterval);
        progressBar.querySelector('.progress-bar').style.width = '100%';
        progressText.textContent = '100%';
        
        const success = response.ok;
        
        if (success) {
            const result = await response.json();
            const chunksCount = result.chunks_count || 'multiple';
            showToast(`Fișierul JSON "${file.name}" a fost procesat cu succes! ${chunksCount} chunk-uri au fost indexate.`);
            return true;
        } else {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Eroare la încărcarea fișierului JSON');
        }
    } catch (error) {
        console.error('Eroare la încărcarea documentului:', error);
        showToast(`Eroare la încărcarea fișierului JSON: ${error.message}`, 'error');
        return false;
    } finally {
        hideLoading('.main-content');
        // Ascundem progress bar-ul
        setTimeout(() => {
            document.getElementById('uploadProgress').classList.add('d-none');
        }, 1000);
    }
}

async function deleteDocument(collectionName, source) {
    try {
        showLoading('.main-content');
        
        // Folosim fetch direct deoarece CORS este configurat corect în backend
        const response = await fetch(`${API_BASE_URL}/collections/${collectionName}/documents`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ source: source })
        });
        
        const success = response.ok;
        
        if (success) {
            showToast(`Fișierul JSON "${source}" a fost șters cu succes.`);
            await loadDocuments(collectionName);
            return true;
        } else {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'Eroare la ștergerea fișierului JSON');
        }
    } catch (error) {
        console.error('Eroare la ștergerea documentului:', error);
        showToast(`Eroare la ștergerea fișierului: ${error.message}`, 'error');
        return false;
    } finally {
        hideLoading('.main-content');
    }
}

// Funcții pentru interogare și generare cu Gemini
async function handleQuery() {
    const queryText = document.getElementById('queryInput').value.trim();
    if (!queryText) {
        showToast('Introduceți o întrebare pentru interogare.', 'error');
        return;
    }
    
    if (!currentCollection) {
        showToast('Selectați o colecție pentru interogare.', 'error');
        return;
    }
    
    const topK = parseInt(document.getElementById('topK').value) || 5;
    const temperature = parseFloat(document.getElementById('temperature').value) || 0.2;
    
    // Validăm parametrii
    if (topK < 1 || topK > 20) {
        showToast('Numărul de chunk-uri trebuie să fie între 1 și 20.', 'warning');
        return;
    }
    
    if (temperature < 0 || temperature > 1) {
        showToast('Temperatura trebuie să fie între 0 și 1.', 'warning');
        return;
    }
    
    showLoading('#query');
    
    try {
        // Afișăm un mesaj că se procesează interogarea
        const resultsContainer = document.getElementById('queryResults');
        resultsContainer.innerHTML = `
            <div class="alert alert-info">
                <div class="d-flex align-items-center">
                    <div class="spinner-border spinner-border-sm me-2" role="status"></div>
                    <div>
                        <strong>Se procesează interogarea...</strong><br>
                        <small>Căutare în chunk-urile JSON și generare răspuns cu Gemini AI</small>
                    </div>
                </div>
            </div>
        `;
        
        // Generăm răspunsul direct cu Gemini
        try {
            const response = await fetch(`${API_BASE_URL}/collections/${currentCollection}/generate`, {
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
            
            if (response.ok) {
                const data = await response.json();
                displayGeneratedResults(data, queryText);
            } else {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Eroare la generarea răspunsului');
            }
        } catch (genError) {
            console.error('Eroare la generarea răspunsului:', genError);
            showToast(`Eroare la generarea răspunsului: ${genError.message}`, 'error');
            
            // Afișăm mesaj de eroare în containerul de rezultate
            resultsContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle"></i> 
                    <strong>Eroare la generarea răspunsului</strong><br>
                    <small>${genError.message}</small>
                </div>
            `;
        }
    } catch (error) {
        console.error('Eroare la procesarea interogării:', error);
        showToast('Eroare la procesarea interogării. Verificați consola pentru detalii.', 'error');
        
        // Resetăm containerul de rezultate
        const resultsContainer = document.getElementById('queryResults');
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-bug"></i> A apărut o eroare la procesarea interogării.
            </div>
        `;
    } finally {
        hideLoading('#query');
    }
}

function displayGeneratedResults(data, queryText) {
    const resultsContainer = document.getElementById('queryResults');
    resultsContainer.innerHTML = '';
    
    // Verificăm dacă avem un răspuns generat
    if (!data || !data.answer) {
        const noResults = document.createElement('div');
        noResults.className = 'alert alert-warning';
        noResults.innerHTML = `
            <i class="bi bi-exclamation-circle"></i> 
            <strong>Nu s-a putut genera un răspuns</strong><br>
            <small>Nu s-au găsit informații relevante pentru această interogare în chunk-urile JSON.</small>
        `;
        resultsContainer.appendChild(noResults);
        
        // Afișăm documentele dacă sunt disponibile
        if (data && data.documents && data.documents.length > 0) {
            displayQueryResults(data.documents);
        }
        return;
    }
    
    // Creăm containerul pentru răspunsul generat
    const answerContainer = document.createElement('div');
    answerContainer.className = 'generated-answer mb-4';
    
    // Adăugăm titlul și întrebarea
    const questionTitle = document.createElement('div');
    questionTitle.className = 'question-title';
    questionTitle.innerHTML = `
        <i class="bi bi-question-circle text-primary"></i> 
        <strong>Întrebare:</strong> ${queryText}
    `;
    answerContainer.appendChild(questionTitle);
    
    // Adăugăm răspunsul generat
    const answerContent = document.createElement('div');
    answerContent.className = 'answer-content mt-3 p-3 border rounded';
    answerContent.innerHTML = `
        <div class="d-flex align-items-center mb-2">
            <i class="bi bi-robot text-success fs-5 me-2"></i> 
            <strong>Răspuns generat de Gemini AI:</strong>
        </div>
        <div class="answer-text">${data.answer.replace(/\n/g, '<br>')}</div>
    `;
    answerContainer.appendChild(answerContent);
    
    // Adăugăm containerul de răspuns la rezultate
    resultsContainer.appendChild(answerContainer);
    
    // Adăugăm un separator
    const separator = document.createElement('hr');
    separator.className = 'my-4';
    resultsContainer.appendChild(separator);
    
    // Adăugăm titlul pentru documente sursă
    const sourcesTitle = document.createElement('h5');
    sourcesTitle.className = 'mt-3 mb-3';
    sourcesTitle.innerHTML = `
        <i class="bi bi-file-text text-info"></i> 
        Chunk-uri JSON utilizate pentru generarea răspunsului:
    `;
    resultsContainer.appendChild(sourcesTitle);
    
    // Afișăm documentele sursă
    if (data.documents && data.documents.length > 0) {
        displayQueryResults(data.documents, true);
    } else {
        const noSources = document.createElement('div');
        noSources.className = 'alert alert-info';
        noSources.innerHTML = `
            <i class="bi bi-info-circle"></i> 
            Nu sunt disponibile detalii despre chunk-urile sursă utilizate.
        `;
        resultsContainer.appendChild(noSources);
    }
}

function displayQueryResults(results, isSourceDocuments = false) {
    const resultsContainer = document.getElementById('queryResults');
    
    // Dacă nu sunt documente sursă, curățăm containerul
    if (!isSourceDocuments) {
        resultsContainer.innerHTML = '';
    }
    
    // Verificăm dacă avem rezultate și dacă sunt în formatul așteptat
    if (!results || (Array.isArray(results) && results.length === 0)) {
        const noResults = document.createElement('div');
        noResults.className = 'alert alert-info';
        noResults.innerHTML = `
            <i class="bi bi-info-circle"></i> 
            Nu s-au găsit chunk-uri relevante pentru această interogare.
        `;
        resultsContainer.appendChild(noResults);
        return;
    }
    
    // Convertim rezultatele în array dacă nu sunt deja
    const resultsArray = Array.isArray(results) ? results : [results];
    
    resultsArray.forEach((result, index) => {
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        
        const score = result.score ? (result.score * 100).toFixed(1) : 'N/A';
        const source = result.meta && result.meta.original_source ? result.meta.original_source : 'Necunoscută';
        const chunkId = result.meta && result.meta.chunk_id ? result.meta.chunk_id : `chunk_${index}`;
        const matchType = result.match_type || 'semantic';
        
        // Icon și culoare în funcție de tipul de potrivire
        let matchIcon = 'bi-search';
        let matchColor = 'text-info';
        if (matchType === 'exact') {
            matchIcon = 'bi-bullseye';
            matchColor = 'text-success';
        } else if (matchType === 'keyword') {
            matchIcon = 'bi-key';
            matchColor = 'text-warning';
        }
        
        resultCard.innerHTML = `
            <div class="result-header">
                <div class="d-flex align-items-center">
                    <i class="bi bi-file-earmark-code text-primary me-2"></i>
                    <span class="result-number fw-bold">${chunkId}</span>
                    <span class="ms-2 badge bg-secondary">${matchType}</span>
                </div>
                <div class="d-flex align-items-center">
                    <i class="bi ${matchIcon} ${matchColor} me-1"></i>
                    <span class="result-score">Relevanță: ${score}%</span>
                </div>
            </div>
            <div class="result-content mt-3">
                <p class="mb-0">${result.content}</p>
            </div>
            <div class="result-meta mt-3 pt-2 border-top">
                <small class="text-muted">
                    <i class="bi bi-file-text"></i> <strong>Sursă:</strong> ${source}
                </small>
            </div>
        `;
        
        resultsContainer.appendChild(resultCard);
    });
}

// Funcție pentru confirmarea ștergerii
function showDeleteConfirmation(type, name) {
    const modal = new bootstrap.Modal(document.getElementById('deleteConfirmModal'));
    const confirmText = document.getElementById('deleteConfirmText');
    const confirmBtn = document.getElementById('confirmDeleteBtn');
    
    if (type === 'collection') {
        confirmText.innerHTML = `
            Ești sigur că vrei să ștergi colecția <strong>"${name}"</strong>?<br>
            <small class="text-muted">Toate fișierele JSON și chunk-urile din această colecție vor fi șterse definitiv.</small>
        `;
        confirmBtn.onclick = async () => {
            modal.hide();
            await deleteCollection(name);
        };
    } else if (type === 'document') {
        confirmText.innerHTML = `
            Ești sigur că vrei să ștergi fișierul JSON <strong>"${name}"</strong> din colecția <strong>"${currentCollection}"</strong>?<br>
            <small class="text-muted">Toate chunk-urile din acest fișier vor fi șterse definitiv.</small>
        `;
        confirmBtn.onclick = async () => {
            modal.hide();
            await deleteDocument(currentCollection, name);
        };
    }
    
    modal.show();
}

// Funcție de validare pentru fișiere JSON înainte de upload
function validateJsonFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const jsonData = JSON.parse(e.target.result);
                
                // Verificăm dacă JSON-ul conține chunk-uri
                const chunkKeys = Object.keys(jsonData).filter(key => key.startsWith('chunk_'));
                
                if (chunkKeys.length === 0) {
                    reject(new Error('Fișierul JSON nu conține chunk-uri în formatul așteptat. Cheile trebuie să înceapă cu "chunk_".'));
                    return;
                }
                
                // Verificăm structura primului chunk
                const firstChunk = jsonData[chunkKeys[0]];
                if (!firstChunk || typeof firstChunk !== 'object') {
                    reject(new Error('Chunk-urile trebuie să fie obiecte JSON.'));
                    return;
                }
                
                if (!firstChunk.hasOwnProperty('metadata') || !firstChunk.hasOwnProperty('chunk')) {
                    reject(new Error('Fiecare chunk trebuie să conțină câmpurile "metadata" și "chunk".'));
                    return;
                }
                
                resolve({
                    isValid: true,
                    chunksCount: chunkKeys.length,
                    sampleChunk: firstChunk
                });
            } catch (error) {
                reject(new Error(`Fișierul nu este un JSON valid: ${error.message}`));
            }
        };
        reader.onerror = () => reject(new Error('Eroare la citirea fișierului.'));
        reader.readAsText(file);
    });
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Încarcă colecțiile la pornirea aplicației
    loadCollections();
    
    // Event listener pentru crearea unei colecții noi
    document.getElementById('saveCollectionBtn').addEventListener('click', async () => {
        const collectionName = document.getElementById('collectionName').value.trim();
        const success = await createCollection(collectionName);
        if (success) {
            bootstrap.Modal.getInstance(document.getElementById('createCollectionModal')).hide();
            document.getElementById('collectionName').value = '';
        }
    });
    
    // Event listener pentru încărcarea unui document JSON - adăugăm o verificare pentru a preveni dubla înregistrare
    const startUploadBtn = document.getElementById('startUploadBtn');
    
    // Eliminăm eventualele event listener-uri existente pentru a preveni duplicarea
    startUploadBtn.replaceWith(startUploadBtn.cloneNode(true));
    
    // Adăugăm noul event listener
    document.getElementById('startUploadBtn').addEventListener('click', async () => {
        const fileInput = document.getElementById('documentFile');
        if (!fileInput.files || fileInput.files.length === 0) {
            showToast('Selectați un fișier JSON pentru încărcare.', 'error');
            return;
        }
        
        if (!currentCollection) {
            showToast('Selectați o colecție pentru încărcare.', 'error');
            return;
        }
        
        const file = fileInput.files[0];
        
        // Validăm fișierul JSON înainte de upload
        try {
            showToast('Validare format JSON...', 'info');
            const validation = await validateJsonFile(file);
            showToast(`Fișier JSON valid: ${validation.chunksCount} chunk-uri detectate.`, 'success');
        } catch (validationError) {
            showToast(`Format JSON invalid: ${validationError.message}`, 'error');
            return;
        }
        
        const success = await uploadDocument(currentCollection, file);
        if (success) {
            bootstrap.Modal.getInstance(document.getElementById('uploadDocumentModal')).hide();
            fileInput.value = '';
            await loadDocuments(currentCollection);
        }
    });
    
    // Event listener pentru interogare
    document.getElementById('runQueryBtn').addEventListener('click', handleQuery);
    
    // Event listener pentru căutarea în colecții
    document.getElementById('searchCollections').addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        document.querySelectorAll('#collectionsList .list-group-item').forEach(item => {
            const collectionText = item.textContent.toLowerCase();
            if (collectionText.includes(searchTerm)) {
                item.style.display = '';
            } else {
                item.style.display = 'none';
            }
        });
    });

    // Event listener pentru Enter în câmpul de interogare
    document.getElementById('queryInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleQuery();
        }
    });
    
    // Event listener pentru validarea input-ului de fișiere
    document.getElementById('documentFile').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && !file.name.toLowerCase().endsWith('.json')) {
            showToast('Selectați doar fișiere JSON cu extensia .json', 'error');
            e.target.value = ''; // Resetăm selecția
        }
    });
    
    // Tooltip pentru butoane (opțional)
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});