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
        loadingText.textContent = 'Se încarcă...';
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
    toast.innerHTML = `
        <div class="toast-content">
            <span>${message}</span>
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
            document.body.removeChild(toast);
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

// Configurare pentru toate cererile fetch
const fetchConfig = {
    headers: {
        'Content-Type': 'application/json'
    }
};

// Funcții pentru gestionarea folderelor
// Funcție pentru popularea listei de foldere în interfață
function populateCollectionsList(collections) {
    const collectionsList = document.getElementById('collectionsList');
    collectionsList.innerHTML = '';
    
    if (collections.length === 0) {
        collectionsList.innerHTML = '<li class="list-group-item">Nu există foldere. Creați una nouă.</li>';
        return;
    }
    
    collections.forEach(collection => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center';
        li.innerHTML = `
            <span>${collection}</span>
            <div class="actions">
                <button class="btn btn-sm btn-danger delete-collection" data-collection="${collection}">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        `;
        li.addEventListener('click', () => selectCollection(collection));
        collectionsList.appendChild(li);
    });
    
    // Adaugă event listeners pentru butoanele de ștergere
    document.querySelectorAll('.delete-collection').forEach(button => {
        button.addEventListener('click', (e) => {
            e.stopPropagation(); // Previne selectarea folder
            const collection = button.getAttribute('data-collection');
            showDeleteConfirmation('folder', collection);
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
        console.log('Serverul nu este disponibil momentan. Se afișează interfața fără foldere.');
        populateCollectionsList([]);
    } finally {
        hideLoading('.sidebar');
    }
}

async function createCollection(name) {
    try {
        // Verificăm dacă numele folder este valid
        if (!name || !/^[a-zA-Z0-9_]+$/.test(name)) {
            showToast('Numele folder trebuie să conțină doar litere, cifre și underscore (_).', 'error');
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
            showToast(`Colecția "${name}" a fost creată cu succes. Încărcați documente pentru a o popula.`);
            await loadCollections();
            selectCollection(name);
        } else {
            showToast(`Eroare la crearea folder "${name}".`, 'error');
            return false;
        }
        return true;
    } catch (error) {
        console.error('Eroare la crearea folder:', error);
        showToast('Eroare la crearea folder. Verificați consola pentru detalii.', 'error');
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
                document.getElementById('currentCollection').textContent = 'Selectați un folder';
                document.getElementById('uploadDocumentBtn').disabled = true;
                document.getElementById('runQueryBtn').disabled = true;
                document.getElementById('documentsList').innerHTML = '';
            }
            await loadCollections();
            return true;
        } else {
            throw new Error('Eroare la ștergerea folder');
        }
    } catch (error) {
        console.error('Eroare la ștergerea folder:', error);
        showToast(`Eroare la ștergerea folder: ${error.message}`, 'error');
        return false;
    } finally {
        hideLoading('.main-content');
    }
}

function selectCollection(name) {
    currentCollection = name;
    document.getElementById('currentCollection').textContent = `Colecția: ${name}`;
    document.getElementById('uploadDocumentBtn').disabled = false;
    document.getElementById('runQueryBtn').disabled = false;
    
    // Marchează colecția selectată în listă
    document.querySelectorAll('#collectionsList .list-group-item').forEach(item => {
        if (item.querySelector('span')?.textContent === name) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Încarcă documentele din folder
    loadDocuments(name);
}

// Funcții pentru gestionarea documentelor
async function loadDocuments(collectionName) {
    try {
        showLoading('.main-content');
        
        // Folosim endpoint-ul actualizat care grupează documentele după fișierul sursă
        const documents = await fetchWithErrorHandling(`${API_BASE_URL}/collections/${collectionName}/documents`);
        
        const documentsList = document.getElementById('documentsList');
        
        if (documents.length === 0) {
            documentsList.innerHTML = '<tr><td colspan="4" class="text-center">Nu există documente în această folder. Încărcați documente noi.</td></tr>';
            return;
        }
        
        // Salvăm documentele în variabila globală
        window.documentsList = documents;
        
        // Construim tabelul cu fișierele originale
        let html = '';
        documents.forEach(file => {
            html += `
            <tr>
                <td>${file.source || 'Necunoscut'}</td>
                <td>Fișier încărcat (${file.doc_count} fragmente)</td>
                <td>${file.created_at || '-'}</td>
                <td>
                    <button class="btn btn-sm btn-danger delete-document" data-source="${file.source}">
                        <i class="bi bi-trash"></i>
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
            '<tr><td colspan="4" class="text-center">Eroare la încărcarea documentelor. Verificați consola pentru detalii.</td></tr>';
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
        
        // Verificăm tipul fișierului
        const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'application/json'];
        if (!allowedTypes.includes(file.type)) {
            showToast('Tip de fișier nesuportat. Sunt acceptate doar PDF, DOCX, TXT și JSON.', 'error');
            return false;
        }
        
        showLoading('.main-content');
        
        // Folosim fetch direct pentru upload de fișiere
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/collections/${collectionName}/upload`, {
            method: 'POST',
            body: formData
        });
        
        // Monitorizăm progresul încărcării (opțional, pentru versiuni viitoare)
        // const reader = response.body.getReader();
        // const contentLength = +response.headers.get('Content-Length');
        // let receivedLength = 0;
        
        const success = response.ok;
        
        if (success) {
            showToast(`Fișierul "${file.name}" a fost încărcat și procesat cu succes.`);
            return true;
        } else {
            throw new Error('Eroare la încărcarea documentului');
        }
    } catch (error) {
        console.error('Eroare la încărcarea documentului:', error);
        showToast(`Eroare la încărcarea documentului: ${error.message}`, 'error');
        return false;
    } finally {
        hideLoading('.main-content');
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
            showToast(`Documentul "${source}" a fost șters cu succes.`);
            await loadDocuments(collectionName);
            return true;
        } else {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'Eroare la ștergerea documentului');
        }
    } catch (error) {
        console.error('Eroare la ștergerea documentului:', error);
        showToast(`Eroare la ștergerea documentului: ${error.message}`, 'error');
        return false;
    } finally {
        hideLoading('.main-content');
    }
}

// Funcții pentru interogare
async function queryCollection(collectionName, queryText, topK = 3) {
    try {
        showLoading('#queryResults');
        
        // Folosim fetch direct deoarece CORS este configurat corect în backend
        const response = await fetch(`${API_BASE_URL}/collections/${collectionName}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: queryText,
                top_k: topK
            })
        });
        
        let results = [];
        
        if (response.ok) {
            results = await response.json();
        } else {
            console.warn('Eroare HTTP la interogarea folder:', response.status);
        }
        
        return results;
    } catch (error) {
        console.error('Eroare la interogarea folder:', error);
        showToast(`Eroare la interogarea folder: ${error.message}`, 'error');
        return [];
    } finally {
        hideLoading('#queryResults');
    }
}

async function handleQuery() {
    const queryText = document.getElementById('queryInput').value.trim();
    if (!queryText) {
        showToast('Introduceți o întrebare pentru interogare.', 'error');
        return;
    }
    
    if (!currentCollection) {
        showToast('Selectați un folder pentru interogare.', 'error');
        return;
    }
    
    const topK = parseInt(document.getElementById('topK').value) || 3;
    showLoading('#query');
    
    try {
        // Afișăm un mesaj că se procesează interogarea
        const resultsContainer = document.getElementById('queryResults');
        resultsContainer.innerHTML = '<div class="alert alert-info">Se procesează interogarea...</div>';
        
        // Interogăm colecția pentru documente
        const results = await queryCollection(currentCollection, queryText, topK);
        
        // Verificăm dacă avem rezultate
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = '<div class="alert alert-warning">Nu s-au găsit documente relevante pentru această interogare.</div>';
            return;
        }
        
        // Afișăm un mesaj că se generează răspunsul
        resultsContainer.innerHTML = '<div class="alert alert-info">Se generează răspunsul cu Gemini...</div>';
        
        // Generăm răspunsul cu Gemini
        try {
            const response = await fetch(`${API_BASE_URL}/collections/${currentCollection}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: queryText,
                    temperature: 0.2,
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
            // Afișăm doar documentele dacă generarea eșuează
            displayQueryResults(results);
        }
    } catch (error) {
        console.error('Eroare la procesarea interogării:', error);
        showToast('Eroare la procesarea interogării. Verificați consola pentru detalii.', 'error');
        
        // Resetăm containerul de rezultate
        const resultsContainer = document.getElementById('queryResults');
        resultsContainer.innerHTML = '<div class="alert alert-danger">A apărut o eroare la procesarea interogării.</div>';
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
        noResults.textContent = 'Nu s-a putut genera un răspuns pentru această interogare.';
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
    questionTitle.innerHTML = `<strong>Întrebare:</strong> ${queryText}`;
    answerContainer.appendChild(questionTitle);
    
    // Adăugăm răspunsul generat
    const answerContent = document.createElement('div');
    answerContent.className = 'answer-content mt-2 p-3 border rounded';
    answerContent.innerHTML = `<strong>Răspuns:</strong><br>${data.answer.replace(/\n/g, '<br>')}`;
    answerContainer.appendChild(answerContent);
    
    // Adăugăm containerul de răspuns la rezultate
    resultsContainer.appendChild(answerContainer);
    
    // Adăugăm un separator
    const separator = document.createElement('hr');
    resultsContainer.appendChild(separator);
    
    // Adăugăm titlul pentru documente sursă
    const sourcesTitle = document.createElement('h5');
    sourcesTitle.className = 'mt-3 mb-3';
    sourcesTitle.textContent = 'Documente sursă:';
    resultsContainer.appendChild(sourcesTitle);
    
    // Afișăm documentele sursă
    if (data.documents && data.documents.length > 0) {
        displayQueryResults(data.documents, true);
    } else {
        const noSources = document.createElement('div');
        noSources.className = 'alert alert-info';
        noSources.textContent = 'Nu sunt disponibile documente sursă.';
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
        noResults.textContent = 'Nu s-au găsit rezultate pentru această interogare.';
        resultsContainer.appendChild(noResults);
        return;
    }
    
    // Convertim rezultatele în array dacă nu sunt deja
    const resultsArray = Array.isArray(results) ? results : [results];
    
    resultsArray.forEach((result, index) => {
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        
        const score = result.score ? (result.score * 100).toFixed(2) : 'N/A';
        const source = result.meta && result.meta.source ? result.meta.source : 'Necunoscută';
        
        resultCard.innerHTML = `
            <div class="result-header">
                <span class="result-number">#${index + 1}</span>
                <span class="result-score">Scor: ${score}%</span>
            </div>
            <div class="result-content">
                <p>${result.content}</p>
            </div>
            <div class="result-meta">
                <span class="result-source">Sursă: ${source}</span>
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
    
    if (type === 'folder') {
        confirmText.textContent = `Ești sigur că vrei să ștergi colecția "${name}"? Această acțiune nu poate fi anulată.`;
        confirmBtn.onclick = async () => {
            modal.hide();
            await deleteCollection(name);
        };
    } else if (type === 'document') {
        confirmText.textContent = `Ești sigur că vrei să ștergi documentul "${name}" din colecția "${currentCollection}"? Această acțiune nu poate fi anulată.`;
        confirmBtn.onclick = async () => {
            modal.hide();
            await deleteDocument(currentCollection, name);
        };
    }
    
    modal.show();
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Încarcă folderele la pornirea aplicației
    loadCollections();
    
    // Event listener pentru crearea unei foldere noi
    document.getElementById('saveCollectionBtn').addEventListener('click', async () => {
        const collectionName = document.getElementById('collectionName').value.trim();
        const success = await createCollection(collectionName);
        if (success) {
            bootstrap.Modal.getInstance(document.getElementById('createCollectionModal')).hide();
            document.getElementById('collectionName').value = '';
        }
    });
    
    // Event listener pentru încărcarea unui document - adăugăm o verificare pentru a preveni dubla înregistrare
    const startUploadBtn = document.getElementById('startUploadBtn');
    
    // Eliminăm eventualele event listener-uri existente pentru a preveni duplicarea
    startUploadBtn.replaceWith(startUploadBtn.cloneNode(true));
    
    // Adăugăm noul event listener
    document.getElementById('startUploadBtn').addEventListener('click', async () => {
        const fileInput = document.getElementById('documentFile');
        if (!fileInput.files || fileInput.files.length === 0) {
            showToast('Selectați un fișier pentru încărcare.', 'error');
            return;
        }
        
        if (!currentCollection) {
            showToast('Selectați un folder pentru încărcare.', 'error');
            return;
        }
        
        const file = fileInput.files[0];
        const success = await uploadDocument(currentCollection, file);
        if (success) {
            bootstrap.Modal.getInstance(document.getElementById('uploadDocumentModal')).hide();
            fileInput.value = '';
            await loadDocuments(currentCollection);
        }
    });
    
    // Event listener pentru interogare
    document.getElementById('runQueryBtn').addEventListener('click', handleQuery);
    
    // Event listener pentru căutarea în foldere
    document.getElementById('searchCollections').addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        document.querySelectorAll('#collectionsList .list-group-item').forEach(item => {
            const collectionName = item.querySelector('span')?.textContent.toLowerCase();
            if (collectionName && collectionName.includes(searchTerm)) {
                item.style.display = '';
            } else {
                item.style.display = 'none';
            }
        });
    });
});
