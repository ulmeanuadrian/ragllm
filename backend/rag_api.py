# rag_api.py - VERSIUNEA DOAR PENTRU JSON CHUNKIZAT

import os
import re
import time
import json
import traceback
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from qdrant_client import QdrantClient
from typing import List, Dict, Any, Optional
from models import CollectionCreate, DocumentDeleteRequest, GenerateRequest
from pydantic import ConfigDict, BaseModel, Field
from typing import Dict, List, Optional, Any

# Implementăm o versiune simplificată a clasei Document pentru a evita problemele cu pydantic
class Document(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    content: str
    content_type: str = "text"
    id: Optional[str] = None
    score: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    id_hash_keys: Optional[List[str]] = None

# Importuri pentru farm-haystack 1.26.4
from haystack.preview.components.preprocessors import PreProcessor
from haystack.document_stores.qdrant import QdrantDocumentStore
from haystack.preview.components.retrievers import EmbeddingRetriever
from haystack.preview.pipelines import Pipeline
from haystack.preview.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.preview.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter

# Configurare logging structurat în format JSON
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        for key, value in record.__dict__.items():
            if key not in ["args", "exc_info", "exc_text", "msg", "message", "levelname", 
                          "levelno", "pathname", "filename", "module", "lineno", 
                          "funcName", "created", "asctime", "msecs", "relativeCreated",
                          "thread", "threadName", "processName", "process", "name"]:
                log_record[key] = value
                
        return json.dumps(log_record)

# Configurare logger
logger = logging.getLogger("rag_api")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Helper pentru măsurarea duratei operațiilor și logging structurat
class TimingLogger:
    def __init__(self, operation_name, collection=None, **extra_fields):
        self.operation_name = operation_name
        self.collection = collection
        self.start_time = None
        self.extra_fields = extra_fields
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Începere {self.operation_name}", 
                   extra={"collection": self.collection, **self.extra_fields})
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        extra = {
            "duration": f"{duration:.2f}s",
            "duration_ms": int(duration * 1000),
            "timestamp_end": datetime.now().isoformat()
        }
        
        if self.collection:
            extra["collection"] = self.collection
            
        extra.update(self.extra_fields)
            
        if exc_type is not None:
            error_details = traceback.format_exception(exc_type, exc_val, exc_tb)
            logger.error(f"{self.operation_name} eșuat în {duration:.2f}s: {exc_val}", 
                        extra={**extra, "error": str(exc_val), "traceback": error_details})
        else:
            logger.info(f"{self.operation_name} finalizat în {duration:.2f}s", extra=extra)

# Importăm generatorul Gemini
from gemini_generator import GeminiGenerator

# Cache pentru interogări pentru a evita procesarea repetată a acelorași întrebări
from collections import OrderedDict
QUERY_CACHE = OrderedDict()
MAX_QUERY_CACHE_SIZE = 100

def get_from_query_cache(key: str):
    """Obține rezultatele din cache pentru o interogare."""
    global QUERY_CACHE
    if key in QUERY_CACHE:
        value = QUERY_CACHE.pop(key)
        QUERY_CACHE[key] = value
        logger.info("Rezultate obținute din cache", extra={"cache_key": key[:30]})
        return value
    return None

def save_to_query_cache(key: str, value: list):
    """Salvează rezultatele unei interogări în cache."""
    global QUERY_CACHE, MAX_QUERY_CACHE_SIZE
    
    if len(QUERY_CACHE) >= MAX_QUERY_CACHE_SIZE:
        QUERY_CACHE.popitem(last=False)
    
    QUERY_CACHE[key] = value
    logger.info("Rezultate salvate în cache", extra={"cache_key": key[:30], "results_count": len(value)})

# --- CONFIGURARE GENERALĂ ---
UPLOAD_DIR = Path("./uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

qdrant_client = QdrantClient(url="http://localhost:6333")

app = FastAPI(title="RAG API - Doar JSON Chunkizat")

# Eveniment de startup pentru inițializarea aplicației
@app.on_event("startup")
def startup_event():
    logger.info("Inițializare aplicație RAG pentru JSON chunkizat")
    try:
        collections_response = qdrant_client.get_collections()
        logger.info(f"Conexiune la Qdrant stabilită cu succes. Colecții disponibile: {len(collections_response.collections)}")
    except Exception as e:
        logger.error(f"Eroare la conectarea la Qdrant: {str(e)}")

# Configurare CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "X-Content-Type-Options"],
    max_age=600
)

# Middleware pentru gestionarea erorilor
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"Eroare server: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "A apărut o eroare la procesarea cererii.",
                "error_type": type(e).__name__
            }
        )

# --- COMPONENTE HAYSTACK REUTILIZABILE ---
# Dimensiunea vectorului de embedding pentru modelul all-mpnet-base-v2
EMBEDDING_DIM = 768

def get_document_store(collection_name: str, recreate: bool = False) -> QdrantDocumentStore:
    """
    Obține un document store pentru o colecție specificată.
    """
    try:
        collections_response = qdrant_client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
        
        if collection_name in existing_collections and not recreate:
            collection_info = qdrant_client.get_collection(collection_name)
            existing_dim = collection_info.config.params.vectors.size
            
            if existing_dim != EMBEDDING_DIM:
                print(f"Dimensiunea vectorului nu se potrivește: {existing_dim} vs {EMBEDDING_DIM}")
                return get_document_store(collection_name, recreate=True)
        
        document_store = QdrantDocumentStore(
            url="http://localhost:6333",
            index=collection_name,
            embedding_dim=EMBEDDING_DIM,
            recreate_index=recreate,
            similarity="cosine"
        )
        return document_store
    except ValueError as e:
        if "vector size" in str(e) and not recreate:
            logger.warning(f"Recreare folder cu dimensiunea corectă a vectorului: {collection_name}", 
                         extra={"collection": collection_name})
            return get_document_store(collection_name, recreate=True)
        else:
            logger.error(f"Eroare la inițializarea document store: {str(e)}", 
                       extra={"collection": collection_name, "error": str(e)})
            raise

def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează un fișier JSON care conține chunk-uri în formatul exact din streamlite.json
    
    Format așteptat:
    {
        "chunk_0": {
            "metadata": "Source: filename.pdf",
            "chunk": "conținutul chunk-ului..."
        },
        "chunk_1": {
            "metadata": "Source: filename.pdf", 
            "chunk": "conținutul chunk-ului..."
        }
    }
    """
    start_time = time.time()
    chunks_data = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        logger.info(f"JSON încărcat cu succes, verificare format...", extra={"file": file_path})
        
        # Verificăm dacă JSON-ul conține chunk-uri în formatul așteptat
        chunk_count = 0
        for key, value in json_data.items():
            if key.startswith("chunk_") and isinstance(value, dict):
                if "metadata" in value and "chunk" in value:
                    chunk_count += 1
                    
                    # Parsăm metadatele pentru a extrage informații
                    metadata = {
                        "chunk_id": key,
                        "original_source": value["metadata"],
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "chunk_index": chunk_count - 1
                    }
                    
                    # Adăugăm chunk-ul în lista de procesare
                    chunks_data.append({
                        "content": value["chunk"],
                        "metadata": metadata,
                        "chunk_id": key
                    })
        
        if chunk_count == 0:
            raise ValueError("Fișierul JSON nu conține chunk-uri în formatul așteptat. "
                           "Formatul așteptat: {'chunk_0': {'metadata': '...', 'chunk': '...'}, ...}")
        
        logger.info(f"Fișier JSON procesat cu succes: {len(chunks_data)} chunk-uri extrase în {time.time() - start_time:.2f} secunde",
                  extra={"file": file_path, "chunks": chunk_count, "duration": f"{time.time() - start_time:.2f}s"})
        
        return chunks_data
    except json.JSONDecodeError as e:
        logger.error(f"Eroare la parsarea JSON: {str(e)}", extra={"file": file_path})
        raise ValueError(f"Fișierul JSON nu este valid: {str(e)}")
    except Exception as e:
        logger.error(f"Eroare la procesarea fișierului JSON: {str(e)}")
        raise e

# --- ENDPOINTS API PENTRU CRUD ---

@app.post("/collections/{collection_name}/upload", tags=["CRUD"])
async def upload_json_file(collection_name: str, file: UploadFile = File(...)):
    """
    Endpoint pentru încărcarea și procesarea fișierelor JSON chunkizate.
    Acceptă DOAR fișiere JSON în formatul din streamlite.json
    """
    try:
        start_time = time.time()
        
        # Verificăm dacă fișierul este JSON
        if not file.filename.lower().endswith('.json'):
            raise HTTPException(
                status_code=400, 
                detail="Sunt acceptate DOAR fișiere JSON în format chunkizat. "
                       "Formatul așteptat: {'chunk_0': {'metadata': '...', 'chunk': '...'}, ...}"
            )
        
        # Verificăm tipul MIME
        if file.content_type != 'application/json':
            logger.warning(f"Tip MIME neașteptat: {file.content_type}, dar continuăm cu procesarea")
        
        print(f"Procesare fișier JSON chunkizat: {file.filename} pentru folder: {collection_name}")
        
        # Creăm un nume de fișier unic pentru a evita suprascrierea
        timestamp = int(time.time() * 1000)
        file_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"
        
        # Scriem fișierul pe disc
        with open(file_path, "wb") as f:
            content_bytes = await file.read()
            print(f"Dimensiune fișier: {len(content_bytes)} bytes")
            f.write(content_bytes)

        logger.info("Procesare JSON chunkizat...", extra={"file": file.filename})
        
        try:
            with TimingLogger("Procesare JSON chunkizat", collection=collection_name, file=file.filename):
                # Folosim funcția optimizată pentru procesarea fișierelor JSON chunkizate
                chunks_data = process_json_chunks(file_path)
            
            if not chunks_data:
                raise HTTPException(
                    status_code=400, 
                    detail="Fișierul JSON nu conține chunk-uri în formatul așteptat. "
                           "Formatul așteptat: {'chunk_0': {'metadata': '...', 'chunk': '...'}, ...}"
                )
            
            logger.info(f"Detectate {len(chunks_data)} chunk-uri predefinite în JSON", 
                       extra={"chunks_count": len(chunks_data), "file": file.filename})
            
            # Creăm document store și configurăm pipeline-ul de indexare
            document_store = get_document_store(collection_name)
            
            # Creăm toate documentele într-o singură listă pentru procesare în batch
            docs_to_index = []
            
            for chunk_data in chunks_data:
                # Adăugăm metadate suplimentare
                metadata = chunk_data["metadata"]
                metadata["source"] = file.filename
                metadata["file_type"] = "json_chunked"
                
                # Creăm documentul
                doc = Document(
                    content=chunk_data["content"],
                    meta=metadata
                )
                docs_to_index.append(doc)
            
            # Folosim retriever-ul pentru a adăuga documentele în document store în batch
            retriever = EmbeddingRetriever(
                document_store=document_store,
                model_name_or_path="sentence-transformers/all-mpnet-base-v2",
                device="cpu",
                batch_size=32
            )
            
            # Actualizăm embedding-urile documentelor într-un singur batch
            logger.info(f"Generare embeddings pentru {len(docs_to_index)} documente", 
                       extra={"docs_count": len(docs_to_index)})
            document_store.update_embeddings(retriever=retriever, documents=docs_to_index)
            
            # Scriem toate documentele în document store într-un singur batch
            document_store.write_documents(docs_to_index)
            
            logger.info(f"Au fost indexate {len(docs_to_index)} chunk-uri din JSON", 
                       extra={"indexed_count": len(docs_to_index), "file": file.filename})
            
            # Curățăm fișierul temporar
            os.remove(file_path)
            
            processing_time = time.time() - start_time
            print(f"Procesare finalizată în {processing_time:.2f} secunde")
            
            return {
                "status": "success", 
                "message": f"Fișierul JSON '{file.filename}' a fost procesat cu succes. {len(docs_to_index)} chunk-uri au fost indexate.",
                "filename": file.filename,
                "collection": collection_name,
                "chunks_count": len(docs_to_index),
                "processing_time": f"{processing_time:.2f}s"
            }
            
        except json.JSONDecodeError as je:
            logger.error(f"Eroare la parsarea JSON: {str(je)}", extra={"file": file.filename, "error": str(je)})
            raise HTTPException(status_code=400, detail=f"Fișierul JSON nu este valid: {str(je)}")
        except ValueError as ve:
            logger.error(f"Format JSON invalid: {str(ve)}", extra={"file": file.filename})
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Eroare la procesarea JSON: {str(e)}", extra={"file": file.filename, "error": str(e)})
            raise HTTPException(status_code=400, detail=f"Eroare la procesarea JSON: {str(e)}")
        finally:
            # Curățăm fișierul temporar în caz de eroare
            if file_path.exists():
                os.remove(file_path)
            
    except HTTPException:
        # Re-ridicăm HTTPException-urile pentru a păstra status code-ul
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"EROARE LA PROCESARE: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"A apărut o eroare: {str(e)}")

# Funcție reutilizabilă pentru interogarea colecției
async def query_documents(collection_name: str, query_text: str, top_k: int = 5):
    """
    Interoghează o colecție și returnează documentele relevante.
    """
    try:
        if not query_text.strip():
            logger.warning("Interogare goală primită", extra={"collection": collection_name})
            return []
            
        # Generăm o cheie pentru cache
        cache_key = f"{collection_name}:{query_text}:{top_k}"
        
        # Verificăm cache-ul pentru a evita procesarea repetată
        cached_result = get_from_query_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Rezultat găsit în cache pentru interogarea: '{query_text[:30]}...'")
            return cached_result
            
        start_time = time.time()
        with TimingLogger("Procesare interogare", collection=collection_name):
            # Inițializăm document store
            document_store = get_document_store(collection_name)
            
            # 1. Căutare exactă - verificăm dacă avem o potrivire exactă a frazei
            all_docs = document_store.get_all_documents()
            exact_matches = []
            
            # Pregătim variante ale interogării pentru căutare mai flexibilă
            query_variants = [
                query_text,
                query_text.lower(),
                ' '.join(query_text.lower().split()),
                re.sub(r'[^\w\s]', '', query_text.lower())
            ]
            
            # Căutăm potriviri exacte folosind toate variantele
            for doc in all_docs:
                content = doc.content
                content_lower = content.lower()
                content_norm = ' '.join(content_lower.split())
                content_clean = re.sub(r'[^\w\s]', '', content_lower)
                
                for variant in query_variants:
                    if variant in content or variant in content_lower or \
                       variant in content_norm or variant in content_clean:
                        exact_matches.append({
                            "content": content,
                            "score": 1.0,
                            "meta": doc.meta,
                            "id": doc.id,
                            "embedding": None,
                            "match_type": "exact"
                        })
                        break
            
            # Dacă avem suficiente potriviri exacte, le returnăm direct
            if len(exact_matches) >= top_k:
                logger.info(f"Găsite {len(exact_matches)} potriviri exacte în {time.time() - start_time:.2f} secunde")
                return exact_matches[:top_k]
            
            # 2. Căutare semantică (embeddings)
            logger.info("Utilizare retriever pentru căutare semantică")
            
            retriever = EmbeddingRetriever(
                document_store=document_store,
                model_name_or_path="sentence-transformers/all-mpnet-base-v2",
                top_k=top_k * 2,
                device="cpu"
            )
            
            semantic_results = retriever.run(query=query_text)
            
            # Convertim rezultatele semantice în dicționare
            semantic_results_dicts = []
            for res in semantic_results:
                if isinstance(res, dict):
                    res_dict = res.copy()
                    res_dict["match_type"] = "semantic"
                    semantic_results_dicts.append(res_dict)
                else:
                    semantic_results_dicts.append({
                        "content": res.content if hasattr(res, "content") else str(res),
                        "score": getattr(res, "score", 0.8),
                        "meta": getattr(res, "meta", {}),
                        "id": getattr(res, "id", f"sem_{id(res)}"),
                        "embedding": None,
                        "match_type": "semantic"
                    })
            
            semantic_results = semantic_results_dicts
            
            # 3. Căutare bazată pe cuvinte cheie
            keyword_results = []
            
            # Lista de cuvinte comune în limba română care pot fi ignorate
            stopwords = {'si', 'in', 'la', 'cu', 'de', 'pe', 'pentru', 'din', 'care', 'este', 'sunt', 
                        'ce', 'cum', 'unde', 'cand', 'cat', 'a', 'al', 'ai', 'ale', 'o', 'un', 'una', 
                        'niste', 'acest', 'aceasta', 'aceste', 'acesti', 'sau', 'ori', 'dar', 'insa'}
            
            query_terms_raw = query_text.lower().split()
            query_terms = [term for term in query_terms_raw if term not in stopwords and len(term) > 2]
            
            if not query_terms:
                query_terms = query_terms_raw
            
            # Calculăm scoruri pentru potrivirea cuvintelor cheie
            for doc in all_docs:
                content_lower = doc.content.lower()
                
                matches = sum(1 for term in query_terms if term in content_lower)
                
                if matches > 0 and len(query_terms) > 0:
                    base_score = matches / len(query_terms)
                    
                    consecutive_bonus = 0
                    for i in range(len(query_terms) - 1):
                        if f"{query_terms[i]} {query_terms[i+1]}" in content_lower:
                            consecutive_bonus += 0.1
                    
                    final_score = min(0.95, base_score + consecutive_bonus)
                    
                    keyword_results.append({
                        "content": doc.content,
                        "score": final_score,
                        "meta": doc.meta,
                        "id": doc.id,
                        "embedding": None,
                        "match_type": "keyword"
                    })
            
            keyword_results = sorted(keyword_results, key=lambda x: x["score"], reverse=True)[:top_k]
            
            # Combinăm toate rezultatele
            combined_results = exact_matches + semantic_results + keyword_results
            
            # Eliminăm duplicatele
            seen_ids = set()
            unique_results = []
            
            for result in combined_results:
                if isinstance(result, dict) and "id" in result:
                    if result["id"] not in seen_ids:
                        seen_ids.add(result["id"])
                        unique_results.append(result)
                elif hasattr(result, "id"):
                    if result.id not in seen_ids:
                        seen_ids.add(result.id)
                        unique_results.append({
                            "content": result.content,
                            "score": getattr(result, "score", 0.0),
                            "meta": result.meta,
                            "id": result.id,
                            "embedding": None,
                            "match_type": "semantic"
                        })
            
            # Sortăm rezultatele finale după scor
            final_results = sorted(unique_results, key=lambda x: x.get("score", 0.0) if isinstance(x, dict) else 0.0, reverse=True)[:top_k]
            
            # Salvăm rezultatul în cache
            save_to_query_cache(cache_key, final_results)
            
            logger.info(f"Interogare finalizată, găsite {len(final_results)} rezultate")
            return final_results
    except Exception as e:
        logger.error(f"Eroare la interogarea colecției: {str(e)}")
        raise e

@app.post("/collections/{collection_name}/query", tags=["CRUD"])
async def query_collection(collection_name: str, request: Request):
    """
    Endpoint pentru interogarea documentelor dintr-o folder.
    """
    try:
        data = await request.json()
        query_text = data.get("question", "")
        top_k = data.get("top_k", 5)
        
        logger.info(f"Delegare interogare către funcția reutilizabilă query_documents", 
                   extra={"collection": collection_name, "query": query_text[:30] if query_text else "", "top_k": top_k})
        
        results = await query_documents(collection_name, query_text, top_k)
        return results
        
    except Exception as e:
        logger.error(f"Eroare la procesarea cererii: {str(e)}")
        raise HTTPException(status_code=500, detail=f"A apărut o eroare la procesarea interogării: {str(e)}")

# READ: Endpoint pentru a lista toate folderele
@app.get("/collections", tags=["Management"])
async def list_collections():
    collections_response = qdrant_client.get_collections()
    return [c.name for c in collections_response.collections]

# READ: Endpoint pentru a lista documentele dintr-un folder
@app.get("/collections/{collection_name}/documents", tags=["Management"])
async def list_documents(collection_name: str):
    try:
        document_store = get_document_store(collection_name)
        documents = document_store.filter_documents()
        
        # Grupăm documentele după fișierul sursă
        files_dict = {}
        
        for doc in documents:
            meta = doc.meta if hasattr(doc, 'meta') else {}
            source = meta.get('source', 'Necunoscut')
            created_at = meta.get('created_at', 'Necunoscut')
            
            if source not in files_dict:
                files_dict[source] = {
                    "source": source,
                    "created_at": created_at,
                    "doc_count": 0,
                    "doc_ids": [],
                    "file_type": meta.get('file_type', 'json_chunked')
                }
            
            files_dict[source]["doc_count"] += 1
            files_dict[source]["doc_ids"].append(doc.id)
        
        result = list(files_dict.values())
        result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la listarea documentelor: {str(e)}")

# CREATE: Endpoint pentru a crea un folder nouă
@app.post("/collections/{collection_name}", tags=["Management"])
async def create_collection(collection_name: str):
    """
    Endpoint pentru crearea unei foldere noi în Qdrant.
    """
    try:
        with TimingLogger("Creare colecție", collection=collection_name):
            collections_response = qdrant_client.get_collections()
            existing_collections = [c.name for c in collections_response.collections]
            
            if collection_name not in existing_collections:
                logger.info(f"Colecția {collection_name} nu există. Se creează automat.", extra={"collection": collection_name})
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "content": {
                            "size": 768,
                            "distance": "Cosine"
                        }
                    }
                )
                return {"status": "created", "collection": collection_name}
            else:
                return {"status": "exists", "message": f"Colecția {collection_name} există deja"}
    except Exception as e:
        logger.error(f"Eroare la crearea folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Eroare la crearea folder: {str(e)}")

# DELETE: Endpoint pentru a șterge documentele asociate cu un fișier
@app.delete("/collections/{collection_name}/documents", tags=["Management"])
async def delete_documents_by_source(collection_name: str, request: Request):
    data = await request.json()
    source_file = data.get("source")
    if not source_file:
        raise HTTPException(status_code=400, detail="Trebuie specificat un 'source'.")
    
    try:
        document_store = get_document_store(collection_name)
        
        # Găsim toate documentele cu sursa specificată
        documents = document_store.filter_documents()
        
        # Filtrăm documentele care au sursa specificată
        docs_to_delete = [doc.id for doc in documents if doc.meta.get("source") == source_file]
        
        if not docs_to_delete:
            return {"status": "no_documents_found", "source": source_file, "collection": collection_name}
        
        # Ștergem documentele folosind ID-urile lor
        document_store.delete_documents(document_ids=docs_to_delete)
        
        return {"status": "deleted", "source": source_file, "collection": collection_name, "count": len(docs_to_delete)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la ștergerea documentelor: {str(e)}")

# DELETE: Endpoint pentru a șterge o întreagă folder
@app.delete("/collections/{collection_name}", tags=["Management"])
async def delete_collection(collection_name: str):
    try:
        result = qdrant_client.delete_collection(collection_name=collection_name)
        return {"status": "collection_deleted", "collection": collection_name, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la ștergerea folder: {str(e)}")

# Endpoint pentru generarea răspunsurilor folosind Gemini
@app.post("/collections/{collection_name}/generate", tags=["Generation"])
async def generate_response(collection_name: str, request: GenerateRequest):
    """
    Endpoint pentru generarea răspunsurilor folosind Gemini pe baza documentelor recuperate.
    """
    try:
        with TimingLogger("Generare răspuns Gemini", collection=collection_name, query=request.query[:50]):
            # Extragem parametrii din cerere
            query = request.query
            temperature = request.temperature if request.temperature is not None else 0.2
            top_k_docs = request.top_k_docs if request.top_k_docs is not None else 5
            
            logger.info(f"Parametri primiti pentru generare", 
                       extra={"query": query[:50], "temperature": temperature, "top_k_docs": top_k_docs})
            
            # Verificăm dacă interogarea este goală
            if not query.strip():
                logger.warning("Interogare goală primită", extra={"collection": collection_name})
                return {"answer": "Interogarea nu poate fi goală."}
            
            # Pas 1: Interogăm colecția pentru a obține documentele relevante
            logger.info(f"Recuperare documente pentru generare folosind funcția reutilizabilă query_documents", 
                       extra={"collection": collection_name, "query": query[:50], "top_k": top_k_docs})
            
            docs_response = await query_documents(collection_name, query, top_k_docs)
            
            # Verificăm dacă am găsit documente
            if not docs_response or len(docs_response) == 0:
                logger.warning("Nu s-au găsit documente pentru generarea răspunsului", 
                             extra={"collection": collection_name, "query": query[:50]})
                return {"answer": "Nu am găsit informații relevante pentru a răspunde la această întrebare.", 
                        "documents": []}
            
            # Pas 2: Folosim Gemini pentru a genera un răspuns bazat pe documentele recuperate
            logger.info(f"Generare răspuns cu Gemini pentru {len(docs_response)} documente", 
                       extra={"collection": collection_name, "doc_count": len(docs_response)})
            
            gemini = GeminiGenerator()
            answer = gemini.generate_response(
                query=query,
                context_docs=docs_response,
                temperature=temperature
            )
            
            return {"answer": answer, "documents": docs_response}
    except Exception as e:
        logger.error(f"Eroare la generarea răspunsului", 
                    extra={"error": str(e), "collection": collection_name, "traceback": traceback.format_exc()})
        return {"error": f"A apărut o eroare la generarea răspunsului: {str(e)}", "answer": ""}

# --- Pornire Uvicorn (pentru rulare locală) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8070, reload=True)