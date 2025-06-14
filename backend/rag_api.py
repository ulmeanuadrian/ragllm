import os
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import aiofiles
from dotenv import load_dotenv
import colorlog
from utils import (
    process_json_chunks, 
    validate_json_format, 
    get_json_statistics,
    preview_json_chunks
)

load_dotenv()

# Configurare logging cu culori
def setup_logging():
    """Configurează logging-ul cu culori pentru debugging mai ușor"""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, os.getenv('LOG_LEVEL', 'INFO')))
    
    # Reduce logging pentru biblioteci externe
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()

# Configurare aplicație
app = FastAPI(
    title="RAG API - JSON Chunkizat (Simplificat)",
    description="API pentru Retrieval-Augmented Generation cu fișiere JSON chunkizate - Versiune simplificată",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurare CORS pentru frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # În producție, specificați domeniile exacte
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurări globale
CONFIG = {
    "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
    "GENERATIVE_MODEL": os.getenv("GENERATIVE_MODEL", "gemini-pro"),
    "MAX_FILE_SIZE": int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024,
    "DEFAULT_TOP_K": int(os.getenv("DEFAULT_TOP_K", 5)),
    "DEFAULT_TEMPERATURE": float(os.getenv("DEFAULT_TEMPERATURE", 0.2))
}

# Instanțe globale
chroma_client = None
genai_model = None

# Modele Pydantic pentru validare
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k_docs: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    total_chunks: int
    created_at: str

class DocumentInfo(BaseModel):
    source: str
    doc_count: int
    created_at: str
    file_size: Optional[int] = None

class DeleteDocumentRequest(BaseModel):
    source: str

# Funcții helper pentru embeddings simple
def simple_text_hash(text: str) -> str:
    """Creează un hash simplu pentru text (înlocuiește embeddings-urile)"""
    return hashlib.md5(text.encode()).hexdigest()

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extrage cuvinte cheie simple din text"""
    # Curățăm textul
    import re
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    # Filtrăm cuvintele comune
    stop_words = {
        'și', 'în', 'la', 'de', 'cu', 'pe', 'din', 'pentru', 'este', 'sunt', 'a', 'al', 'ale',
        'că', 'să', 'se', 'nu', 'mai', 'dar', 'sau', 'dacă', 'când', 'cum', 'unde', 'care',
        'and', 'in', 'to', 'of', 'with', 'on', 'from', 'for', 'is', 'are', 'the', 'a', 'an',
        'that', 'this', 'it', 'be', 'have', 'has', 'do', 'does', 'will', 'would', 'could'
    }
    
    # Numărăm frecvența cuvintelor (excluzând stop words și cuvinte scurte)
    word_freq = {}
    for word in words:
        if len(word) > 2 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sortăm după frecvență
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:max_keywords]]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculează similaritatea între două texte folosind keywords overlap"""
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Calculăm Jaccard similarity
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0

# Funcții de inițializare
async def initialize_services():
    """Inițializează serviciile necesare pentru RAG"""
    global chroma_client, genai_model
    
    try:
        logger.info("🚀 Inițializare servicii RAG simplificat...")
        
        # 1. Inițializare ChromaDB
        logger.info("📊 Conectare la ChromaDB...")
        persist_dir = Path(CONFIG["CHROMA_PERSIST_DIR"])
        persist_dir.mkdir(exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        logger.info(f"✅ ChromaDB inițializat în: {persist_dir}")
        
        # 2. Configurare Google Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key or google_api_key == "your_google_api_key_here":
            logger.error("❌ GOOGLE_API_KEY nu este configurat în .env!")
            logger.error("   Obține un API key de la: https://makersuite.google.com/app/apikey")
            raise ValueError("Google API Key nu este configurat")
        
        logger.info("🔑 Configurare Google Gemini...")
        genai.configure(api_key=google_api_key)
        genai_model = genai.GenerativeModel(CONFIG["GENERATIVE_MODEL"])
        
        # Test rapid Gemini
        try:
            test_response = genai_model.generate_content("Test")
            logger.info("✅ Google Gemini configurat și funcțional")
        except Exception as e:
            logger.warning(f"⚠️ Test Gemini: {str(e)} (poate fi temporar)")
        
        logger.info("🎉 Serviciile au fost inițializate cu succes!")
        
    except Exception as e:
        logger.error(f"❌ Eroare la inițializarea serviciilor: {str(e)}")
        raise

def get_or_create_collection(collection_name: str):
    """Obține sau creează o colecție ChromaDB"""
    try:
        # Încercăm să obținem colecția existentă
        collection = chroma_client.get_collection(name=collection_name)
        logger.debug(f"📁 Colecție existentă găsită: {collection_name}")
    except Exception:
        # Dacă nu există, o creăm
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"created_at": datetime.now().isoformat()}
        )
        logger.info(f"📁 Colecție nouă creată: {collection_name}")
    
    return collection

async def process_and_store_chunks(collection_name: str, chunks_data: List[Dict[str, Any]], source_file: str):
    """Procesează și stochează chunk-urile în ChromaDB (fără embeddings complexe)"""
    if not chunks_data:
        raise ValueError("Nu există chunk-uri de procesat")
    
    collection = get_or_create_collection(collection_name)
    
    # Pregătim datele pentru stocare
    documents = []
    metadatas = []
    ids = []
    
    for i, chunk_data in enumerate(chunks_data):
        content = chunk_data["content"]
        metadata = chunk_data["metadata"]
        
        # Creăm un ID unic pentru chunk
        chunk_id = f"{source_file}_{metadata.get('chunk_id', f'chunk_{i}')}"
        
        documents.append(content)
        ids.append(chunk_id)
        
        # Metadata pentru ChromaDB
        chunk_metadata = {
            "source": source_file,
            "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
            "original_source": metadata.get("original_source", ""),
            "created_at": metadata.get("created_at", datetime.now().isoformat()),
            "content_length": len(content),
            "chunk_index": i,
            "keywords": ", ".join(extract_keywords(content, 5))  # Salvăm keywords pentru căutare
        }
        metadatas.append(chunk_metadata)
    
    # Stocăm în ChromaDB (fără embeddings - ChromaDB va folosi default embeddings)
    logger.info(f"💾 Stocare chunk-uri în colecția '{collection_name}'...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"✅ {len(chunks_data)} chunk-uri stocate cu succes!")
    return len(chunks_data)

def search_relevant_chunks(collection_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Caută chunk-uri relevante pentru query (folosind ChromaDB default search)"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Colecția '{collection_name}' nu există")
    
    # Căutăm chunk-uri similare folosind ChromaDB default search
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Formatăm rezultatele
    relevant_chunks = []
    if results["documents"] and results["documents"][0]:
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0], 
            results["distances"][0]
        )):
            # Convertim distanța în scor de similaritate (0-1)
            similarity_score = max(0, 1 - distance)
            
            # Calculăm și un scor suplimentar bazat pe keywords
            keyword_similarity = calculate_text_similarity(query, doc)
            combined_score = (similarity_score + keyword_similarity) / 2
            
            relevant_chunks.append({
                "content": doc,
                "meta": metadata,
                "score": combined_score,
                "rank": i + 1,
                "match_type": "semantic" if similarity_score > 0.5 else "keyword"
            })
    
    # Sortăm după scorul combinat
    relevant_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    return relevant_chunks

async def generate_answer_with_context(query: str, relevant_chunks: List[Dict[str, Any]], temperature: float = 0.2) -> str:
    """Generează răspuns folosind Google Gemini cu context din chunk-uri"""
    if not relevant_chunks:
        return "Nu am găsit informații relevante pentru această întrebare."
    
    # Construim contextul din chunk-uri
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        source = chunk["meta"].get("original_source", "Necunoscută")
        content = chunk["content"][:800]  # Limităm lungimea pentru a nu depăși limitele
        score = chunk["score"]
        context_parts.append(f"[Sursă {i}: {source} - Relevanță: {score:.1%}]\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # Prompt optimizat pentru Gemini
    prompt = f"""Ești un asistent AI specializat în analiza documentelor JSON. 

Bazându-te EXCLUSIV pe informațiile din contextul de mai jos, răspunde la întrebarea utilizatorului într-un mod clar, concis și util.

CONTEXT:
{context}

ÎNTREBARE: {query}

INSTRUCȚIUNI:
- Răspunde doar pe baza informațiilor din context
- Dacă informațiile nu sunt suficiente, spune acest lucru
- Fii precis și oferă exemple concrete din context când este posibil
- Organizează răspunsul într-un mod logic și ușor de înțeles
- Nu inventa informații care nu sunt în context
- Menționează sursa informațiilor când este relevant

RĂSPUNS:"""
    
    try:
        # Configurăm parametrii de generare
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=64,
            max_output_tokens=1000,
        )
        
        response = genai_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Eroare la generarea răspunsului cu Gemini: {str(e)}")
        return f"Eroare la generarea răspunsului: {str(e)}"

# Endpoints API (identice cu versiunea complexă)

@app.on_event("startup")
async def startup_event():
    """Inițializare la pornirea aplicației"""
    await initialize_services()

@app.get("/health")
async def health_check():
    """Verificare sănătate API"""
    return {
        "status": "healthy",
        "version": "simplified",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "chromadb": chroma_client is not None,
            "gemini": genai_model is not None,
            "embeddings": "ChromaDB default (simple)"
        }
    }

@app.get("/collections", response_model=List[str])
async def list_collections():
    """Listează toate colecțiile disponibile"""
    try:
        collections = chroma_client.list_collections()
        collection_names = [col.name for col in collections]
        logger.debug(f"📁 Găsite {len(collection_names)} colecții")
        return collection_names
    except Exception as e:
        logger.error(f"Eroare la listarea colecțiilor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}")
async def create_collection(collection_name: str):
    """Creează o colecție nouă"""
    try:
        # Validare nume colecție
        if not collection_name.replace("_", "").isalnum():
            raise HTTPException(status_code=400, detail="Numele colecției poate conține doar litere, cifre și underscore")
        
        # Verificăm dacă colecția există deja
        try:
            chroma_client.get_collection(name=collection_name)
            raise HTTPException(status_code=409, detail=f"Colecția '{collection_name}' există deja")
        except Exception:
            # Colecția nu există, o putem crea
            pass
        
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"created_at": datetime.now().isoformat()}
        )
        
        logger.info(f"📁 Colecție creată: {collection_name}")
        return {"message": f"Colecția '{collection_name}' a fost creată cu succes"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la crearea colecției: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Șterge o colecție și toate documentele ei"""
    try:
        chroma_client.delete_collection(name=collection_name)
        logger.info(f"🗑️ Colecție ștearsă: {collection_name}")
        return {"message": f"Colecția '{collection_name}' a fost ștearsă cu succes"}
    except Exception as e:
        logger.error(f"Eroare la ștergerea colecției: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Colecția '{collection_name}' nu există")

@app.get("/collections/{collection_name}/documents", response_model=List[DocumentInfo])
async def list_documents(collection_name: str):
    """Listează documentele dintr-o colecție"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        # Obținem toate documentele și grupăm după sursă
        all_docs = collection.get(include=["metadatas"])
        
        # Grupăm după sursă
        sources = {}
        for metadata in all_docs["metadatas"]:
            source = metadata.get("source", "Necunoscută")
            if source not in sources:
                sources[source] = {
                    "source": source,
                    "doc_count": 0,
                    "created_at": metadata.get("created_at", ""),
                    "file_size": metadata.get("file_size")
                }
            sources[source]["doc_count"] += 1
        
        result = list(sources.values())
        logger.debug(f"📄 Găsite {len(result)} surse în colecția {collection_name}")
        return result
        
    except Exception as e:
        logger.error(f"Eroare la listarea documentelor: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Colecția '{collection_name}' nu există")

@app.post("/collections/{collection_name}/upload")
async def upload_document(collection_name: str, file: UploadFile = File(...)):
    """Încarcă și procesează un fișier JSON chunkizat"""
    start_time = time.time()
    
    try:
        # Validări fișier
        if not file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="Doar fișierele JSON sunt acceptate")
        
        # Verifică dimensiunea fișierului
        content = await file.read()
        if len(content) > CONFIG["MAX_FILE_SIZE"]:
            raise HTTPException(status_code=413, detail=f"Fișierul este prea mare. Maximum {CONFIG['MAX_FILE_SIZE']//1024//1024}MB")
        
        # Salvăm temporar fișierul
        temp_file_path = f"temp_{int(time.time())}_{file.filename}"
        async with aiofiles.open(temp_file_path, 'wb') as f:
            await f.write(content)
        
        # Validăm formatul JSON
        is_valid, error_msg, chunks_count = validate_json_format(temp_file_path)
        if not is_valid:
            os.unlink(temp_file_path)  # Ștergem fișierul temporar
            raise HTTPException(status_code=400, detail=f"Format JSON invalid: {error_msg}")
        
        # Procesăm chunk-urile
        logger.info(f"📄 Procesare fișier: {file.filename} ({chunks_count} chunk-uri)")
        chunks_data = process_json_chunks(temp_file_path)
        
        # Stocăm în ChromaDB
        stored_chunks = await process_and_store_chunks(collection_name, chunks_data, file.filename)
        
        # Curățăm fișierul temporar
        os.unlink(temp_file_path)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Fișier procesat cu succes în {processing_time:.2f}s: {stored_chunks} chunk-uri")
        
        return {
            "message": f"Fișierul '{file.filename}' a fost procesat cu succes",
            "filename": file.filename,
            "chunks_count": stored_chunks,
            "processing_time": f"{processing_time:.2f}s"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la încărcarea documentului: {str(e)}")
        # Curățăm fișierul temporar în caz de eroare
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}/documents")
async def delete_document(collection_name: str, request: DeleteDocumentRequest):
    """Șterge un document din colecție"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        # Găsim toate chunk-urile din acest document
        results = collection.get(
            where={"source": request.source},
            include=["metadatas"]
        )
        
        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"Documentul '{request.source}' nu a fost găsit")
        
        # Ștergem toate chunk-urile
        collection.delete(ids=results["ids"])
        
        logger.info(f"🗑️ Document șters: {request.source} ({len(results['ids'])} chunk-uri)")
        
        return {
            "message": f"Documentul '{request.source}' a fost șters cu succes",
            "deleted_chunks": len(results["ids"])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la ștergerea documentului: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}/generate", response_model=QueryResponse)
async def generate_answer(collection_name: str, request: QueryRequest):
    """Generează răspuns pe baza unei întrebări și a documentelor din colecție"""
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Procesare întrebare: '{request.query[:50]}...' în colecția '{collection_name}'")
        
        # Căutăm chunk-uri relevante
        relevant_chunks = search_relevant_chunks(
            collection_name=collection_name,
            query=request.query,
            top_k=request.top_k_docs
        )
        
        if not relevant_chunks:
            return QueryResponse(
                query=request.query,
                answer="Nu am găsit informații relevante pentru această întrebare în documentele disponibile.",
                documents=[],
                metadata={
                    "processing_time": f"{time.time() - start_time:.2f}s",
                    "chunks_found": 0,
                    "collection": collection_name,
                    "search_method": "simplified"
                }
            )
        
        # Generăm răspuns cu Gemini
        answer = await generate_answer_with_context(
            query=request.query,
            relevant_chunks=relevant_chunks,
            temperature=request.temperature
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Răspuns generat în {processing_time:.2f}s cu {len(relevant_chunks)} chunk-uri")
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            documents=relevant_chunks,
            metadata={
                "processing_time": f"{processing_time:.2f}s",
                "chunks_found": len(relevant_chunks),
                "collection": collection_name,
                "temperature": request.temperature,
                "search_method": "simplified"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la generarea răspunsului: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/search")
async def search_documents(collection_name: str, query: str, top_k: int = 5):
    """Caută documente fără generare de răspuns"""
    try:
        relevant_chunks = search_relevant_chunks(collection_name, query, top_k)
        
        return {
            "query": query,
            "results": relevant_chunks,
            "total_found": len(relevant_chunks),
            "search_method": "simplified"
        }
        
    except Exception as e:
        logger.error(f"Eroare la căutarea documentelor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Configurare pentru rulare
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8070))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info("=" * 50)
    logger.info("🚀 RAG API pentru JSON Chunkizat (SIMPLIFICAT)")
    logger.info("=" * 50)
    logger.info(f"📡 Server: http://{host}:{port}")
    logger.info(f"📚 Docs: http://{host}:{port}/docs")
    logger.info(f"🔧 Debug: {debug}")
    logger.info(f"⚡ Versiune: Simplificată (fără sentence-transformers)")
    logger.info("=" * 50)
    
    uvicorn.run(
        "rag_api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )