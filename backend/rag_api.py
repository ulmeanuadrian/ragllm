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
import re
from collections import Counter
from utils import (
    process_json_chunks, 
    validate_json_format, 
    get_json_statistics,
    preview_json_chunks
)
from gemini_generator import GeminiGenerator

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
    title="RAG API - JSON Chunkizat OPTIMIZAT",
    description="API pentru Retrieval-Augmented Generation cu fișiere JSON chunkizate - Versiune optimizată pentru căutare îmbunătățită",
    version="3.0.0",
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

# Configurări globale OPTIMIZATE
CONFIG = {
    "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
    "GENERATIVE_MODEL": os.getenv("GENERATIVE_MODEL", "gemini-2.5-flash"),
    "MAX_FILE_SIZE": int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024,
    "DEFAULT_TOP_K": int(os.getenv("DEFAULT_TOP_K", 10)),  # Creștem de la 5 la 10
    "DEFAULT_TEMPERATURE": float(os.getenv("DEFAULT_TEMPERATURE", 0.3)),  # Creștem de la 0.2
    "SIMILARITY_THRESHOLD": float(os.getenv("SIMILARITY_THRESHOLD", 0.15)),  # Threshold mai mic pentru mai multe rezultate
    "MAX_CHUNKS_FOR_CONTEXT": int(os.getenv("MAX_CHUNKS_FOR_CONTEXT", 10)),
    "ENABLE_HYBRID_SEARCH": os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
}

# Instanțe globale
chroma_client = None
gemini_generator = None

# Modele Pydantic pentru validare - ACTUALIZATE
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k_docs: int = Field(default=10, ge=1, le=20)  # Creștem default de la 5 la 10
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)  # Creștem de la 0.2
    use_hybrid_search: bool = Field(default=True)  # Nou: activează căutarea hibridă
    similarity_threshold: float = Field(default=0.15, ge=0.0, le=1.0)  # Nou: threshold configurabil

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

# Funcții helper pentru embeddings și căutare îmbunătățită
def extract_keywords_advanced(text: str, max_keywords: int = 15) -> List[str]:
    """
    Extrage cuvinte cheie avansate din text cu normalizare și filtrare îmbunătățită.
    """
    # Normalizăm textul
    text_normalized = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text_normalized.split()
    
    # Stop words extinse în română și engleză
    stop_words = {
        # Română
        'și', 'în', 'la', 'de', 'cu', 'pe', 'din', 'pentru', 'este', 'sunt', 'a', 'al', 'ale',
        'că', 'să', 'se', 'nu', 'mai', 'dar', 'sau', 'dacă', 'când', 'cum', 'unde', 'care',
        'această', 'acest', 'aceasta', 'acesta', 'unei', 'unui', 'o', 'un', 'am', 'ai', 'au',
        'avea', 'fi', 'fost', 'fiind', 'va', 'vor', 'foarte', 'mult', 'puțin', 'către', 'despre',
        # Engleză
        'and', 'in', 'to', 'of', 'with', 'on', 'from', 'for', 'is', 'are', 'the', 'a', 'an',
        'that', 'this', 'it', 'be', 'have', 'has', 'do', 'does', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'must', 'shall', 'at', 'by', 'up', 'as', 'if',
        'what', 'when', 'where', 'who', 'why', 'how', 'which', 'than', 'or', 'but', 'not',
        'was', 'were', 'been', 'being', 'had', 'having', 'very', 'more', 'most', 'other',
        'some', 'all', 'any', 'each', 'every', 'few', 'many', 'much', 'several', 'such'
    }
    
    # Filtrăm cuvintele și calculăm frecvența
    word_freq = Counter()
    for word in words:
        if (len(word) > 2 and 
            word not in stop_words and 
            not word.isdigit() and
            word.isalpha()):  # Doar cuvinte cu litere
            word_freq[word] += 1
    
    # Sortăm după frecvență și returnăm cele mai comune
    keywords = [word for word, freq in word_freq.most_common(max_keywords)]
    
    return keywords

def calculate_hybrid_similarity(query: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Calculează similaritatea hibridă folosind multiple metode.
    """
    if not query or not content:
        return {'final': 0.0, 'keyword': 0.0, 'semantic': 0.0, 'metadata': 0.0}
    
    query_lower = query.lower()
    content_lower = content.lower()
    
    # 1. KEYWORD SIMILARITY - îmbunătățită
    query_keywords = set(extract_keywords_advanced(query, 10))
    content_keywords = set(extract_keywords_advanced(content, 20))
    
    if query_keywords and content_keywords:
        intersection = query_keywords.intersection(content_keywords)
        union = query_keywords.union(content_keywords)
        keyword_similarity = len(intersection) / len(union) if union else 0.0
        
        # Bonus pentru potriviri exacte
        exact_matches = sum(1 for keyword in query_keywords if keyword in content_lower)
        keyword_bonus = min(0.3, exact_matches * 0.1)
        keyword_similarity += keyword_bonus
    else:
        keyword_similarity = 0.0
    
    # 2. PHRASE SIMILARITY - căutare de fraze și sub-fraze
    phrase_similarity = 0.0
    
    # Căutăm fraze exacte din query în content
    query_phrases = [phrase.strip() for phrase in query.split() if len(phrase.strip()) > 3]
    for phrase in query_phrases:
        if phrase.lower() in content_lower:
            phrase_similarity += min(0.2, len(phrase) / 30)
    
    # Căutăm fraze mai lungi (2-3 cuvinte)
    query_words = query.split()
    for i in range(len(query_words) - 1):
        bigram = ' '.join(query_words[i:i+2]).lower()
        if len(bigram) > 6 and bigram in content_lower:
            phrase_similarity += 0.15
        
        if i < len(query_words) - 2:
            trigram = ' '.join(query_words[i:i+3]).lower()
            if len(trigram) > 10 and trigram in content_lower:
                phrase_similarity += 0.25
    
    phrase_similarity = min(1.0, phrase_similarity)
    
    # 3. FUZZY MATCHING - pentru potriviri aproximative
    fuzzy_similarity = 0.0
    
    # Căutăm variante ale cuvintelor cheie (plurale, conjugări simple)
    for keyword in query_keywords:
        if len(keyword) > 4:
            # Variante simple
            variants = [
                keyword + 's',     # plural engleza
                keyword + 'e',     # variante simple
                keyword + 'i',     # plural romana
                keyword + 'le',    # articol hotarat
                keyword + 'ul',    # articol hotarat masculin
                keyword + 'a',     # feminin
            ]
            
            for variant in variants:
                if variant in content_lower:
                    fuzzy_similarity += 0.05
    
    fuzzy_similarity = min(0.3, fuzzy_similarity)
    
    # 4. METADATA SIMILARITY - dacă avem metadata
    metadata_similarity = 0.0
    if metadata:
        metadata_text = ' '.join(str(v) for v in metadata.values()).lower()
        for keyword in query_keywords:
            if keyword in metadata_text:
                metadata_similarity += 0.1
    
    metadata_similarity = min(0.2, metadata_similarity)
    
    # 5. POSITION BONUS - cuvintele la început sunt mai importante
    position_bonus = 0.0
    content_start = content_lower[:300]
    for keyword in query_keywords:
        if keyword in content_start:
            position_bonus += 0.05
    
    position_bonus = min(0.2, position_bonus)
    
    # 6. LENGTH NORMALIZATION - preferăm documente cu lungime rezonabilă
    length_factor = 1.0
    content_length = len(content)
    if content_length < 50:
        length_factor = 0.5  # Penalizăm documentele foarte scurte
    elif content_length > 5000:
        length_factor = 0.8  # Penalizăm ușor documentele foarte lungi
    
    # Calculăm scorul final cu ponderi optimizate
    weights = {
        'keyword': 0.35,      # 35% - keyword matching
        'phrase': 0.30,       # 30% - phrase matching
        'fuzzy': 0.15,        # 15% - fuzzy matching
        'metadata': 0.10,     # 10% - metadata matching
        'position': 0.10      # 10% - position bonus
    }
    
    scores = {
        'keyword': min(1.0, keyword_similarity),
        'phrase': phrase_similarity,
        'fuzzy': fuzzy_similarity,
        'metadata': metadata_similarity,
        'position': position_bonus,
        'semantic': 0.0  # Pentru compatibilitate (nu implementăm embeddings aici)
    }
    
    # Calculăm scorul final
    final_score = sum(scores[key] * weights[key] for key in weights if key in scores)
    final_score *= length_factor  # Aplicăm factorul de lungime
    final_score = min(1.0, final_score)
    
    scores['final'] = final_score
    scores['length_factor'] = length_factor
    
    return scores

async def initialize_services():
    """Inițializează serviciile necesare pentru RAG cu optimizări"""
    global chroma_client, gemini_generator
    
    try:
        logger.info("🚀 Inițializare servicii RAG OPTIMIZAT...")
        
        # 1. Inițializare ChromaDB cu configurări optimizate
        logger.info("📊 Conectare la ChromaDB cu configurări optimizate...")
        persist_dir = Path(CONFIG["CHROMA_PERSIST_DIR"])
        persist_dir.mkdir(exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                # Configurări optimizate pentru performanță
                chroma_server_host=None,
                chroma_server_http_port=None,
                chroma_server_ssl_enabled=False,
                chroma_server_headers=None,
                chroma_server_cors_allow_origins=[]
            )
        )
        logger.info(f"✅ ChromaDB optimizat inițializat în: {persist_dir}")
        
        # 2. Inițializare Gemini Generator optimizat
        logger.info("🤖 Inițializare Gemini Generator optimizat...")
        gemini_generator = GeminiGenerator(model_name=CONFIG["GENERATIVE_MODEL"])
        logger.info("✅ Gemini Generator optimizat inițializat cu succes")
        
        logger.info("🎉 Toate serviciile optimizate au fost inițializate cu succes!")
        logger.info(f"⚙️ Configurări active:")
        logger.info(f"   - Top K documente: {CONFIG['DEFAULT_TOP_K']}")
        logger.info(f"   - Temperatură: {CONFIG['DEFAULT_TEMPERATURE']}")
        logger.info(f"   - Threshold similaritate: {CONFIG['SIMILARITY_THRESHOLD']}")
        logger.info(f"   - Căutare hibridă: {CONFIG['ENABLE_HYBRID_SEARCH']}")
        
    except Exception as e:
        logger.error(f"❌ Eroare la inițializarea serviciilor optimizate: {str(e)}")
        raise

def get_or_create_collection(collection_name: str):
    """Obține sau creează o colecție ChromaDB cu configurări optimizate"""
    try:
        # Încercăm să obținem colecția existentă
        collection = chroma_client.get_collection(name=collection_name)
        logger.debug(f"📁 Colecție existentă găsită: {collection_name}")
    except Exception:
        # Dacă nu există, o creăm cu metadata îmbunătățită
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={
                "created_at": datetime.now().isoformat(),
                "version": "3.0.0",
                "search_optimized": True,
                "hybrid_search_enabled": CONFIG["ENABLE_HYBRID_SEARCH"]
            }
        )
        logger.info(f"📁 Colecție nouă optimizată creată: {collection_name}")
    
    return collection

async def process_and_store_chunks_optimized(collection_name: str, chunks_data: List[Dict[str, Any]], source_file: str):
    """Procesează și stochează chunk-urile în ChromaDB cu optimizări pentru căutare"""
    if not chunks_data:
        raise ValueError("Nu există chunk-uri de procesat")
    
    collection = get_or_create_collection(collection_name)
    
    # Pregătim datele pentru stocare cu îmbunătățiri
    documents = []
    metadatas = []
    ids = []
    
    for i, chunk_data in enumerate(chunks_data):
        content = chunk_data["content"]
        metadata = chunk_data["metadata"]
        
        # Creăm un ID unic pentru chunk
        chunk_id = f"{source_file}_{metadata.get('chunk_id', f'chunk_{i}')}"
        
        # Curățăm și normalizăm conținutul
        cleaned_content = re.sub(r'\s+', ' ', content.strip())
        
        documents.append(cleaned_content)
        ids.append(chunk_id)
        
        # Extragem keywords pentru căutare îmbunătățită
        keywords = extract_keywords_advanced(cleaned_content, 15)
        
        # Metadata îmbunătățită pentru ChromaDB
        chunk_metadata = {
            "source": source_file,
            "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
            "original_source": metadata.get("original_source", ""),
            "created_at": metadata.get("created_at", datetime.now().isoformat()),
            "content_length": len(cleaned_content),
            "word_count": len(cleaned_content.split()),
            "chunk_index": i,
            "keywords": ", ".join(keywords[:10]),  # Top 10 keywords
            "keywords_count": len(keywords),
            "file_source": source_file,
            "processing_version": "3.0.0",
            "language_detected": "auto",  # Placeholder pentru detectia de limbă
            "content_type": "json_chunk",
            # Statistici pentru ranking
            "avg_word_length": sum(len(word) for word in cleaned_content.split()) / len(cleaned_content.split()) if cleaned_content.split() else 0,
            "sentence_count": len([s for s in cleaned_content.split('.') if s.strip()]),
        }
        metadatas.append(chunk_metadata)
    
    # Stocăm în ChromaDB
    logger.info(f"💾 Stocare optimizată: {len(chunks_data)} chunk-uri în colecția '{collection_name}'...")
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"✅ {len(chunks_data)} chunk-uri stocate și optimizate cu succes!")
    return len(chunks_data)

def search_relevant_chunks_optimized(collection_name: str, query: str, top_k: int = 10, use_hybrid: bool = True, threshold: float = 0.15) -> List[Dict[str, Any]]:
    """
    Caută chunk-uri relevante cu algoritm optimizat și căutare hibridă.
    
    Args:
        collection_name: Numele colecției
        query: Query-ul de căutare
        top_k: Numărul de rezultate dorite
        use_hybrid: Dacă să folosească căutarea hibridă
        threshold: Pragul de similaritate minim
        
    Returns:
        Lista de chunk-uri relevante ordonate după relevanță
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Colecția '{collection_name}' nu există")
    
    logger.info(f"🔍 Căutare optimizată în '{collection_name}': '{query[:50]}...' (hybrid: {use_hybrid})")
    
    # Obținem mai multe rezultate pentru re-ranking
    search_limit = min(top_k * 3, 50)  # Căutăm de 3x mai multe pentru re-ranking
    
    # 1. CĂUTARE SEMANTICĂ cu ChromaDB (embedding-based)
    semantic_results = collection.query(
        query_texts=[query],
        n_results=search_limit,
        include=["documents", "metadatas", "distances"]
    )
    
    # 2. Procesăm rezultatele și calculăm scoruri hibride
    hybrid_results = []
    
    if semantic_results["documents"] and semantic_results["documents"][0]:
        for i, (doc, metadata, distance) in enumerate(zip(
            semantic_results["documents"][0],
            semantic_results["metadatas"][0], 
            semantic_results["distances"][0]
        )):
            # Convertim distanța ChromaDB în scor de similaritate
            chroma_similarity = max(0, 1 - distance)
            
            if use_hybrid:
                # Calculăm scorurile hibride
                hybrid_scores = calculate_hybrid_similarity(query, doc, metadata)
                
                # Combinăm scorul semantic cu cel hibrid
                combined_score = (chroma_similarity * 0.4) + (hybrid_scores['final'] * 0.6)
                
                match_type = "hybrid"
                detail_scores = {
                    'chroma_semantic': chroma_similarity,
                    'keyword': hybrid_scores['keyword'],
                    'phrase': hybrid_scores['phrase'],
                    'fuzzy': hybrid_scores.get('fuzzy', 0.0),
                    'metadata': hybrid_scores['metadata'],
                    'position': hybrid_scores['position'],
                    'final_hybrid': hybrid_scores['final']
                }
            else:
                # Folosim doar scorul semantic
                combined_score = chroma_similarity
                match_type = "semantic"
                detail_scores = {
                    'chroma_semantic': chroma_similarity
                }
            
            # Aplicăm threshold-ul
            if combined_score >= threshold:
                hybrid_results.append({
                    "content": doc,
                    "meta": metadata,
                    "score": combined_score,
                    "rank": i + 1,
                    "match_type": match_type,
                    "distance": distance,
                    "detail_scores": detail_scores
                })
    
    # 3. SORTARE și FILTRARE FINALĂ
    # Sortăm după scorul combinat
    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Luăm doar top_k rezultate
    final_results = hybrid_results[:top_k]
    
    # 4. LOGGING pentru debugging
    logger.info(f"🎯 Găsite {len(final_results)} rezultate relevante din {len(hybrid_results)} candidați")
    
    if final_results:
        top_score = final_results[0]["score"]
        avg_score = sum(r["score"] for r in final_results) / len(final_results)
        logger.info(f"📊 Scoruri: max={top_score:.3f}, avg={avg_score:.3f}, threshold={threshold}")
        
        # Logăm top 3 rezultate pentru debugging
        for i, result in enumerate(final_results[:3]):
            logger.debug(f"🏆 #{i+1}: score={result['score']:.3f}, type={result['match_type']}, content_preview='{result['content'][:100]}...'")
    else:
        logger.warning(f"⚠️ Nu s-au găsit rezultate peste threshold-ul {threshold}")
        
        # Dacă nu găsim nimic, încercăm cu threshold mai mic
        if threshold > 0.05:
            logger.info("🔄 Reîncerc cu threshold mai mic...")
            return search_relevant_chunks_optimized(collection_name, query, top_k, use_hybrid, threshold=0.05)
    
    return final_results

async def generate_answer_with_context_optimized(query: str, relevant_chunks: List[Dict[str, Any]], temperature: float = 0.3) -> str:
    """Generează răspuns optimizat folosind Gemini cu context îmbunătățit"""
    if not relevant_chunks:
        return "Nu am găsit informații relevante pentru această întrebare în documentele disponibile. Încercați să reformulați întrebarea sau să verificați dacă colecția conține informațiile căutate."
    
    try:
        # Folosim generatorul optimizat
        answer = gemini_generator.generate_response(
            query=query,
            context_docs=relevant_chunks,
            temperature=temperature,
            max_output_tokens=1500,  # Creștem pentru răspunsuri mai detaliate
            top_k=40,
            top_p=0.95
        )
        
        return answer
        
    except Exception as e:
        logger.error(f"Eroare la generarea răspunsului optimizat: {str(e)}")
        return f"Eroare la generarea răspunsului: {str(e)}. Verificați configurarea serviciului AI."

# Endpoints API - ACTUALIZAȚI cu optimizări

@app.on_event("startup")
async def startup_event():
    """Inițializare optimizată la pornirea aplicației"""
    await initialize_services()

@app.get("/health")
async def health_check():
    """Verificare sănătate API cu informații despre optimizări"""
    cache_info = gemini_generator.get_cache_info() if gemini_generator else {}
    
    return {
        "status": "healthy",
        "version": "3.0.0 - OPTIMIZED",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "chromadb": chroma_client is not None,
            "gemini": gemini_generator is not None,
            "hybrid_search": CONFIG["ENABLE_HYBRID_SEARCH"]
        },
        "configuration": {
            "default_top_k": CONFIG["DEFAULT_TOP_K"],
            "default_temperature": CONFIG["DEFAULT_TEMPERATURE"],
            "similarity_threshold": CONFIG["SIMILARITY_THRESHOLD"],
            "max_chunks_context": CONFIG["MAX_CHUNKS_FOR_CONTEXT"]
        },
        "cache_stats": cache_info,
        "optimizations": [
            "Hybrid search (semantic + keyword)",
            "Advanced similarity scoring",
            "Improved query expansion",
            "Enhanced context generation",
            "Optimized cache management",
            "Better error handling"
        ]
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
    """Creează o colecție nouă optimizată"""
    try:
        # Validare nume colecție îmbunătățită
        if not re.match(r'^[a-zA-Z0-9_]{1,50}, collection_name):
            raise HTTPException(
                status_code=400, 
                detail="Numele colecției poate conține doar litere, cifre și underscore (_), maximum 50 caractere"
            )
        
        # Verificăm dacă colecția există deja
        try:
            chroma_client.get_collection(name=collection_name)
            raise HTTPException(status_code=409, detail=f"Colecția '{collection_name}' există deja")
        except Exception:
            # Colecția nu există, o putem crea
            pass
        
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={
                "created_at": datetime.now().isoformat(),
                "version": "3.0.0",
                "optimized": True,
                "hybrid_search": CONFIG["ENABLE_HYBRID_SEARCH"]
            }
        )
        
        logger.info(f"📁 Colecție optimizată creată: {collection_name}")
        return {
            "message": f"Colecția '{collection_name}' a fost creată cu succes",
            "optimizations_enabled": True,
            "hybrid_search": CONFIG["ENABLE_HYBRID_SEARCH"]
        }
        
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
        
        # Curățăm și cache-ul Gemini dacă este necesar
        if gemini_generator:
            gemini_generator.clear_cache()
        
        logger.info(f"🗑️ Colecție ștearsă: {collection_name}")
        return {"message": f"Colecția '{collection_name}' a fost ștearsă cu succes"}
    except Exception as e:
        logger.error(f"Eroare la ștergerea colecției: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Colecția '{collection_name}' nu există")

@app.get("/collections/{collection_name}/documents", response_model=List[DocumentInfo])
async def list_documents(collection_name: str):
    """Listează documentele dintr-o colecție cu statistici îmbunătățite"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        # Obținem toate documentele și grupăm după sursă
        all_docs = collection.get(include=["metadatas"])
        
        # Grupăm după sursă cu statistici îmbunătățite
        sources = {}
        for metadata in all_docs["metadatas"]:
            source = metadata.get("source", "Necunoscută")
            if source not in sources:
                sources[source] = {
                    "source": source,
                    "doc_count": 0,
                    "created_at": metadata.get("created_at", ""),
                    "file_size": metadata.get("file_size"),
                    "total_words": 0,
                    "avg_chunk_length": 0,
                    "keywords_diversity": 0
                }
            
            sources[source]["doc_count"] += 1
            sources[source]["total_words"] += metadata.get("word_count", 0)
            
            # Calculăm statistici suplimentare
            if sources[source]["doc_count"] > 0:
                sources[source]["avg_chunk_length"] = metadata.get("content_length", 0)
                sources[source]["keywords_diversity"] = metadata.get("keywords_count", 0)
        
        result = list(sources.values())
        logger.debug(f"📄 Găsite {len(result)} surse în colecția {collection_name}")
        return result
        
    except Exception as e:
        logger.error(f"Eroare la listarea documentelor: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Colecția '{collection_name}' nu există")

@app.post("/collections/{collection_name}/upload")
async def upload_document(collection_name: str, file: UploadFile = File(...)):
    """Încarcă și procesează un fișier JSON chunkizat cu optimizări"""
    start_time = time.time()
    
    try:
        # Validări fișier îmbunătățite
        if not file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="Doar fișierele JSON sunt acceptate")
        
        # Verifică dimensiunea fișierului
        content = await file.read()
        if len(content) > CONFIG["MAX_FILE_SIZE"]:
            raise HTTPException(
                status_code=413, 
                detail=f"Fișierul este prea mare. Maximum {CONFIG['MAX_FILE_SIZE']//1024//1024}MB"
            )
        
        # Salvăm temporar fișierul
        temp_file_path = f"temp_{int(time.time())}_{file.filename}"
        async with aiofiles.open(temp_file_path, 'wb') as f:
            await f.write(content)
        
        # Validăm formatul JSON
        is_valid, error_msg, chunks_count = validate_json_format(temp_file_path)
        if not is_valid:
            os.unlink(temp_file_path)  # Ștergem fișierul temporar
            raise HTTPException(status_code=400, detail=f"Format JSON invalid: {error_msg}")
        
        # Procesăm chunk-urile cu optimizări
        logger.info(f"📄 Procesare optimizată fișier: {file.filename} ({chunks_count} chunk-uri)")
        chunks_data = process_json_chunks(temp_file_path)
        
        # Stocăm în ChromaDB cu optimizări
        stored_chunks = await process_and_store_chunks_optimized(collection_name, chunks_data, file.filename)
        
        # Curățăm fișierul temporar
        os.unlink(temp_file_path)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Fișier procesat cu optimizări în {processing_time:.2f}s: {stored_chunks} chunk-uri")
        
        return {
            "message": f"Fișierul '{file.filename}' a fost procesat cu succes",
            "filename": file.filename,
            "chunks_count": stored_chunks,
            "processing_time": f"{processing_time:.2f}s",
            "optimizations_applied": [
                "Enhanced keyword extraction",
                "Improved metadata enrichment", 
                "Advanced content normalization",
                "Optimized indexing for hybrid search"
            ]
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
    """Șterge un document din colecție cu cache cleanup"""
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
        
        # Curățăm cache-ul pentru această colecție
        if gemini_generator:
            gemini_generator.clear_cache()
        
        logger.info(f"🗑️ Document șters: {request.source} ({len(results['ids'])} chunk-uri)")
        
        return {
            "message": f"Documentul '{request.source}' a fost șters cu succes",
            "deleted_chunks": len(results["ids"]),
            "cache_cleared": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la ștergerea documentului: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}/generate", response_model=QueryResponse)
async def generate_answer(collection_name: str, request: QueryRequest):
    """Generează răspuns optimizat pe baza unei întrebări și a documentelor din colecție"""
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Procesare optimizată întrebare: '{request.query[:50]}...' în colecția '{collection_name}'")
        
        # Căutăm chunk-uri relevante cu algoritmul optimizat
        relevant_chunks = search_relevant_chunks_optimized(
            collection_name=collection_name,
            query=request.query,
            top_k=request.top_k_docs,
            use_hybrid=request.use_hybrid_search,
            threshold=request.similarity_threshold
        )
        
        if not relevant_chunks:
            return QueryResponse(
                query=request.query,
                answer="Nu am găsit informații relevante pentru această întrebare în documentele disponibile. Încercați să:\n\n1. Reformulați întrebarea folosind termeni diferiți\n2. Verificați dacă documentele din colecție conțin informațiile căutate\n3. Reduceți pragul de similaritate în cerere\n4. Folosiți termeni mai generali sau mai specifici",
                documents=[],
                metadata={
                    "processing_time": f"{time.time() - start_time:.2f}s",
                    "chunks_found": 0,
                    "collection": collection_name,
                    "search_method": "optimized_hybrid" if request.use_hybrid_search else "semantic",
                    "threshold_used": request.similarity_threshold,
                    "suggestions": [
                        "Reformulați întrebarea",
                        "Verificați conținutul colecției",
                        "Ajustați pragul de similaritate",
                        "Folosiți termeni mai generali"
                    ]
                }
            )
        
        # Generăm răspuns cu Gemini optimizat
        answer = await generate_answer_with_context_optimized(
            query=request.query,
            relevant_chunks=relevant_chunks,
            temperature=request.temperature
        )
        
        processing_time = time.time() - start_time
        
        # Calculăm statistici de calitate
        avg_score = sum(chunk["score"] for chunk in relevant_chunks) / len(relevant_chunks)
        score_distribution = {
            "excellent": len([c for c in relevant_chunks if c["score"] > 0.7]),
            "good": len([c for c in relevant_chunks if 0.4 <= c["score"] <= 0.7]),
            "moderate": len([c for c in relevant_chunks if 0.2 <= c["score"] < 0.4]),
            "low": len([c for c in relevant_chunks if c["score"] < 0.2])
        }
        
        logger.info(f"✅ Răspuns optimizat generat în {processing_time:.2f}s cu {len(relevant_chunks)} chunk-uri (avg_score: {avg_score:.3f})")
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            documents=relevant_chunks,
            metadata={
                "processing_time": f"{processing_time:.2f}s",
                "chunks_found": len(relevant_chunks),
                "collection": collection_name,
                "temperature": request.temperature,
                "search_method": "optimized_hybrid" if request.use_hybrid_search else "semantic",
                "threshold_used": request.similarity_threshold,
                "quality_metrics": {
                    "average_score": round(avg_score, 3),
                    "score_distribution": score_distribution,
                    "top_score": round(relevant_chunks[0]["score"], 3) if relevant_chunks else 0,
                    "score_range": f"{round(relevant_chunks[-1]['score'], 3)}-{round(relevant_chunks[0]['score'], 3)}" if relevant_chunks else "0-0"
                },
                "optimizations_used": [
                    "Hybrid semantic + keyword search",
                    "Advanced similarity scoring",
                    "Query term expansion",
                    "Enhanced context building",
                    "Intelligent result ranking"
                ]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eroare la generarea răspunsului optimizat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/search")
async def search_documents(
    collection_name: str, 
    query: str, 
    top_k: int = 10, 
    use_hybrid: bool = True,
    threshold: float = 0.15
):
    """Caută documente fără generare de răspuns - cu optimizări"""
    try:
        start_time = time.time()
        
        relevant_chunks = search_relevant_chunks_optimized(
            collection_name=collection_name, 
            query=query, 
            top_k=top_k,
            use_hybrid=use_hybrid,
            threshold=threshold
        )
        
        search_time = time.time() - start_time
        
        # Analiză de calitate a rezultatelor
        if relevant_chunks:
            scores = [chunk["score"] for chunk in relevant_chunks]
            quality_analysis = {
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "score_std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
                "high_quality_results": len([s for s in scores if s > 0.5]),
                "relevant_results": len([s for s in scores if s > threshold])
            }
        else:
            quality_analysis = {
                "avg_score": 0,
                "max_score": 0,
                "min_score": 0,
                "score_std": 0,
                "high_quality_results": 0,
                "relevant_results": 0
            }
        
        return {
            "query": query,
            "results": relevant_chunks,
            "total_found": len(relevant_chunks),
            "search_method": "optimized_hybrid" if use_hybrid else "semantic",
            "search_time": f"{search_time:.3f}s",
            "parameters": {
                "top_k": top_k,
                "threshold": threshold,
                "hybrid_search": use_hybrid
            },
            "quality_analysis": quality_analysis,
            "recommendations": [
                "Pentru mai multe rezultate, reduceți threshold-ul",
                "Pentru rezultate mai precise, creșteți threshold-ul", 
                "Încercați căutarea hibridă pentru rezultate mai bune",
                "Reformulați query-ul dacă rezultatele nu sunt satisfăcătoare"
            ] if len(relevant_chunks) < 3 else []
        }
        
    except Exception as e:
        logger.error(f"Eroare la căutarea optimizată documentelor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/stats")
async def get_collection_stats(collection_name: str):
    """Obține statistici detaliate despre o colecție"""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        # Obținem toate documentele cu metadata
        all_docs = collection.get(include=["metadatas", "documents"])
        
        if not all_docs["metadatas"]:
            return {
                "collection": collection_name,
                "total_chunks": 0,
                "message": "Colecția este goală"
            }
        
        # Calculăm statistici detaliate
        total_chunks = len(all_docs["metadatas"])
        total_words = sum(meta.get("word_count", 0) for meta in all_docs["metadatas"])
        total_chars = sum(meta.get("content_length", 0) for meta in all_docs["metadatas"])
        
        # Analiză surse
        sources = {}
        for meta in all_docs["metadatas"]:
            source = meta.get("source", "Unknown")
            if source not in sources:
                sources[source] = {"chunks": 0, "words": 0, "chars": 0}
            sources[source]["chunks"] += 1
            sources[source]["words"] += meta.get("word_count", 0)
            sources[source]["chars"] += meta.get("content_length", 0)
        
        # Top keywords
        all_keywords = []
        for meta in all_docs["metadatas"]:
            keywords = meta.get("keywords", "").split(", ")
            all_keywords.extend([kw.strip() for kw in keywords if kw.strip()])
        
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(20)
        
        # Statistici de lungime
        lengths = [meta.get("content_length", 0) for meta in all_docs["metadatas"]]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        
        return {
            "collection": collection_name,
            "summary": {
                "total_chunks": total_chunks,
                "total_words": total_words,
                "total_characters": total_chars,
                "unique_sources": len(sources),
                "average_chunk_length": round(avg_length, 2),
                "unique_keywords": len(keyword_counts)
            },
            "sources": dict(sources),
            "top_keywords": top_keywords,
            "length_distribution": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "avg": round(avg_length, 2),
                "median": sorted(lengths)[len(lengths)//2] if lengths else 0
            },
            "collection_metadata": collection.metadata,
            "optimization_status": {
                "version": "3.0.0",
                "hybrid_search_ready": True,
                "advanced_scoring": True,
                "enhanced_metadata": True
            }
        }
        
    except Exception as e:
        logger.error(f"Eroare la obținerea statisticilor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache():
    """Curăță cache-ul sistemului"""
    try:
        cleared_count = 0
        if gemini_generator:
            cleared_count = gemini_generator.clear_cache()
        
        return {
            "message": "Cache-ul a fost curățat cu succes",
            "cleared_entries": cleared_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Eroare la curățarea cache-ului: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/info")
async def get_cache_info():
    """Obține informații despre cache"""
    try:
        if gemini_generator:
            cache_info = gemini_generator.get_cache_info()
            return {
                "cache_info": cache_info,
                "status": "active",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "cache_info": {},
                "status": "inactive",
                "message": "Gemini generator nu este inițializat",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Eroare la obținerea informațiilor cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Configurare pentru rulare
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8070))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info("=" * 60)
    logger.info("🚀 RAG API pentru JSON Chunkizat - VERSIUNEA OPTIMIZATĂ 3.0.0")
    logger.info("=" * 60)
    logger.info(f"📡 Server: http://{host}:{port}")
    logger.info(f"📚 Docs: http://{host}:{port}/docs")
    logger.info(f"🔧 Debug: {debug}")
    logger.info(f"⚡ Optimizări active:")
    logger.info(f"   🔍 Căutare hibridă (semantic + keyword)")
    logger.info(f"   🎯 Scor de similaritate avansat") 
    logger.info(f"   📈 Expandare inteligentă de termeni")
    logger.info(f"   🧠 Context îmbunătățit pentru AI")
    logger.info(f"   💾 Cache optimizat cu LRU")
    logger.info(f"   📊 Threshold adaptiv: {CONFIG['SIMILARITY_THRESHOLD']}")
    logger.info(f"   🎛️ Top-K default: {CONFIG['DEFAULT_TOP_K']}")
    logger.info(f"   🌡️ Temperatură default: {CONFIG['DEFAULT_TEMPERATURE']}")
    logger.info("=" * 60)
    
    uvicorn.run(
        "rag_api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )