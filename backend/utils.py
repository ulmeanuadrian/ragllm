import time
import json
import pypdf
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("rag_api")

def process_pdf_chunk(file_path: str, start_page: int, end_page: int) -> str:
    """
    Procesează un subset de pagini dintr-un PDF.
    
    Args:
        file_path: Calea către fișierul PDF
        start_page: Indexul paginii de start (inclusiv)
        end_page: Indexul paginii de final (exclusiv)
        
    Returns:
        Textul extras din paginile specificate
    """
    try:
        reader = pypdf.PdfReader(file_path)
        content = ""
        
        # Asigurăm că end_page nu depășește numărul total de pagini
        end_page = min(end_page, len(reader.pages))
        
        for i in range(start_page, end_page):
            page_text = reader.pages[i].extract_text() or ""
            content += page_text + "\n\n"  # Adăugăm spații între pagini
            
        return content
    except Exception as e:
        logger.error(f"Eroare la procesarea chunk-ului PDF {start_page}-{end_page}: {str(e)}")
        return ""

def process_pdf_optimized(file_path: str, chunk_size: int = 10, max_workers: int = 4) -> str:
    """
    Procesează un fișier PDF în mod optimizat folosind procesare paralelă.
    
    Args:
        file_path: Calea către fișierul PDF
        chunk_size: Numărul de pagini per chunk
        max_workers: Numărul maxim de procese paralele
        
    Returns:
        Textul extras din întregul PDF
    """
    start_time = time.time()
    
    try:
        reader = pypdf.PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info(f"PDF detectat cu {total_pages} pagini", extra={"pages": total_pages})
        
        # Pentru PDF-uri mici, nu folosim procesare paralelă
        if total_pages <= chunk_size:
            content = process_pdf_chunk(file_path, 0, total_pages)
            logger.info(f"PDF procesat secvențial în {time.time() - start_time:.2f} secunde")
            return content
        
        # Pentru PDF-uri mari, folosim procesare paralelă
        chunks = []
        for i in range(0, total_pages, chunk_size):
            chunks.append((i, min(i + chunk_size, total_pages)))
        
        content = ""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Creăm și lansăm task-urile
            futures = [executor.submit(process_pdf_chunk, file_path, start, end) for start, end in chunks]
            
            # Colectăm rezultatele pe măsură ce sunt finalizate
            for future in as_completed(futures):
                content += future.result()
        
        logger.info(f"PDF procesat în paralel în {time.time() - start_time:.2f} secunde", 
                   extra={"duration": f"{time.time() - start_time:.2f}s", "pages": total_pages})
        return content
        
    except Exception as e:
        logger.error(f"Eroare la procesarea PDF: {str(e)}")
        raise e

def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează un fișier JSON care conține chunk-uri predefinite.
    Optimizat pentru fișiere mari și structura specifică de chunk-uri.
    
    Args:
        file_path: Calea către fișierul JSON
        
    Returns:
        Lista de chunk-uri procesate, fiecare cu conținut și metadate
    """
    start_time = time.time()
    chunks_data = []
    
    try:
        # Folosim context manager pentru a asigura închiderea corectă a fișierului
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        # Verificăm dacă JSON-ul conține chunk-uri în formatul așteptat
        chunk_count = 0
        for key, value in json_data.items():
            if key.startswith("chunk_") and isinstance(value, dict):
                if "metadata" in value and "chunk" in value:
                    chunk_count += 1
                    # Extragem metadatele și conținutul într-un format standardizat
                    metadata = {}
                    if isinstance(value["metadata"], str):
                        metadata["source_info"] = value["metadata"]
                    elif isinstance(value["metadata"], dict):
                        metadata = value["metadata"]
                    
                    # Adăugăm chunk-ul în lista de procesare
                    chunks_data.append({
                        "content": value["chunk"],
                        "metadata": metadata,
                        "chunk_id": key
                    })
        
        logger.info(f"Fișier JSON procesat cu succes: {len(chunks_data)} chunk-uri extrase în {time.time() - start_time:.2f} secunde",
                  extra={"file": file_path, "chunks": chunk_count, "duration": f"{time.time() - start_time:.2f}s"})
        
        return chunks_data
    except Exception as e:
        logger.error(f"Eroare la procesarea fișierului JSON: {str(e)}")
        raise e

# Cache pentru interogări frecvente
from collections import OrderedDict
_query_cache = OrderedDict()
MAX_QUERY_CACHE_SIZE = 50

def save_to_query_cache(key: str, value: Any) -> None:
    """
    Adaugă un rezultat în cache-ul de interogări.
    
    Args:
        key: Cheia pentru cache
        value: Valoarea de stocat
    """
    global _query_cache, MAX_QUERY_CACHE_SIZE
    
    # Dacă cache-ul a atins dimensiunea maximă, eliminăm cea mai veche intrare
    if len(_query_cache) >= MAX_QUERY_CACHE_SIZE:
        # Eliminăm prima intrare (cea mai veche) - FIFO
        _query_cache.popitem(last=False)
    
    _query_cache[key] = value
    logger.info(f"Rezultat salvat în cache pentru cheia: {key[:30]}...")

def get_from_query_cache(key: str) -> Tuple[bool, Any]:
    """
    Obține un rezultat din cache-ul de interogări.
    
    Args:
        key: Cheia pentru cache
        
    Returns:
        Tuple (există_în_cache, valoare)
    """
    global _query_cache
    
    if key in _query_cache:
        # Mutăm elementul la sfârșitul dicționarului pentru a-l marca ca recent utilizat (LRU)
        value = _query_cache.pop(key)
        _query_cache[key] = value
        return True, value
    return False, None
