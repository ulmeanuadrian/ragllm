import time
import json
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("rag_api")

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
        
        logger.info(f"JSON încărcat cu succes, verificare format...", extra={"file": file_path})
        
        # Verificăm dacă JSON-ul conține chunk-uri în formatul așteptat
        chunk_count = 0
        valid_chunks = 0
        
        # Sortăm cheile pentru a procesa chunk-urile în ordine
        sorted_keys = sorted([k for k in json_data.keys() if k.startswith("chunk_")], 
                            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 999999)
        
        for key in sorted_keys:
            value = json_data[key]
            chunk_count += 1
            
            if isinstance(value, dict) and "metadata" in value and "chunk" in value:
                valid_chunks += 1
                
                # Parsăm metadatele pentru a extrage informații
                metadata = {
                    "chunk_id": key,
                    "original_source": value["metadata"],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "chunk_index": valid_chunks - 1,
                    "file_type": "json_chunked"
                }
                
                # Verificăm dacă chunk-ul are conținut valid
                chunk_content = value["chunk"].strip()
                if len(chunk_content) < 10:  # Ignorăm chunk-urile prea scurte
                    logger.warning(f"Chunk {key} este prea scurt și va fi ignorat", 
                                 extra={"chunk_length": len(chunk_content)})
                    continue
                
                # Adăugăm chunk-ul în lista de procesare
                chunks_data.append({
                    "content": chunk_content,
                    "metadata": metadata,
                    "chunk_id": key
                })
                
                logger.debug(f"Procesat chunk {key}: {len(chunk_content)} caractere", 
                           extra={"chunk_id": key, "content_length": len(chunk_content)})
            else:
                logger.warning(f"Chunk {key} nu are formatul corect și va fi ignorat", 
                             extra={"chunk_structure": list(value.keys()) if isinstance(value, dict) else type(value).__name__})
        
        if len(chunks_data) == 0:
            raise ValueError(
                f"Fișierul JSON nu conține chunk-uri valide în formatul așteptat. "
                f"Găsite {chunk_count} chunk-uri potențiale, dar niciun chunk valid. "
                f"Formatul așteptat: {{'chunk_0': {{'metadata': '...', 'chunk': '...'}}, ...}}"
            )
        
        logger.info(
            f"Fișier JSON procesat cu succes: {len(chunks_data)} chunk-uri valide din {chunk_count} chunk-uri totale în {time.time() - start_time:.2f} secunde",
            extra={
                "file": file_path, 
                "valid_chunks": len(chunks_data), 
                "total_chunks": chunk_count,
                "duration": f"{time.time() - start_time:.2f}s"
            }
        )
        
        return chunks_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Eroare la parsarea JSON: {str(e)}", extra={"file": file_path})
        raise ValueError(f"Fișierul JSON nu este valid: {str(e)}")
    except FileNotFoundError:
        logger.error(f"Fișierul nu a fost găsit: {file_path}")
        raise ValueError(f"Fișierul nu a fost găsit: {file_path}")
    except Exception as e:
        logger.error(f"Eroare neașteptată la procesarea fișierului JSON: {str(e)}")
        raise e

def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează dacă un fișier JSON are formatul corect pentru chunk-uri.
    
    Args:
        file_path: Calea către fișierul JSON
        
    Returns:
        Tuple (is_valid, error_message, chunks_count)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        if not isinstance(json_data, dict):
            return False, "JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip", 0
        
        chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
        
        if len(chunk_keys) == 0:
            return False, "Nu s-au găsit chunk-uri. Cheile trebuie să înceapă cu 'chunk_'", 0
        
        valid_chunks = 0
        for key in chunk_keys:
            value = json_data[key]
            
            if not isinstance(value, dict):
                return False, f"Chunk-ul '{key}' trebuie să fie un obiect (dicționar)", valid_chunks
            
            if "metadata" not in value:
                return False, f"Chunk-ul '{key}' nu conține câmpul 'metadata'", valid_chunks
            
            if "chunk" not in value:
                return False, f"Chunk-ul '{key}' nu conține câmpul 'chunk'", valid_chunks
            
            if not isinstance(value["chunk"], str) or len(value["chunk"].strip()) < 10:
                return False, f"Chunk-ul '{key}' nu conține text valid (minim 10 caractere)", valid_chunks
            
            valid_chunks += 1
        
        return True, "", valid_chunks
        
    except json.JSONDecodeError as e:
        return False, f"Fișierul JSON nu este valid: {str(e)}", 0
    except FileNotFoundError:
        return False, "Fișierul nu a fost găsit", 0
    except Exception as e:
        return False, f"Eroare la validarea fișierului: {str(e)}", 0

def get_json_statistics(file_path: str) -> Dict[str, Any]:
    """
    Obține statistici despre un fișier JSON chunkizat.
    
    Args:
        file_path: Calea către fișierul JSON
        
    Returns:
        Dicționar cu statistici despre fișier
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        chunk_keys = sorted([k for k in json_data.keys() if k.startswith("chunk_")], 
                           key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 999999)
        
        statistics = {
            "total_chunks": len(chunk_keys),
            "valid_chunks": 0,
            "total_characters": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": float('inf'),
            "max_chunk_length": 0,
            "sources": set(),
            "chunk_ids": []
        }
        
        for key in chunk_keys:
            value = json_data[key]
            
            if isinstance(value, dict) and "metadata" in value and "chunk" in value:
                chunk_content = value["chunk"].strip()
                chunk_length = len(chunk_content)
                
                if chunk_length >= 10:  # Considerăm doar chunk-urile valide
                    statistics["valid_chunks"] += 1
                    statistics["total_characters"] += chunk_length
                    statistics["min_chunk_length"] = min(statistics["min_chunk_length"], chunk_length)
                    statistics["max_chunk_length"] = max(statistics["max_chunk_length"], chunk_length)
                    statistics["sources"].add(value["metadata"])
                    statistics["chunk_ids"].append(key)
        
        if statistics["valid_chunks"] > 0:
            statistics["avg_chunk_length"] = statistics["total_characters"] / statistics["valid_chunks"]
        else:
            statistics["min_chunk_length"] = 0
            
        # Convertim set-ul în listă pentru JSON serialization
        statistics["sources"] = list(statistics["sources"])
        
        return statistics
        
    except Exception as e:
        logger.error(f"Eroare la calcularea statisticilor pentru {file_path}: {str(e)}")
        return {
            "error": str(e),
            "total_chunks": 0,
            "valid_chunks": 0
        }

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

def clear_query_cache() -> int:
    """
    Curăță cache-ul de interogări.
    
    Returns:
        Numărul de intrări șterse din cache
    """
    global _query_cache
    
    cleared_count = len(_query_cache)
    _query_cache.clear()
    logger.info(f"Cache-ul de interogări a fost curățat: {cleared_count} intrări șterse")
    return cleared_count

def get_cache_info() -> Dict[str, Any]:
    """
    Obține informații despre cache-ul de interogări.
    
    Returns:
        Dicționar cu informații despre cache
    """
    global _query_cache, MAX_QUERY_CACHE_SIZE
    
    return {
        "current_size": len(_query_cache),
        "max_size": MAX_QUERY_CACHE_SIZE,
        "keys": list(_query_cache.keys())[:10]  # Primele 10 chei pentru debug
    }

# Funcții specifice pentru JSON chunkizat - eliminate funcțiile pentru PDF
# Nu mai avem nevoie de process_pdf_optimized sau alte funcții pentru alte formate

def preview_json_chunks(file_path: str, max_chunks: int = 3) -> Dict[str, Any]:
    """
    Oferă o previzualizare a primelor chunk-uri dintr-un fișier JSON.
    
    Args:
        file_path: Calea către fișierul JSON
        max_chunks: Numărul maxim de chunk-uri de previzualizat
        
    Returns:
        Dicționar cu informații despre preview
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        chunk_keys = sorted([k for k in json_data.keys() if k.startswith("chunk_")], 
                           key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 999999)
        
        preview_data = {
            "total_chunks": len(chunk_keys),
            "previewed_chunks": min(max_chunks, len(chunk_keys)),
            "chunks": []
        }
        
        for i, key in enumerate(chunk_keys[:max_chunks]):
            value = json_data[key]
            
            if isinstance(value, dict) and "metadata" in value and "chunk" in value:
                chunk_content = value["chunk"].strip()
                preview_data["chunks"].append({
                    "chunk_id": key,
                    "metadata": value["metadata"],
                    "content_preview": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                    "content_length": len(chunk_content)
                })
        
        return preview_data
        
    except Exception as e:
        logger.error(f"Eroare la previzualizarea fișierului JSON: {str(e)}")
        return {
            "error": str(e),
            "total_chunks": 0,
            "previewed_chunks": 0,
            "chunks": []
        }