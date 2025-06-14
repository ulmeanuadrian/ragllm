import time
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

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
        
    Raises:
        ValueError: Dacă fișierul nu are formatul corect
        FileNotFoundError: Dacă fișierul nu există
    """
    start_time = time.time()
    chunks_data = []
    
    try:
        # Verificăm dacă fișierul există
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Fișierul nu a fost găsit: {file_path}")
        
        # Verificăm dimensiunea fișierului
        file_size = Path(file_path).stat().st_size
        max_size = 50 * 1024 * 1024  # 50MB limit
        
        if file_size > max_size:
            raise ValueError(f"Fișierul este prea mare: {file_size / (1024*1024):.1f}MB > {max_size / (1024*1024)}MB")
        
        # Încărcăm JSON-ul cu gestionare optimizată a memoriei
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Fișierul JSON nu este valid: {str(e)}")
        
        logger.info(f"JSON încărcat cu succes ({file_size / 1024:.1f}KB), verificare format...", 
                   extra={"file": file_path, "size_kb": file_size / 1024})
        
        # Verificăm că JSON-ul este un dicționar
        if not isinstance(json_data, dict):
            raise ValueError("JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip")
        
        # Găsim toate chunk-urile și le sortăm
        chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
        
        if not chunk_keys:
            raise ValueError("Nu s-au găsit chunk-uri. Cheile trebuie să înceapă cu 'chunk_'")
        
        # Sortăm cheile pentru a procesa chunk-urile în ordine
        def extract_chunk_number(key: str) -> int:
            try:
                return int(key.split("_")[1])
            except (IndexError, ValueError):
                return 999999  # Punem la sfârșit chunk-urile cu nume invalid
        
        sorted_keys = sorted(chunk_keys, key=extract_chunk_number)
        
        # Procesăm chunk-urile
        chunk_count = 0
        valid_chunks = 0
        
        for key in sorted_keys:
            value = json_data[key]
            chunk_count += 1
            
            # Validăm structura chunk-ului
            if not isinstance(value, dict):
                logger.warning(f"Chunk {key} nu este un dicționar și va fi ignorat", 
                             extra={"chunk_type": type(value).__name__})
                continue
            
            if "metadata" not in value:
                logger.warning(f"Chunk {key} nu conține câmpul 'metadata' și va fi ignorat")
                continue
            
            if "chunk" not in value:
                logger.warning(f"Chunk {key} nu conține câmpul 'chunk' și va fi ignorat")
                continue
            
            # Validăm conținutul chunk-ului
            chunk_content = str(value["chunk"]).strip()
            
            if len(chunk_content) < 10:
                logger.warning(f"Chunk {key} este prea scurt ({len(chunk_content)} caractere) și va fi ignorat")
                continue
            
            # Procesăm metadatele
            metadata = {
                "chunk_id": key,
                "original_source": str(value["metadata"]),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_index": valid_chunks,
                "file_type": "json_chunked",
                "content_length": len(chunk_content),
                "file_path": str(file_path)
            }
            
            # Adăugăm chunk-ul în lista de procesare
            chunks_data.append({
                "content": chunk_content,
                "metadata": metadata,
                "chunk_id": key
            })
            
            valid_chunks += 1
            
            logger.debug(f"Procesat chunk {key}: {len(chunk_content)} caractere", 
                       extra={"chunk_id": key, "content_length": len(chunk_content)})
        
        # Verificăm că avem chunk-uri valide
        if len(chunks_data) == 0:
            raise ValueError(
                f"Fișierul JSON nu conține chunk-uri valide în formatul așteptat. "
                f"Găsite {chunk_count} chunk-uri potențiale, dar niciun chunk valid. "
                f"Formatul așteptat: {{'chunk_0': {{'metadata': '...', 'chunk': '...'}}, ...}}"
            )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Fișier JSON procesat cu succes: {len(chunks_data)} chunk-uri valide din {chunk_count} chunk-uri totale în {processing_time:.2f} secunde",
            extra={
                "file": file_path, 
                "valid_chunks": len(chunks_data), 
                "total_chunks": chunk_count,
                "duration": f"{processing_time:.2f}s",
                "avg_chunk_size": sum(len(c["content"]) for c in chunks_data) / len(chunks_data)
            }
        )
        
        return chunks_data
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Eroare la procesarea fișierului JSON: {str(e)}", extra={"file": file_path})
        raise
    except Exception as e:
        logger.error(f"Eroare neașteptată la procesarea fișierului JSON: {str(e)}", 
                    extra={"file": file_path, "error_type": type(e).__name__})
        raise ValueError(f"Eroare neașteptată la procesarea fișierului: {str(e)}")-ul este un dicționar
        if not isinstance(json_data, dict):
            raise ValueError("JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip")
        
        # Găsim toate chunk-urile și le sortăm
        chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
        
        if not chunk_keys:
            raise ValueError("Nu s-au găsit chunk-uri. Cheile trebuie să înceapă cu 'chunk_'")
        
        # Sortăm cheile pentru a procesa chunk-urile în ordine
        def extract_chunk_number(key: str) -> int:
            try:
                return int(key.split("_")[1])
            except (IndexError, ValueError):
                return 999999  # Punem la sfârșit chunk-urile cu nume invalid
        
        sorted_keys = sorted(chunk_keys, key=extract_chunk_number)
        
        # Procesăm chunk-urile
        chunk_count = 0
        valid_chunks = 0
        
        for key in sorted_keys:
            value = json_data[key]
            chunk_count += 1
            
            # Validăm structura chunk-ului
            if not isinstance(value, dict):
                logger.warning(f"Chunk {key} nu este un dicționar și va fi ignorat", 
                             extra={"chunk_type": type(value).__name__})
                continue
            
            if "metadata" not in value:
                logger.warning(f"Chunk {key} nu conține câmpul 'metadata' și va fi ignorat")
                continue
            
            if "chunk" not in value:
                logger.warning(f"Chunk {key} nu conține câmpul 'chunk' și va fi ignorat")
                continue
            
            # Validăm conținutul chunk-ului
            chunk_content = str(value["chunk"]).strip()
            
            if len(chunk_content) < 10:
                logger.warning(f"Chunk {key} este prea scurt ({len(chunk_content)} caractere) și va fi ignorat")
                continue
            
            # Procesăm metadatele
            metadata = {
                "chunk_id": key,
                "original_source": str(value["metadata"]),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_index": valid_chunks,
                "file_type": "json_chunked",
                "content_length": len(chunk_content),
                "file_path": str(file_path)
            }
            
            # Adăugăm chunk-ul în lista de procesare
            chunks_data.append({
                "content": chunk_content,
                "metadata": metadata,
                "chunk_id": key
            })
            
            valid_chunks += 1
            
            logger.debug(f"Procesat chunk {key}: {len(chunk_content)} caractere", 
                       extra={"chunk_id": key, "content_length": len(chunk_content)})
        
        # Verificăm că avem chunk-uri valide
        if len(chunks_data) == 0:
            raise ValueError(
                f"Fișierul JSON nu conține chunk-uri valide în formatul așteptat. "
                f"Găsite {chunk_count} chunk-uri potențiale, dar niciun chunk valid. "
                f"Formatul așteptat: {{'chunk_0': {{'metadata': '...', 'chunk': '...'}}, ...}}"
            )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Fișier JSON procesat cu succes: {len(chunks_data)} chunk-uri valide din {chunk_count} chunk-uri totale în {processing_time:.2f} secunde",
            extra={
                "file": file_path, 
                "valid_chunks": len(chunks_data), 
                "total_chunks": chunk_count,
                "duration": f"{processing_time:.2f}s",
                "avg_chunk_size": sum(len(c["content"]) for c in chunks_data) / len(chunks_data)
            }
        )
        
        return chunks_data
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Eroare la procesarea fișierului JSON: {str(e)}", extra={"file": file_path})
        raise
    except Exception as e:
        logger.error(f"Eroare neașteptată la procesarea fișierului JSON: {str(e)}", 
                    extra={"file": file_path, "error_type": type(e).__name__})
        raise ValueError(f"Eroare neașteptată la procesarea fișierului: {str(e)}")


def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează dacă un fișier JSON are formatul corect pentru chunk-uri.
    
    Args:
        file_path: Calea către fișierul JSON
        
    Returns:
        Tuple (is_valid, error_message, chunks_count)
    """
    try:
        # Verificăm dacă fișierul există
        if not Path(file_path).exists():
            return False, "Fișierul nu a fost găsit", 0
        
        # Verificăm dimensiunea fișierului
        file_size = Path(file_path).stat().st_size
        if file_size == 0:
            return False, "Fișierul este gol", 0
        
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            return False, f"Fișierul este prea mare: {file_size / (1024*1024):.1f}MB", 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"Fișierul JSON nu este valid: {str(e)}", 0
        
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
        if not Path(file_path).exists():
            return {"error": "Fișierul nu a fost găsit", "total_chunks": 0, "valid_chunks": 0}
        
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        if not isinstance(json_data, dict):
            return {"error": "JSON-ul nu este un dicționar", "total_chunks": 0, "valid_chunks": 0}
        
        chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
        
        # Sortăm chunk-urile
        def extract_chunk_number(key: str) -> int:
            try:
                return int(key.split("_")[1])
            except (IndexError, ValueError):
                return 999999
        
        sorted_keys = sorted(chunk_keys, key=extract_chunk_number)
        
        statistics = {
            "total_chunks": len(chunk_keys),
            "valid_chunks": 0,
            "total_characters": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": float('inf'),
            "max_chunk_length": 0,
            "sources": set(),
            "chunk_ids": [],
            "file_size_bytes": Path(file_path).stat().st_size,
            "file_size_kb": Path(file_path).stat().st_size / 1024
        }
        
        for key in sorted_keys:
            value = json_data[key]
            
            if isinstance(value, dict) and "metadata" in value and "chunk" in value:
                chunk_content = str(value["chunk"]).strip()
                chunk_length = len(chunk_content)
                
                if chunk_length >= 10:  # Considerăm doar chunk-urile valide
                    statistics["valid_chunks"] += 1
                    statistics["total_characters"] += chunk_length
                    statistics["min_chunk_length"] = min(statistics["min_chunk_length"], chunk_length)
                    statistics["max_chunk_length"] = max(statistics["max_chunk_length"], chunk_length)
                    statistics["sources"].add(str(value["metadata"]))
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
            "valid_chunks": 0,
            "file_size_bytes": 0,
            "file_size_kb": 0
        }


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
        if not Path(file_path).exists():
            return {"error": "Fișierul nu a fost găsit", "total_chunks": 0, "previewed_chunks": 0, "chunks": []}
        
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        if not isinstance(json_data, dict):
            return {"error": "JSON-ul nu este un dicționar", "total_chunks": 0, "previewed_chunks": 0, "chunks": []}
        
        chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
        
        # Sortăm chunk-urile
        def extract_chunk_number(key: str) -> int:
            try:
                return int(key.split("_")[1])
            except (IndexError, ValueError):
                return 999999
        
        sorted_keys = sorted(chunk_keys, key=extract_chunk_number)
        
        preview_data = {
            "total_chunks": len(chunk_keys),
            "previewed_chunks": min(max_chunks, len(chunk_keys)),
            "chunks": [],
            "file_info": {
                "size_kb": Path(file_path).stat().st_size / 1024,
                "name": Path(file_path).name
            }
        }
        
        for i, key in enumerate(sorted_keys[:max_chunks]):
            value = json_data[key]
            
            if isinstance(value, dict) and "metadata" in value and "chunk" in value:
                chunk_content = str(value["chunk"]).strip()
                preview_length = 200
                
                preview_data["chunks"].append({
                    "chunk_id": key,
                    "metadata": str(value["metadata"]),
                    "content_preview": (chunk_content[:preview_length] + "..." 
                                      if len(chunk_content) > preview_length 
                                      else chunk_content),
                    "content_length": len(chunk_content),
                    "is_valid": len(chunk_content) >= 10
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


def generate_file_hash(file_path: str) -> str:
    """
    Generează un hash pentru un fișier pentru detectarea duplicatelor.
    
    Args:
        file_path: Calea către fișier
        
    Returns:
        Hash-ul MD5 al fișierului
    """
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Eroare la generarea hash-ului pentru {file_path}: {str(e)}")
        return ""


def optimize_json_structure(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimizează structura unui JSON pentru procesare mai eficientă.
    
    Args:
        json_data: Datele JSON de optimizat
        
    Returns:
        JSON optimizat
    """
    if not isinstance(json_data, dict):
        return json_data
    
    optimized = {}
    chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
    
    # Sortăm și re-indexăm chunk-urile
    def extract_chunk_number(key: str) -> int:
        try:
            return int(key.split("_")[1])
        except (IndexError, ValueError):
            return 999999
    
    sorted_keys = sorted(chunk_keys, key=extract_chunk_number)
    
    for i, key in enumerate(sorted_keys):
        value = json_data[key]
        
        if isinstance(value, dict) and "metadata" in value and "chunk" in value:
            # Optimizăm conținutul chunk-ului
            chunk_content = str(value["chunk"]).strip()
            
            if len(chunk_content) >= 10:  # Doar chunk-urile valide
                optimized[f"chunk_{i}"] = {
                    "metadata": str(value["metadata"]).strip(),
                    "chunk": chunk_content
                }
    
    return optimized


def batch_process_json_files(file_paths: List[str]) -> List[Tuple[str, List[Dict[str, Any]], Optional[str]]]:
    """
    Procesează mai multe fișiere JSON în batch pentru eficiență.
    
    Args:
        file_paths: Lista căilor către fișierele JSON
        
    Returns:
        Lista de tuple (file_path, chunks_data, error_message)
    """
    results = []
    
    for file_path in file_paths:
        try:
            chunks_data = process_json_chunks(file_path)
            results.append((file_path, chunks_data, None))
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Eroare la procesarea {file_path}: {error_msg}")
            results.append((file_path, [], error_msg))
    
    return results


def detect_json_encoding(file_path: str) -> str:
    """
    Detectează encoding-ul unui fișier JSON.
    
    Args:
        file_path: Calea către fișierul JSON
        
    Returns:
        Encoding-ul detectat
    """
    import chardet
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(min(32768, Path(file_path).stat().st_size))  # Citim maxim 32KB
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8') or 'utf-8'
    except Exception as e:
        logger.warning(f"Nu s-a putut detecta encoding-ul pentru {file_path}: {str(e)}")
        return 'utf-8'


def compress_json_content(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprimă conținutul JSON prin eliminarea spațiilor și optimizări.
    
    Args:
        json_data: Datele JSON de comprimat
        
    Returns:
        JSON comprimat
    """
    if not isinstance(json_data, dict):
        return json_data
    
    compressed = {}
    
    for key, value in json_data.items():
        if key.startswith("chunk_") and isinstance(value, dict):
            if "metadata" in value and "chunk" in value:
                # Comprimăm conținutul
                chunk_content = str(value["chunk"]).strip()
                # Eliminăm spațiile multiple
                chunk_content = ' '.join(chunk_content.split())
                
                compressed[key] = {
                    "metadata": str(value["metadata"]).strip(),
                    "chunk": chunk_content
                }
    
    return compressed


def create_json_backup(file_path: str, backup_dir: Optional[str] = None) -> str:
    """
    Creează o copie de backup pentru un fișier JSON.
    
    Args:
        file_path: Calea către fișierul original
        backup_dir: Directorul pentru backup (opțional)
        
    Returns:
        Calea către fișierul de backup
    """
    try:
        original_path = Path(file_path)
        
        if backup_dir:
            backup_directory = Path(backup_dir)
        else:
            backup_directory = original_path.parent / "backups"
        
        # Creăm directorul de backup dacă nu există
        backup_directory.mkdir(exist_ok=True)
        
        # Generăm numele fișierului de backup cu timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{original_path.stem}_backup_{timestamp}{original_path.suffix}"
        backup_path = backup_directory / backup_name
        
        # Copiem fișierul
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"Backup creat: {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        logger.error(f"Eroare la crearea backup-ului pentru {file_path}: {str(e)}")
        raise


def validate_chunk_content_quality(content: str) -> Dict[str, Any]:
    """
    Validează calitatea conținutului unui chunk.
    
    Args:
        content: Conținutul chunk-ului
        
    Returns:
        Dicționar cu rezultatele validării
    """
    result = {
        "is_valid": True,
        "quality_score": 0.0,
        "issues": [],
        "suggestions": []
    }
    
    content = content.strip()
    
    # Verificări de bază
    if len(content) < 10:
        result["is_valid"] = False
        result["issues"].append("Conținut prea scurt")
        return result
    
    # Calculăm scorul de calitate
    quality_factors = []
    
    # Factor 1: Lungimea conținutului
    if 50 <= len(content) <= 2000:
        quality_factors.append(1.0)
    elif 10 <= len(content) < 50:
        quality_factors.append(0.6)
    elif len(content) > 2000:
        quality_factors.append(0.8)
    else:
        quality_factors.append(0.2)
    
    # Factor 2: Diversitatea caracterelor
    unique_chars = len(set(content.lower()))
    char_diversity = min(unique_chars / 50, 1.0)  # Normalizăm la maxim 50 caractere unice
    quality_factors.append(char_diversity)
    
    # Factor 3: Prezența propozițiilor complete
    sentences = content.count('.') + content.count('!') + content.count('?')
    if sentences > 0:
        quality_factors.append(1.0)
    else:
        quality_factors.append(0.5)
        result["suggestions"].append("Conținutul ar trebui să conțină propoziții complete")
    
    # Factor 4: Raportul spații/text
    spaces = content.count(' ')
    words = len(content.split())
    if words > 1:
        space_ratio = spaces / len(content)
        if 0.1 <= space_ratio <= 0.2:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.7)
    else:
        quality_factors.append(0.3)
        result["suggestions"].append("Conținutul ar trebui să conțină mai multe cuvinte")
    
    # Calculăm scorul final
    result["quality_score"] = sum(quality_factors) / len(quality_factors)
    
    # Stabilim pragul pentru validitate
    if result["quality_score"] < 0.5:
        result["is_valid"] = False
        result["issues"].append(f"Scor de calitate scăzut: {result['quality_score']:.2f}")
    
    return result


def merge_json_chunks_files(file_paths: List[str], output_path: str) -> Dict[str, Any]:
    """
    Combină mai multe fișiere JSON cu chunk-uri într-un singur fișier.
    
    Args:
        file_paths: Lista căilor către fișierele JSON
        output_path: Calea către fișierul de output
        
    Returns:
        Statistici despre procesul de combinare
    """
    merged_data = {}
    chunk_index = 0
    stats = {
        "total_files": len(file_paths),
        "processed_files": 0,
        "total_chunks": 0,
        "valid_chunks": 0,
        "errors": []
    }
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if not isinstance(json_data, dict):
                stats["errors"].append(f"{file_path}: Nu este un dicționar JSON valid")
                continue
            
            chunk_keys = sorted([k for k in json_data.keys() if k.startswith("chunk_")],
                              key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 999999)
            
            for key in chunk_keys:
                value = json_data[key]
                if isinstance(value, dict) and "metadata" in value and "chunk" in value:
                    # Adăugăm informații despre fișierul sursă în metadata
                    enhanced_metadata = str(value["metadata"])
                    if not enhanced_metadata.endswith(f" (din {Path(file_path).name})"):
                        enhanced_metadata += f" (din {Path(file_path).name})"
                    
                    merged_data[f"chunk_{chunk_index}"] = {
                        "metadata": enhanced_metadata,
                        "chunk": value["chunk"]
                    }
                    
                    chunk_index += 1
                    stats["valid_chunks"] += 1
                
                stats["total_chunks"] += 1
            
            stats["processed_files"] += 1
            
        except Exception as e:
            error_msg = f"{file_path}: {str(e)}"
            stats["errors"].append(error_msg)
            logger.error(f"Eroare la procesarea {file_path} pentru combinare: {str(e)}")
    
    # Salvăm fișierul combinat
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        stats["output_file"] = output_path
        stats["success"] = True
        
        logger.info(f"Fișiere JSON combinate cu succes în {output_path}")
        
    except Exception as e:
        stats["success"] = False
        stats["errors"].append(f"Eroare la salvarea fișierului combinat: {str(e)}")
        logger.error(f"Eroare la salvarea fișierului combinat {output_path}: {str(e)}")
    
    return stats-ul este un dicționar
        if not isinstance(json_data, dict):
            raise ValueError("JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip")
        
        # Găsim toate chunk-urile și le sortăm
        chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
        
        if not chunk_keys:
            raise ValueError("Nu s-au găsit chunk-uri. Cheile trebuie să înceapă cu 'chunk_'")
        
        # Sortăm cheile pentru a procesa chunk-urile în ordine
        def extract_chunk_number(key: str) -> int:
            try:
                return int(key.split("_")[1])
            except (IndexError, ValueError):
                return 999999  # Punem la sfârșit chunk-urile cu nume invalid
        
        sorted_keys = sorted(chunk_keys, key=extract_chunk_number)
        
        # Procesăm chunk-urile
        chunk_count = 0
        valid_chunks = 0
        
        for key in sorted_keys:
            value = json_data[key]
            chunk_count += 1
            
            # Validăm structura chunk-ului
            if not isinstance(value, dict):
                logger.warning(f"Chunk {key} nu este un dicționar și va fi ignorat", 
                             extra={"chunk_type": type(value).__name__})
                continue
            
            if "metadata" not in value:
                logger.warning(f"Chunk {key} nu conține câmpul 'metadata' și va fi ignorat")
                continue
            
            if "chunk" not in value:
                logger.warning(f"Chunk {key} nu conține câmpul 'chunk' și va fi ignorat")
                continue
            
            # Validăm conținutul chunk-ului
            chunk_content = str(value["chunk"]).strip()
            
            if len(chunk_content) < 10:
                logger.warning(f"Chunk {key} este prea scurt ({len(chunk_content)} caractere) și va fi ignorat")
                continue
            
            # Procesăm metadatele
            metadata = {
                "chunk_id": key,
                "original_source": str(value["metadata"]),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_index": valid_chunks,
                "file_type": "json_chunked",
                "content_length": len(chunk_content),
                "file_path": str(file_path)
            }
            
            # Adăugăm chunk-ul în lista de procesare
            chunks_data.append({
                "content": chunk_content,
                "metadata": metadata,
                "chunk_id": key
            })
            
            valid_chunks += 1
            
            logger.debug(f"Procesat chunk {key}: {len(chunk_content)} caractere", 
                       extra={"chunk_id": key, "content_length": len(chunk_content)})
        
        # Verificăm că avem chunk-uri valide
        if len(chunks_data) == 0:
            raise ValueError(
                f"Fișierul JSON nu conține chunk-uri valide în formatul așteptat. "
                f"Găsite {chunk_count} chunk-uri potențiale, dar niciun chunk valid. "
                f"Formatul așteptat: {{'chunk_0': {{'metadata': '...', 'chunk': '...'}}, ...}}"
            )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Fișier JSON procesat cu succes: {len(chunks_data)} chunk-uri valide din {chunk_count} chunk-uri totale în {processing_time:.2f} secunde",
            extra={
                "file": file_path, 
                "valid_chunks": len(chunks_data), 
                "total_chunks": chunk_count,
                "duration": f"{processing_time:.2f}s",
                "avg_chunk_size": sum(len(c["content"]) for c in chunks_data) / len(chunks_data)
            }
        )
        
        return chunks_data
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Eroare la procesarea fișierului JSON: {str(e)}", extra={"file": file_path})
        raise
    except Exception as e:
        logger.error(f"Eroare neașteptată la procesarea fișierului JSON: {str(e)}", 
                    extra={"file": file_path, "error_type": type(e).__name__})
        raise ValueError(f"Eroare neașteptată la procesarea fișierului: {str(e)}")


def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează dacă un fișier JSON are formatul corect pentru chunk-uri.
    
    Args:
        file_path: Calea către fișierul JSON
        
    Returns:
        Tuple (is_valid, error_message, chunks_count)
    """
    try:
        # Verificăm dacă fișierul există
        if not Path(file_path).exists():
            return False, "Fișierul nu a fost găsit", 0
        
        # Verificăm dimensiunea fișierului
        file_size = Path(file_path).stat().st_size
        if file_size == 0:
            return False, "Fișierul este gol", 0
        
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            return False, f"Fișierul este prea mare: {file_size / (1024*1024):.1f}MB", 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"Fișierul JSON nu este valid: {str(e)}", 0
        
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
        if not Path(file_path).exists():
            return {"error": "Fișierul nu a fost găsit", "total_chunks": 0, "valid_chunks": 0}
        
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        if not isinstance(json_data, dict):
            return {"error": "JSON-ul nu este un dicționar", "total_chunks": 0, "valid_chunks": 0}
        
        chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
        
        # Sortăm chunk-urile
        def extract_chunk_number(key: str) -> int:
            try:
                return int(key.split("_")[1])
            except (IndexError, ValueError):
                return 999999
        
        sorted_keys = sorted(chunk_keys, key=extract_chunk_number)
        
        statistics = {
            "total_chunks": len(chunk_keys),
            "valid_chunks": 0,
            "total_characters": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": float('inf'),
            "max_chunk_length": 0,
            "sources": set(),
            "chunk_ids": [],
            "file_size_bytes": Path(file_path).stat().st_size,
            "file_size_kb": Path(file_path).stat().st_size / 1024
        }
        
        for key in sorted_keys:
            value = json_data[key]
            
            if isinstance(value, dict) and "metadata" in value and "chunk" in value:
                chunk_content = str(value["chunk"]).strip()
                chunk_length = len(chunk_content)
                
                if chunk_length >= 10:  # Considerăm doar chunk-urile valide
                    statistics["valid_chunks"] += 1
                    statistics["total_characters"] += chunk_length
                    statistics["min_chunk_length"] = min(statistics["min_chunk_length"], chunk_length)
                    statistics["max_chunk_length"] = max(statistics["max_chunk_length"], chunk_length)
                    statistics["sources"].add(str(value["metadata"]))
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
            "valid_chunks": 0,
            "file_size_bytes": 0,
            "file_size_kb": 0
        }


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
        if not Path(file_path).exists():
            return {"error": "Fișierul nu a fost găsit", "total_chunks": 0, "previewed_chunks": 0, "chunks": []}
        
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        if not isinstance(json_data, dict):
            return {"error": "JSON-ul nu este un dicționar", "total_chunks": 0, "previewed_chunks": 0, "chunks": []}
        
        chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
        
        # Sortăm chunk-urile
        def extract_chunk_number(key: str) -> int:
            try:
                return int(key.split("_")[1])
            except (IndexError, ValueError):
                return 999999
        
        sorted_keys = sorted(chunk_keys, key=extract_chunk_number)
        
        preview_data = {
            "total_chunks": len(chunk_keys),
            "previewed_chunks": min(max_chunks, len(chunk_keys)),
            "chunks": [],
            "file_info": {
                "size_kb": Path(file_path).stat().st_size / 1024,
                "name": Path(file_path).name
            }
        }
        
        for i, key in enumerate(sorted_keys[:max_chunks]):
            value = json_data[key]
            
            if isinstance(value, dict) and "metadata" in value and "chunk" in value:
                chunk_content = str(value["chunk"]).strip()
                preview_length = 200
                
                preview_data["chunks"].append({
                    "chunk_id": key,
                    "metadata": str(value["metadata"]),
                    "content_preview": (chunk_content[:preview_length] + "..." 
                                      if len(chunk_content) > preview_length 
                                      else chunk_content),
                    "content_length": len(chunk_content),
                    "is_valid": len(chunk_content) >= 10
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


def generate_file_hash(file_path: str) -> str:
    """
    Generează un hash pentru un fișier pentru detectarea duplicatelor.
    
    Args:
        file_path: Calea către fișier
        
    Returns:
        Hash-ul MD5 al fișierului
    """
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Eroare la generarea hash-ului pentru {file_path}: {str(e)}")
        return ""


def optimize_json_structure(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimizează structura unui JSON pentru procesare mai eficientă.
    
    Args:
        json_data: Datele JSON de optimizat
        
    Returns:
        JSON optimizat
    """
    if not isinstance(json_data, dict):
        return json_data
    
    optimized = {}
    chunk_keys = [k for k in json_data.keys() if k.startswith("chunk_")]
    
    # Sortăm și re-indexăm chunk-urile
    def extract_chunk_number(key: str) -> int:
        try:
            return int(key.split("_")[1])
        except (IndexError, ValueError):
            return 999999
    
    sorted_keys = sorted(chunk_keys, key=extract_chunk_number)
    
    for i, key in enumerate(sorted_keys):
        value = json_data[key]
        
        if isinstance(value, dict) and "metadata" in value and "chunk" in value:
            # Optimizăm conținutul chunk-ului
            chunk_content = str(value["chunk"]).strip()
            
            if len(chunk_content) >= 10:  # Doar chunk-urile valide
                optimized[f"chunk_{i}"] = {
                    "metadata": str(value["metadata"]).strip(),
                    "chunk": chunk_content
                }
    
    return optimized


def batch_process_json_files(file_paths: List[str]) -> List[Tuple[str, List[Dict[str, Any]], Optional[str]]]:
    """
    Procesează mai multe fișiere JSON în batch pentru eficiență.
    
    Args:
        file_paths: Lista căilor către fișierele JSON
        
    Returns:
        Lista de tuple (file_path, chunks_data, error_message)
    """
    results = []
    
    for file_path in file_paths:
        try:
            chunks_data = process_json_chunks(file_path)
            results.append((file_path, chunks_data, None))
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Eroare la procesarea {file_path}: {error_msg}")
            results.append((file_path, [], error_msg))
    
    return results