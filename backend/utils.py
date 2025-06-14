"""
Utilitare pentru procesarea fișierelor JSON chunkizate
Optimizat pentru formatul EXACT specificat de utilizator
"""

import json
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează că fișierul JSON are formatul EXACT specificat:
    {
        "chunk_0": {
            "metadata": "string",
            "chunk": "content"
        }
    }
    
    Returns:
        (is_valid, error_message, chunks_count)
    """
    try:
        if not os.path.exists(file_path):
            return False, "Fișierul nu există", 0
        
        if os.path.getsize(file_path) == 0:
            return False, "Fișierul este gol", 0
        
        # Încercăm să încărcăm JSON-ul
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"JSON invalid: {str(e)}", 0
        
        # Verificăm că este un dicționar
        if not isinstance(data, dict):
            return False, "JSON-ul trebuie să fie un obiect (dicționar), nu o listă", 0
        
        # Căutăm chunk-uri în formatul chunk_0, chunk_1, etc.
        chunk_keys = [key for key in data.keys() if key.startswith('chunk_')]
        
        if len(chunk_keys) == 0:
            return False, "Nu s-au găsit chunk-uri în formatul așteptat (chunk_0, chunk_1, etc.)", 0
        
        # Validăm primele 3 chunk-uri pentru structura EXACTĂ
        valid_chunks = 0
        for key in chunk_keys[:3]:
            chunk = data[key]
            
            # Verificăm că chunk-ul este un dicționar
            if not isinstance(chunk, dict):
                continue
            
            # Verificăm că are EXACT cheile: "metadata" și "chunk"
            if "metadata" not in chunk or "chunk" not in chunk:
                continue
            
            # Verificăm că metadata este STRING (nu dict!)
            if not isinstance(chunk["metadata"], str):
                continue
            
            # Verificăm că chunk este STRING cu conținut
            if not isinstance(chunk["chunk"], str) or len(chunk["chunk"].strip()) < 10:
                continue
            
            valid_chunks += 1
        
        if valid_chunks == 0:
            return False, "Nu s-au găsit chunk-uri valide cu structura EXACTĂ (metadata: string, chunk: string)", 0
        
        logger.info(f"✅ JSON valid găsit cu {len(chunk_keys)} chunk-uri, {valid_chunks} validate")
        return True, f"Valid - {len(chunk_keys)} chunk-uri găsite", len(chunk_keys)
        
    except Exception as e:
        logger.error(f"Eroare la validarea JSON: {str(e)}")
        return False, f"Eroare la validare: {str(e)}", 0

def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează fișierul JSON cu structura EXACTĂ și extrage chunk-urile pentru stocare
    
    Returns:
        Lista de dicționare cu chunk-uri procesate
    """
    try:
        # Validăm mai întâi formatul
        is_valid, error_msg, chunks_count = validate_json_format(file_path)
        if not is_valid:
            raise ValueError(f"Fișier JSON invalid: {error_msg}")
        
        # Încărcăm datele
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extragem chunk-urile
        chunks_data = []
        chunk_keys = sorted([key for key in data.keys() if key.startswith('chunk_')])
        
        for i, key in enumerate(chunk_keys):
            chunk = data[key]
            
            # Verificăm structura EXACTĂ
            if not isinstance(chunk, dict) or "metadata" not in chunk or "chunk" not in chunk:
                logger.warning(f"Chunk invalid sărit: {key}")
                continue
            
            content = chunk["chunk"]
            metadata_string = chunk["metadata"]
            
            # Verificăm că sunt string-uri valide
            if not isinstance(content, str) or not isinstance(metadata_string, str):
                logger.warning(f"Chunk cu tipuri invalide sărit: {key}")
                continue
            
            # Verificăm că avem conținut suficient
            if len(content.strip()) < 10:
                logger.warning(f"Chunk cu conținut insuficient sărit: {key}")
                continue
            
            # Curățăm conținutul
            content = content.strip()
            
            # Convertim metadata string în dict pentru ChromaDB
            enhanced_metadata = {
                'chunk_id': key,
                'chunk_index': i,
                'content_length': len(content),
                'word_count': len(content.split()),
                'processed_at': datetime.now().isoformat(),
                'file_source': os.path.basename(file_path),
                'original_source': metadata_string,  # Păstrăm metadata originală
                'source': metadata_string  # Pentru compatibilitate
            }
            
            chunks_data.append({
                'content': content,
                'metadata': enhanced_metadata
            })
        
        if not chunks_data:
            raise ValueError("Nu s-au putut extrage chunk-uri valide din fișier")
        
        logger.info(f"✅ Procesat cu succes: {len(chunks_data)} chunk-uri din {file_path}")
        return chunks_data
        
    except Exception as e:
        logger.error(f"Eroare la procesarea fișierului {file_path}: {str(e)}")
        raise ValueError(f"Eroare neașteptată la procesarea fișierului: {str(e)}")

def get_json_statistics(file_path: str) -> Dict[str, Any]:
    """
    Obține statistici despre fișierul JSON chunkizat cu structura EXACTĂ
    
    Returns:
        Dicționar cu statistici despre fișier
    """
    try:
        if not os.path.exists(file_path):
            return {"error": "Fișierul nu există"}
        
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            return {"error": "JSON-ul nu este un dicționar"}
        
        # Analiza chunk-urilor cu structura EXACTĂ
        chunk_keys = [key for key in data.keys() if key.startswith('chunk_')]
        
        if not chunk_keys:
            return {"error": "Nu s-au găsit chunk-uri"}
        
        # Statistici detaliate
        total_content_length = 0
        total_words = 0
        valid_chunks = 0
        metadata_sources = set()
        
        for key in chunk_keys:
            chunk = data[key]
            if (isinstance(chunk, dict) and 
                "metadata" in chunk and 
                "chunk" in chunk and 
                isinstance(chunk["metadata"], str) and 
                isinstance(chunk["chunk"], str)):
                
                content = chunk["chunk"]
                metadata_string = chunk["metadata"]
                
                if len(content.strip()) >= 10:
                    valid_chunks += 1
                    total_content_length += len(content)
                    total_words += len(content.split())
                    metadata_sources.add(metadata_string)
        
        stats = {
            'file_name': os.path.basename(file_path),
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'total_chunks': len(chunk_keys),
            'valid_chunks': valid_chunks,
            'invalid_chunks': len(chunk_keys) - valid_chunks,
            'total_content_length': total_content_length,
            'total_words': total_words,
            'average_chunk_length': round(total_content_length / valid_chunks) if valid_chunks > 0 else 0,
            'average_words_per_chunk': round(total_words / valid_chunks) if valid_chunks > 0 else 0,
            'unique_metadata_sources': len(metadata_sources),
            'metadata_sources_list': list(metadata_sources),
            'processed_at': datetime.now().isoformat()
        }
        
        logger.debug(f"📊 Statistici generate pentru {file_path}: {valid_chunks} chunk-uri valide")
        return stats
        
    except Exception as e:
        logger.error(f"Eroare la generarea statisticilor pentru {file_path}: {str(e)}")
        return {"error": f"Eroare la analiza fișierului: {str(e)}"}

def preview_json_chunks(file_path: str, max_chunks: int = 3) -> Dict[str, Any]:
    """
    Oferă o previzualizare a chunk-urilor din fișier cu structura EXACTĂ
    
    Args:
        file_path: Calea către fișierul JSON
        max_chunks: Numărul maxim de chunk-uri de previzualizat
        
    Returns:
        Dicționar cu previzualizarea chunk-urilor
    """
    try:
        if not os.path.exists(file_path):
            return {"error": "Fișierul nu există"}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            return {"error": "JSON-ul nu este un dicționar"}
        
        chunk_keys = sorted([key for key in data.keys() if key.startswith('chunk_')])
        
        if not chunk_keys:
            return {"error": "Nu s-au găsit chunk-uri"}
        
        # Selectăm primele chunk-uri pentru previzualizare
        preview_chunks = []
        
        for key in chunk_keys[:max_chunks]:
            chunk = data[key]
            
            if (isinstance(chunk, dict) and 
                "metadata" in chunk and 
                "chunk" in chunk and 
                isinstance(chunk["metadata"], str) and 
                isinstance(chunk["chunk"], str)):
                
                content = chunk["chunk"]
                metadata_string = chunk["metadata"]
                
                # Truncăm conținutul pentru previzualizare
                preview_content = content[:200] + "..." if len(content) > 200 else content
                
                preview_chunks.append({
                    'chunk_id': key,
                    'content_preview': preview_content,
                    'content_length': len(content),
                    'word_count': len(content.split()),
                    'metadata': metadata_string
                })
        
        result = {
            'file_name': os.path.basename(file_path),
            'total_chunks': len(chunk_keys),
            'preview_chunks': preview_chunks,
            'showing': f"{min(len(preview_chunks), max_chunks)} din {len(chunk_keys)} chunk-uri"
        }
        
        logger.debug(f"👁️ Previzualizare generată pentru {file_path}: {len(preview_chunks)} chunk-uri")
        return result
        
    except Exception as e:
        logger.error(f"Eroare la previzualizarea fișierului {file_path}: {str(e)}")
        return {"error": f"Eroare la previzualizare: {str(e)}"}

def validate_chunk_structure(chunk_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validează structura EXACTĂ a unui chunk individual
    
    Args:
        chunk_data: Datele chunk-ului de validat
        
    Returns:
        (is_valid, error_message)
    """
    try:
        if not isinstance(chunk_data, dict):
            return False, "Chunk-ul trebuie să fie un dicționar"
        
        # Verificăm cheile EXACTE
        required_keys = ["metadata", "chunk"]
        for key in required_keys:
            if key not in chunk_data:
                return False, f"Lipsește cheia obligatorie: {key}"
        
        # Verificăm că nu are chei în plus
        if set(chunk_data.keys()) != set(required_keys):
            return False, f"Chunk-ul trebuie să aibă EXACT cheile: {required_keys}"
        
        # Validăm metadata - TREBUIE să fie string
        metadata = chunk_data["metadata"]
        if not isinstance(metadata, str):
            return False, "Metadata trebuie să fie string, nu dict sau alt tip"
        
        if len(metadata.strip()) == 0:
            return False, "Metadata nu poate fi string gol"
        
        # Validăm conținutul - TREBUIE să fie string
        content = chunk_data["chunk"]
        if not isinstance(content, str):
            return False, "Conținutul chunk-ului trebuie să fie string"
        
        if len(content.strip()) < 10:
            return False, "Conținutul chunk-ului este prea scurt (minimum 10 caractere)"
        
        return True, "Chunk valid cu structura exactă"
        
    except Exception as e:
        return False, f"Eroare la validarea chunk-ului: {str(e)}"

def clean_chunk_content(content: str) -> str:
    """
    Curăță și normalizează conținutul unui chunk
    
    Args:
        content: Conținutul de curățat
        
    Returns:
        Conținutul curățat
    """
    if not isinstance(content, str):
        return ""
    
    # Eliminăm spațiile în plus
    content = content.strip()
    
    # Înlocuim multiple spații cu unul singur
    import re
    content = re.sub(r'\s+', ' ', content)
    
    # Eliminăm caracterele de control
    content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\t')
    
    return content

def extract_metadata_info(metadata_string: str) -> Dict[str, Any]:
    """
    Extrage informații din metadata string și le convertește în dict
    
    Args:
        metadata_string: Metadata originală ca string
        
    Returns:
        Metadata convertită în dict
    """
    if not isinstance(metadata_string, str):
        return {"original_source": "Necunoscut"}
    
    # Încercăm să parsăm informații din string
    metadata_dict = {
        "original_source": metadata_string,
        "source": metadata_string
    }
    
    # Dacă stringul conține "Source: filename", extragem filename-ul
    if "Source:" in metadata_string:
        try:
            source_part = metadata_string.split("Source:")[1].strip()
            metadata_dict["document_name"] = source_part
            metadata_dict["source"] = source_part
        except:
            pass
    
    return metadata_dict

# Funcții helper pentru debugging și testing
def test_json_file(file_path: str) -> None:
    """
    Testează complet un fișier JSON cu structura EXACTĂ și afișează rezultatele
    """
    print(f"\n🔍 Testare fișier: {file_path}")
    print("=" * 50)
    
    # Test validare
    is_valid, error_msg, chunks_count = validate_json_format(file_path)
    print(f"✅ Validare: {'VALID' if is_valid else 'INVALID'}")
    if not is_valid:
        print(f"❌ Eroare: {error_msg}")
        return
    
    # Test statistici
    stats = get_json_statistics(file_path)
    if 'error' not in stats:
        print(f"📊 Statistici:")
        print(f"   - Total chunk-uri: {stats['total_chunks']}")
        print(f"   - Chunk-uri valide: {stats['valid_chunks']}")
        print(f"   - Dimensiune fișier: {stats['file_size_mb']} MB")
        print(f"   - Total cuvinte: {stats['total_words']}")
        print(f"   - Surse metadata unice: {stats['unique_metadata_sources']}")
    
    # Test previzualizare
    preview = preview_json_chunks(file_path, 2)
    if 'error' not in preview:
        print(f"👁️ Previzualizare (primele 2 chunk-uri):")
        for chunk in preview['preview_chunks']:
            print(f"   - {chunk['chunk_id']}: {chunk['content_length']} caractere")
            print(f"     Metadata: {chunk['metadata']}")
    
    # Test procesare
    try:
        chunks_data = process_json_chunks(file_path)
        print(f"✅ Procesare: {len(chunks_data)} chunk-uri procesate cu succes")
    except Exception as e:
        print(f"❌ Eroare la procesare: {str(e)}")
    
    print("=" * 50)