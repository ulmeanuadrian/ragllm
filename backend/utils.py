def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează fișierul JSON cu îmbunătățiri pentru căutare și indexare optimizată.
    
    Returns:
        Lista de dicționare cu chunk-uri procesate și îmbunătățite
    """
    try:
        # Validăm mai întâi formatul
        is_valid, error_msg, chunks_count = validate_json_format(file_path)
        if not is_valid:
            raise ValueError(f"Fișier JSON invalid: {error_msg}")
        
        # Încărcăm datele
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extragem chunk-urile cu procesare îmbunătățită
        chunks_data = []
        chunk_pattern = re.compile(r'^chunk_(\d+)"""
Utilitare OPTIMIZATE pentru procesarea fișierelor JSON chunkizate
Versiunea 3.0.0 - Îmbunătățiri pentru căutare și procesare
"""

import json
import os
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)

def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează că fișierul JSON are formatul EXACT specificat cu verificări îmbunătățite.
    
    Formatul acceptat:
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
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Fișierul este gol", 0
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return False, f"Fișierul este prea mare ({file_size // 1024 // 1024}MB). Maximum 100MB.", 0
        
        # Încercăm să încărcăm JSON-ul
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"JSON invalid: {str(e)[:100]}...", 0
        
        # Verificăm că este un dicționar
        if not isinstance(data, dict):
            return False, "JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip", 0
        
        if len(data) == 0:
            return False, "JSON-ul nu conține date", 0
        
        # Căutăm chunk-uri în formatul chunk_X
        chunk_pattern = re.compile(r'^chunk_\d+$')
        chunk_keys = [key for key in data.keys() if chunk_pattern.match(key)]
        
        if len(chunk_keys) == 0:
            # Încercăm să găsim alte patterns comune
            other_keys = list(data.keys())[:10]  # Primele 10 chei pentru debugging
            return False, f"Nu s-au găsit chunk-uri în formatul așteptat (chunk_0, chunk_1, etc.). Chei găsite: {other_keys}", 0
        
        # Validăm structura chunk-urilor - verificăm mai multe pentru siguranță
        valid_chunks = 0
        invalid_chunks = []
        sample_content_lengths = []
        
        for key in chunk_keys[:min(10, len(chunk_keys))]:  # Verificăm până la 10 chunk-uri
            chunk = data[key]
            
            # Verificăm că chunk-ul este un dicționar
            if not isinstance(chunk, dict):
                invalid_chunks.append(f"{key}: nu este dicționar")
                continue
            
            # Verificăm că are EXACT cheile: "metadata" și "chunk"
            required_keys = {"metadata", "chunk"}
            chunk_keys_set = set(chunk.keys())
            
            if not required_keys.issubset(chunk_keys_set):
                missing_keys = required_keys - chunk_keys_set
                invalid_chunks.append(f"{key}: lipsesc cheile {missing_keys}")
                continue
            
            # Verificăm că metadata este STRING
            if not isinstance(chunk["metadata"], str):
                invalid_chunks.append(f"{key}: metadata nu este string (este {type(chunk['metadata']).__name__})")
                continue
            
            if len(chunk["metadata"].strip()) == 0:
                invalid_chunks.append(f"{key}: metadata este string gol")
                continue
            
            # Verificăm că chunk este STRING cu conținut suficient
            if not isinstance(chunk["chunk"], str):
                invalid_chunks.append(f"{key}: chunk nu este string (este {type(chunk['chunk']).__name__})")
                continue
            
            content = chunk["chunk"].strip()
            if len(content) < 10:
                invalid_chunks.append(f"{key}: conținut prea scurt ({len(content)} caractere)")
                continue
            
            sample_content_lengths.append(len(content))
            valid_chunks += 1
        
        if valid_chunks == 0:
            error_details = "; ".join(invalid_chunks[:5])  # Primele 5 erori
            return False, f"Nu s-au găsit chunk-uri valide. Erori: {error_details}", 0
        
        # Verificăm consistența numerotării chunk-urilor
        chunk_numbers = []
        for key in chunk_keys:
            match = re.match(r'chunk_(\d+)', key)
            if match:
                chunk_numbers.append(int(match.group(1)))
        
        chunk_numbers.sort()
        expected_sequence = list(range(len(chunk_numbers)))
        
        # Warning pentru numerotare inconsistentă (nu blochez validarea)
        if chunk_numbers != expected_sequence:
            logger.warning(f"Numerotarea chunk-urilor nu este consecutivă: găsite {chunk_numbers[:10]}..., așteptate {expected_sequence[:10]}...")
        
        # Statistici pentru logging
        avg_length = sum(sample_content_lengths) / len(sample_content_lengths) if sample_content_lengths else 0
        
        logger.info(f"✅ JSON valid găsit cu {len(chunk_keys)} chunk-uri")
        logger.info(f"📊 Validare: {valid_chunks}/{min(10, len(chunk_keys))} chunk-uri verificate cu succes")
        logger.info(f"📏 Lungime medie conținut: {avg_length:.0f} caractere")
        
        return True, f"Valid - {len(chunk_keys)} chunk-uri găsite", len(chunk_keys)
        
    except Exception as e:
        logger.error(f"Eroare la validarea JSON: {str(e)}")
        return False, f"Eroare neașteptată la validare: {str(e)}", 0

def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează fișierul JSON cu îmbunătățiri pentru căutare și indexare optimizată.
    
    Returns:
        Lista de dicționare cu chunk-uri procesate și îmbunătățite
    """
    try:
        # Validăm mai întâi formatul
        is_vali)
        chunk_keys = [key for key in data.keys() if chunk_pattern.match(key)]
        
        # Sortăm chunk-urile după numărul lor
        def extract_chunk_number(key):
            match = chunk_pattern.match(key)
            return int(match.group(1)) if match else 0
        
        chunk_keys.sort(key=extract_chunk_number)
        
        logger.info(f"🔄 Procesare {len(chunk_keys)} chunk-uri cu optimizări...")
        
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
            
            # ÎMBUNĂTĂȚIRI pentru procesare optimizată
            
            # 1. Curățăm și normalizăm conținutul
            cleaned_content = clean_and_normalize_content(content)
            
            # 2. Extragem keywords avansate
            keywords = extract_advanced_keywords(cleaned_content)
            
            # 3. Detectăm limba (simplu)
            detected_language = detect_simple_language(cleaned_content)
            
            # 4. Calculăm statistici de conținut
            content_stats = calculate_content_statistics(cleaned_content)
            
            # 5. Extragem entități și concepte importante
            important_terms = extract_important_terms(cleaned_content)
            
            # 6. Creăm un hash pentru deduplicare
            content_hash = create_content_hash(cleaned_content)
            
            # Metadata îmbunătățită pentru ChromaDB
            enhanced_metadata = {
                # Metadata de bază
                'chunk_id': key,
                'chunk_index': i,
                'chunk_number': extract_chunk_number(key),
                'file_source': os.path.basename(file_path),
                'original_source': metadata_string,
                'source': metadata_string,  # Pentru compatibilitate
                'processed_at': datetime.now().isoformat(),
                'processing_version': "3.0.0",
                
                # Statistici de conținut
                'content_length': len(cleaned_content),
                'word_count': content_stats['word_count'],
                'sentence_count': content_stats['sentence_count'],
                'paragraph_count': content_stats['paragraph_count'],
                'avg_word_length': content_stats['avg_word_length'],
                'reading_complexity': content_stats['reading_complexity'],
                
                # Keywords și termeni importanți
                'keywords': ", ".join(keywords[:15]),  # Top 15 keywords
                'keywords_count': len(keywords),
                'important_terms': ", ".join(important_terms[:10]),  # Top 10 termeni importanți
                'important_terms_count': len(important_terms),
                
                # Metadata pentru căutare
                'language_detected': detected_language,
                'content_type': 'json_chunk',
                'content_hash': content_hash,
                
                # Metadata pentru ranking
                'has_technical_terms': any(term in cleaned_content.lower() for term in 
                    ['function', 'method', 'class', 'variable', 'parameter', 'return', 'import', 'export']),
                'has_code_snippets': bool(re.search(r'```|`.*`|\{.*\}|\(.*\)', cleaned_content)),
                'has_numbered_lists': bool(re.search(r'\d+\.\s', cleaned_content)),
                'has_bullet_points': bool(re.search(r'[•\-\*]\s', cleaned_content)),
                
                # Scoring hints pentru căutare
                'content_density': len(cleaned_content.split()) / max(1, cleaned_content.count('.') + 1),  # Cuvinte per propoziție
                'information_richness': len(set(keywords)) / max(1, len(keywords)),  # Diversitatea keywords
                'structural_elements': content_stats['structural_elements'],
            }
            
            chunks_data.append({
                'content': cleaned_content,
                'metadata': enhanced_metadata,
                'original_content': content,  # Păstrăm și originalul pentru debugging
                'quality_score': calculate_chunk_quality_score(cleaned_content, enhanced_metadata)
            })
        
        if not chunks_data:
            raise ValueError("Nu s-au putut extrage chunk-uri valide din fișier")
        
        # Statistici finale
        total_words = sum(chunk['metadata']['word_count'] for chunk in chunks_data)
        avg_quality = sum(chunk['quality_score'] for chunk in chunks_data) / len(chunks_data)
        
        logger.info(f"✅ Procesat cu succes: {len(chunks_data)} chunk-uri din {file_path}")
        logger.info(f"📊 Statistici: {total_words} cuvinte total, calitate medie: {avg_quality:.2f}")
        
        return chunks_data
        
    except Exception as e:
        logger.error(f"Eroare la procesarea fișierului {file_path}: {str(e)}")
        raise ValueError(f"Eroare neașteptată la procesarea fișierului: {str(e)}")

def clean_and_normalize_content(content: str) -> str:
    """
    Curăță și normalizează conținutul pentru indexare optimizată.
    """
    if not isinstance(content, str):
        return ""
    
    # 1. Eliminăm comentariile HTML/Markdown
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'<!-- image -->', '', content)
    
    # 2. Normalizăm spațiile și line breaks
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Păstrăm paragrafele
    
    # 3. Eliminăm caracterele de control și caractere speciale problematice
    content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\t')
    
    # 4. Normalizăm punctuația
    content = re.sub(r'[""''`]', '"', content)  # Uniformizăm ghilimelele
    content = re.sub(r'[–—]', '-', content)     # Uniformizăm liniuțele
    
    # 5. Eliminăm liniile care sunt doar punctuație sau numere
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and not re.match(r'^[^\w]*"""
Utilitare OPTIMIZATE pentru procesarea fișierelor JSON chunkizate
Versiunea 3.0.0 - Îmbunătățiri pentru căutare și procesare
"""

import json
import os
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)

def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează că fișierul JSON are formatul EXACT specificat cu verificări îmbunătățite.
    
    Formatul acceptat:
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
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Fișierul este gol", 0
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return False, f"Fișierul este prea mare ({file_size // 1024 // 1024}MB). Maximum 100MB.", 0
        
        # Încercăm să încărcăm JSON-ul
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"JSON invalid: {str(e)[:100]}...", 0
        
        # Verificăm că este un dicționar
        if not isinstance(data, dict):
            return False, "JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip", 0
        
        if len(data) == 0:
            return False, "JSON-ul nu conține date", 0
        
        # Căutăm chunk-uri în formatul chunk_X
        chunk_pattern = re.compile(r'^chunk_\d+$')
        chunk_keys = [key for key in data.keys() if chunk_pattern.match(key)]
        
        if len(chunk_keys) == 0:
            # Încercăm să găsim alte patterns comune
            other_keys = list(data.keys())[:10]  # Primele 10 chei pentru debugging
            return False, f"Nu s-au găsit chunk-uri în formatul așteptat (chunk_0, chunk_1, etc.). Chei găsite: {other_keys}", 0
        
        # Validăm structura chunk-urilor - verificăm mai multe pentru siguranță
        valid_chunks = 0
        invalid_chunks = []
        sample_content_lengths = []
        
        for key in chunk_keys[:min(10, len(chunk_keys))]:  # Verificăm până la 10 chunk-uri
            chunk = data[key]
            
            # Verificăm că chunk-ul este un dicționar
            if not isinstance(chunk, dict):
                invalid_chunks.append(f"{key}: nu este dicționar")
                continue
            
            # Verificăm că are EXACT cheile: "metadata" și "chunk"
            required_keys = {"metadata", "chunk"}
            chunk_keys_set = set(chunk.keys())
            
            if not required_keys.issubset(chunk_keys_set):
                missing_keys = required_keys - chunk_keys_set
                invalid_chunks.append(f"{key}: lipsesc cheile {missing_keys}")
                continue
            
            # Verificăm că metadata este STRING
            if not isinstance(chunk["metadata"], str):
                invalid_chunks.append(f"{key}: metadata nu este string (este {type(chunk['metadata']).__name__})")
                continue
            
            if len(chunk["metadata"].strip()) == 0:
                invalid_chunks.append(f"{key}: metadata este string gol")
                continue
            
            # Verificăm că chunk este STRING cu conținut suficient
            if not isinstance(chunk["chunk"], str):
                invalid_chunks.append(f"{key}: chunk nu este string (este {type(chunk['chunk']).__name__})")
                continue
            
            content = chunk["chunk"].strip()
            if len(content) < 10:
                invalid_chunks.append(f"{key}: conținut prea scurt ({len(content)} caractere)")
                continue
            
            sample_content_lengths.append(len(content))
            valid_chunks += 1
        
        if valid_chunks == 0:
            error_details = "; ".join(invalid_chunks[:5])  # Primele 5 erori
            return False, f"Nu s-au găsit chunk-uri valide. Erori: {error_details}", 0
        
        # Verificăm consistența numerotării chunk-urilor
        chunk_numbers = []
        for key in chunk_keys:
            match = re.match(r'chunk_(\d+)', key)
            if match:
                chunk_numbers.append(int(match.group(1)))
        
        chunk_numbers.sort()
        expected_sequence = list(range(len(chunk_numbers)))
        
        # Warning pentru numerotare inconsistentă (nu blochez validarea)
        if chunk_numbers != expected_sequence:
            logger.warning(f"Numerotarea chunk-urilor nu este consecutivă: găsite {chunk_numbers[:10]}..., așteptate {expected_sequence[:10]}...")
        
        # Statistici pentru logging
        avg_length = sum(sample_content_lengths) / len(sample_content_lengths) if sample_content_lengths else 0
        
        logger.info(f"✅ JSON valid găsit cu {len(chunk_keys)} chunk-uri")
        logger.info(f"📊 Validare: {valid_chunks}/{min(10, len(chunk_keys))} chunk-uri verificate cu succes")
        logger.info(f"📏 Lungime medie conținut: {avg_length:.0f} caractere")
        
        return True, f"Valid - {len(chunk_keys)} chunk-uri găsite", len(chunk_keys)
        
    except Exception as e:
        logger.error(f"Eroare la validarea JSON: {str(e)}")
        return False, f"Eroare neașteptată la validare: {str(e)}", 0

def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează fișierul JSON cu îmbunătățiri pentru căutare și indexare optimizată.
    
    Returns:
        Lista de dicționare cu chunk-uri procesate și îmbunătățite
    """
    try:
        # Validăm mai întâi formatul
        is_vali, line) and len(line) > 3:
            cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    # 6. Eliminăm spațiile în plus
    content = content.strip()
    
    return content

def extract_advanced_keywords(content: str, max_keywords: int = 20) -> List[str]:
    """
    Extrage keywords avansate cu filtrare și scoring îmbunătățit.
    """
    if not content:
        return []
    
    # Normalizăm textul
    text_normalized = re.sub(r'[^\w\s]', ' ', content.lower())
    words = text_normalized.split()
    
    # Stop words extinse în română și engleză
    stop_words = {
        # Română - extinse
        'și', 'în', 'la', 'de', 'cu', 'pe', 'din', 'pentru', 'este', 'sunt', 'a', 'al', 'ale',
        'că', 'să', 'se', 'nu', 'mai', 'dar', 'sau', 'dacă', 'când', 'cum', 'unde', 'care',
        'această', 'acest', 'aceasta', 'acesta', 'unei', 'unui', 'o', 'un', 'am', 'ai', 'au',
        'avea', 'fi', 'fost', 'fiind', 'va', 'vor', 'foarte', 'mult', 'puțin', 'către', 'despre',
        'după', 'înainte', 'asupra', 'printre', 'între', 'sub', 'peste', 'aproape', 'departe',
        'aici', 'acolo', 'undeva', 'oriunde', 'nicăieri', 'atunci', 'acum', 'ieri', 'mâine',
        'întotdeauna', 'niciodată', 'uneori', 'adesea', 'rar', 'prima', 'primul', 'ultima',
        'ultimul', 'toate', 'toți', 'fiecare', 'oricare', 'niciunul', 'niciuna', 'alt', 'alta',
        
        # Engleză - extinse
        'and', 'in', 'to', 'of', 'with', 'on', 'from', 'for', 'is', 'are', 'the', 'a', 'an',
        'that', 'this', 'it', 'be', 'have', 'has', 'do', 'does', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'must', 'shall', 'at', 'by', 'up', 'as', 'if',
        'what', 'when', 'where', 'who', 'why', 'how', 'which', 'than', 'or', 'but', 'not',
        'was', 'were', 'been', 'being', 'had', 'having', 'very', 'more', 'most', 'other',
        'some', 'all', 'any', 'each', 'every', 'few', 'many', 'much', 'several', 'such',
        'first', 'last', 'next', 'previous', 'same', 'different', 'new', 'old', 'good', 'bad',
        'big', 'small', 'long', 'short', 'high', 'low', 'right', 'left', 'here', 'there',
        'now', 'then', 'today', 'yesterday', 'tomorrow', 'always', 'never', 'sometimes',
        'often', 'rarely', 'usually', 'generally', 'probably', 'maybe', 'perhaps', 'also',
        'too', 'only', 'just', 'even', 'still', 'already', 'yet', 'again', 'once', 'twice'
    }
    
    # Calculăm frecvența cuvintelor cu scoruri îmbunătățite
    word_scores = Counter()
    
    for i, word in enumerate(words):
        if (len(word) > 2 and 
            word not in stop_words and 
            not word.isdigit() and
            word.isalpha()):  # Doar cuvinte cu litere
            
            # Scor de bază pe frecvență
            base_score = 1
            
            # Bonus pentru cuvinte mai lungi (sunt mai specifice)
            if len(word) > 6:
                base_score += 0.5
            elif len(word) > 4:
                base_score += 0.2
            
            # Bonus pentru cuvinte care apar la începutul textului (sunt mai importante)
            if i < len(words) * 0.2:  # Primele 20%
                base_score += 0.3
            
            # Bonus pentru cuvinte tehnice/specifice
            if any(pattern in word for pattern in ['tion', 'sion', 'ment', 'ing', 'ize', 'ise']):
                base_score += 0.2
            
            # Bonus pentru cuvinte cu majuscule în textul original (pot fi nume proprii)
            if any(word.capitalize() in content for _ in range(1)):
                base_score += 0.1
            
            word_scores[word] += base_score
    
    # Extragem și compound terms (2-3 cuvinte consecutive)
    compound_terms = []
    for i in range(len(words) - 1):
        if i < len(words) - 2:
            # Trigram
            trigram = ' '.join(words[i:i+3])
            if (all(len(w) > 2 and w not in stop_words for w in words[i:i+3]) and
                len(trigram) > 10):
                compound_terms.append((trigram, 2.0))  # Scor mai mare pentru trigrams
        
        # Bigram
        bigram = ' '.join(words[i:i+2])
        if (all(len(w) > 2 and w not in stop_words for w in words[i:i+2]) and
            len(bigram) > 6):
            compound_terms.append((bigram, 1.5))  # Scor moderat pentru bigrams
    
    # Combinăm single words și compound terms
    all_terms = []
    
    # Adăugăm single words
    for word, score in word_scores.most_common(max_keywords):
        all_terms.append((word, score))
    
    # Adăugăm compound terms (până la 1/3 din total)
    compound_limit = max_keywords // 3
    for term, score in sorted(compound_terms, key=lambda x: x[1], reverse=True)[:compound_limit]:
        all_terms.append((term, score))
    
    # Sortăm după scor și returnăm
    all_terms.sort(key=lambda x: x[1], reverse=True)
    keywords = [term for term, score in all_terms[:max_keywords]]
    
    return keywords

def detect_simple_language(content: str) -> str:
    """
    Detectează limba conținutului (simplu, pe baza unor pattern-uri).
    """
    if not content:
        return "unknown"
    
    content_lower = content.lower()
    
    # Contorizăm cuvinte specifice fiecărei limbi
    romanian_indicators = [
        'și', 'în', 'cu', 'pentru', 'că', 'să', 'de', 'la', 'pe', 'din',
        'este', 'sunt', 'avea', 'acest', 'această', 'aceasta', 'României'
    ]
    
    english_indicators = [
        'the', 'and', 'with', 'for', 'that', 'this', 'have', 'has', 'are', 'is',
        'to', 'of', 'in', 'on', 'at', 'by', 'from', 'will', 'would', 'could'
    ]
    
    ro_count = sum(1 for word in romanian_indicators if f' {word} ' in f' {content_lower} ')
    en_count = sum(1 for word in english_indicators if f' {word} ' in f' {content_lower} ')
    
    if ro_count > en_count and ro_count > 3:
        return "romanian"
    elif en_count > ro_count and en_count > 3:
        return "english"
    elif ro_count > 0 or en_count > 0:
        return "mixed"
    else:
        return "unknown"

def calculate_content_statistics(content: str) -> Dict[str, Any]:
    """
    Calculează statistici detaliate despre conținut.
    """
    if not content:
        return {
            'word_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'avg_word_length': 0,
            'reading_complexity': 'low',
            'structural_elements': 0
        }
    
    # Statistici de bază
    words = content.split()
    word_count = len(words)
    
    # Calculăm propoziții (aproximativ)
    sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
    sentence_count = len(sentences)
    
    # Calculăm paragrafe
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    # Lungimea medie a cuvintelor
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    
    # Complexitatea de citire (simplificată)
    if avg_word_length > 6 and word_count > 100:
        reading_complexity = 'high'
    elif avg_word_length > 4 and word_count > 50:
        reading_complexity = 'medium'
    else:
        reading_complexity = 'low'
    
    # Elemente structurale
    structural_elements = 0
    structural_elements += len(re.findall(r'#+\s', content))  # Headers
    structural_elements += len(re.findall(r'\d+\.\s', content))  # Numbered lists
    structural_elements += len(re.findall(r'[•\-\*]\s', content))  # Bullet points
    structural_elements += len(re.findall(r'```|`.*`', content))  # Code blocks
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'avg_word_length': round(avg_word_length, 2),
        'reading_complexity': reading_complexity,
        'structural_elements': structural_elements
    }

def extract_important_terms(content: str, max_terms: int = 15) -> List[str]:
    """
    Extrage termeni importanți (entități, concepte tehnice, etc.).
    """
    if not content:
        return []
    
    important_terms = []
    
    # 1. Cuvinte cu majuscule (posibile nume proprii, acronime)
    capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', content)
    important_terms.extend(capitalized_words[:5])
    
    # 2. Acronime (2-5 litere mari)
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', content)
    important_terms.extend(acronyms[:3])
    
    # 3. Termeni tehnici comuni
    technical_patterns = [
        r'\b\w*(?:tion|sion|ment|ing|ity|ness)\b',  # Sufixe tehnice
        r'\b(?:config|setup|install|process|method|function|class|object)\w*\b',  # Termeni tehnici
        r'\b\w*(?:able|ible|ful|less|ous|ive)\b',  # Adjective complexe
    ]
    
    for pattern in technical_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        important_terms.extend([m for m in matches if len(m) > 4][:3])
    
    # 4. Numere și versiuni
    versions = re.findall(r'\b\d+\.\d+(?:\.\d+)?\b', content)
    important_terms.extend(versions[:2])
    
    # 5. URLs și paths (fără protocol)
    paths = re.findall(r'\b\w+[/\\]\w+(?:[/\\]\w+)*\b', content)
    important_terms.extend(paths[:2])
    
    # Curățăm și eliminăm duplicatele
    important_terms = list(set([term.strip() for term in important_terms if len(term.strip()) > 2]))
    
    # Sortăm după lungime (termenii mai lungi sunt mai specifici)
    important_terms.sort(key=len, reverse=True)
    
    return important_terms[:max_terms]

def create_content_hash(content: str) -> str:
    """
    Creează un hash unic pentru conținut (pentru deduplicare).
    """
    if not content:
        return ""
    
    # Normalizăm conținutul pentru hash consistent
    normalized = re.sub(r'\s+', ' ', content.lower().strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]

def calculate_chunk_quality_score(content: str, metadata: Dict[str, Any]) -> float:
    """
    Calculează un scor de calitate pentru chunk (0-1).
    """
    if not content:
        return 0.0
    
    score = 0.5  # Scor de bază
    
    # Bonus pentru lungimea optimă
    content_length = len(content)
    if 200 <= content_length <= 2000:
        score += 0.2
    elif 100 <= content_length < 200 or 2000 < content_length <= 3000:
        score += 0.1
    elif content_length < 50:
        score -= 0.2
    
    # Bonus pentru diversitatea keywords
    keywords_count = metadata.get('keywords_count', 0)
    if keywords_count > 10:
        score += 0.1
    elif keywords_count > 5:
        score += 0.05
    
    # Bonus pentru complexitatea de citire
    complexity = metadata.get('reading_complexity', 'low')
    if complexity == 'medium':
        score += 0.1
    elif complexity == 'high':
        score += 0.05
    
    # Bonus pentru elemente structurale
    structural = metadata.get('structural_elements', 0)
    if structural > 3:
        score += 0.1
    elif structural > 1:
        score += 0.05
    
    # Bonus pentru termeni importanți
    important_count = metadata.get('important_terms_count', 0)
    if important_count > 5:
        score += 0.05
    
    return min(1.0, max(0.0, score))

def get_json_statistics(file_path: str) -> Dict[str, Any]:
    """
    Obține statistici îmbunătățite despre fișierul JSON chunkizat.
    
    Returns:
        Dicționar cu statistici detaliate despre fișier
    """
    try:
        if not os.path.exists(file_path):
            return {"error": "Fișierul nu există"}
        
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            return {"error": "JSON-ul nu este un dicționar"}
        
        # Analiza chunk-urilor îmbunătățită
        chunk_pattern = re.compile(r'^chunk_(\d+)"""
Utilitare OPTIMIZATE pentru procesarea fișierelor JSON chunkizate
Versiunea 3.0.0 - Îmbunătățiri pentru căutare și procesare
"""

import json
import os
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)

def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează că fișierul JSON are formatul EXACT specificat cu verificări îmbunătățite.
    
    Formatul acceptat:
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
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Fișierul este gol", 0
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return False, f"Fișierul este prea mare ({file_size // 1024 // 1024}MB). Maximum 100MB.", 0
        
        # Încercăm să încărcăm JSON-ul
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"JSON invalid: {str(e)[:100]}...", 0
        
        # Verificăm că este un dicționar
        if not isinstance(data, dict):
            return False, "JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip", 0
        
        if len(data) == 0:
            return False, "JSON-ul nu conține date", 0
        
        # Căutăm chunk-uri în formatul chunk_X
        chunk_pattern = re.compile(r'^chunk_\d+$')
        chunk_keys = [key for key in data.keys() if chunk_pattern.match(key)]
        
        if len(chunk_keys) == 0:
            # Încercăm să găsim alte patterns comune
            other_keys = list(data.keys())[:10]  # Primele 10 chei pentru debugging
            return False, f"Nu s-au găsit chunk-uri în formatul așteptat (chunk_0, chunk_1, etc.). Chei găsite: {other_keys}", 0
        
        # Validăm structura chunk-urilor - verificăm mai multe pentru siguranță
        valid_chunks = 0
        invalid_chunks = []
        sample_content_lengths = []
        
        for key in chunk_keys[:min(10, len(chunk_keys))]:  # Verificăm până la 10 chunk-uri
            chunk = data[key]
            
            # Verificăm că chunk-ul este un dicționar
            if not isinstance(chunk, dict):
                invalid_chunks.append(f"{key}: nu este dicționar")
                continue
            
            # Verificăm că are EXACT cheile: "metadata" și "chunk"
            required_keys = {"metadata", "chunk"}
            chunk_keys_set = set(chunk.keys())
            
            if not required_keys.issubset(chunk_keys_set):
                missing_keys = required_keys - chunk_keys_set
                invalid_chunks.append(f"{key}: lipsesc cheile {missing_keys}")
                continue
            
            # Verificăm că metadata este STRING
            if not isinstance(chunk["metadata"], str):
                invalid_chunks.append(f"{key}: metadata nu este string (este {type(chunk['metadata']).__name__})")
                continue
            
            if len(chunk["metadata"].strip()) == 0:
                invalid_chunks.append(f"{key}: metadata este string gol")
                continue
            
            # Verificăm că chunk este STRING cu conținut suficient
            if not isinstance(chunk["chunk"], str):
                invalid_chunks.append(f"{key}: chunk nu este string (este {type(chunk['chunk']).__name__})")
                continue
            
            content = chunk["chunk"].strip()
            if len(content) < 10:
                invalid_chunks.append(f"{key}: conținut prea scurt ({len(content)} caractere)")
                continue
            
            sample_content_lengths.append(len(content))
            valid_chunks += 1
        
        if valid_chunks == 0:
            error_details = "; ".join(invalid_chunks[:5])  # Primele 5 erori
            return False, f"Nu s-au găsit chunk-uri valide. Erori: {error_details}", 0
        
        # Verificăm consistența numerotării chunk-urilor
        chunk_numbers = []
        for key in chunk_keys:
            match = re.match(r'chunk_(\d+)', key)
            if match:
                chunk_numbers.append(int(match.group(1)))
        
        chunk_numbers.sort()
        expected_sequence = list(range(len(chunk_numbers)))
        
        # Warning pentru numerotare inconsistentă (nu blochez validarea)
        if chunk_numbers != expected_sequence:
            logger.warning(f"Numerotarea chunk-urilor nu este consecutivă: găsite {chunk_numbers[:10]}..., așteptate {expected_sequence[:10]}...")
        
        # Statistici pentru logging
        avg_length = sum(sample_content_lengths) / len(sample_content_lengths) if sample_content_lengths else 0
        
        logger.info(f"✅ JSON valid găsit cu {len(chunk_keys)} chunk-uri")
        logger.info(f"📊 Validare: {valid_chunks}/{min(10, len(chunk_keys))} chunk-uri verificate cu succes")
        logger.info(f"📏 Lungime medie conținut: {avg_length:.0f} caractere")
        
        return True, f"Valid - {len(chunk_keys)} chunk-uri găsite", len(chunk_keys)
        
    except Exception as e:
        logger.error(f"Eroare la validarea JSON: {str(e)}")
        return False, f"Eroare neașteptată la validare: {str(e)}", 0

def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează fișierul JSON cu îmbunătățiri pentru căutare și indexare optimizată.
    
    Returns:
        Lista de dicționare cu chunk-uri procesate și îmbunătățite
    """
    try:
        # Validăm mai întâi formatul
        is_vali)
        chunk_keys = [key for key in data.keys() if chunk_pattern.match(key)]
        
        if not chunk_keys:
            return {"error": "Nu s-au găsit chunk-uri"}
        
        # Statistici detaliate
        total_content_length = 0
        total_words = 0
        valid_chunks = 0
        metadata_sources = set()
        content_samples = []
        quality_scores = []
        language_distribution = {'romanian': 0, 'english': 0, 'mixed': 0, 'unknown': 0}
        
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
                    
                    # Procesăm conținutul
                    cleaned_content = clean_and_normalize_content(content)
                    content_stats = calculate_content_statistics(cleaned_content)
                    
                    total_content_length += len(cleaned_content)
                    total_words += content_stats['word_count']
                    metadata_sources.add(metadata_string)
                    
                    # Sample pentru preview
                    if len(content_samples) < 3:
                        content_samples.append({
                            'chunk_id': key,
                            'preview': cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content,
                            'word_count': content_stats['word_count'],
                            'complexity': content_stats['reading_complexity']
                        })
                    
                    # Detectăm limba
                    language = detect_simple_language(cleaned_content)
                    if language in language_distribution:
                        language_distribution[language] += 1
                    
                    # Calculăm calitatea
                    metadata_enhanced = {
                        'keywords_count': len(extract_advanced_keywords(cleaned_content, 10)),
                        'reading_complexity': content_stats['reading_complexity'],
                        'structural_elements': content_stats['structural_elements'],
                        'important_terms_count': len(extract_important_terms(cleaned_content, 10))
                    }
                    quality_score = calculate_chunk_quality_score(cleaned_content, metadata_enhanced)
                    quality_scores.append(quality_score)
        
        # Calculăm statistici finale
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        dominant_language = max(language_distribution.items(), key=lambda x: x[1])[0] if any(language_distribution.values()) else 'unknown'
        
        stats = {
            'file_info': {
                'file_name': os.path.basename(file_path),
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'processed_at': datetime.now().isoformat()
            },
            'chunk_analysis': {
                'total_chunks': len(chunk_keys),
                'valid_chunks': valid_chunks,
                'invalid_chunks': len(chunk_keys) - valid_chunks,
                'chunk_numbering_consistent': len(chunk_keys) == max([int(re.match(r'chunk_(\d+)', key).group(1)) for key in chunk_keys]) + 1 if chunk_keys else True
            },
            'content_statistics': {
                'total_content_length': total_content_length,
                'total_words': total_words,
                'average_chunk_length': round(total_content_length / valid_chunks) if valid_chunks > 0 else 0,
                'average_words_per_chunk': round(total_words / valid_chunks) if valid_chunks > 0 else 0,
                'average_quality_score': round(avg_quality, 3)
            },
            'metadata_analysis': {
                'unique_metadata_sources': len(metadata_sources),
                'metadata_sources_list': list(metadata_sources)[:10],  # Primele 10
                'dominant_language': dominant_language,
                'language_distribution': language_distribution
            },
            'content_samples': content_samples,
            'quality_distribution': {
                'high_quality': len([s for s in quality_scores if s > 0.7]),
                'medium_quality': len([s for s in quality_scores if 0.4 <= s <= 0.7]),
                'low_quality': len([s for s in quality_scores if s < 0.4]),
            },
            'processing_recommendations': generate_processing_recommendations(
                valid_chunks, avg_quality, dominant_language, len(metadata_sources)
            )
        }
        
        logger.debug(f"📊 Statistici îmbunătățite generate pentru {file_path}: {valid_chunks} chunk-uri valide, calitate {avg_quality:.2f}")
        return stats
        
    except Exception as e:
        logger.error(f"Eroare la generarea statisticilor pentru {file_path}: {str(e)}")
        return {"error": f"Eroare la analiza fișierului: {str(e)}"}

def generate_processing_recommendations(valid_chunks: int, avg_quality: float, language: str, source_count: int) -> List[str]:
    """
    Generează recomandări pentru procesarea optimă a fișierului.
    """
    recommendations = []
    
    if valid_chunks < 5:
        recommendations.append("Fișierul conține puține chunk-uri. Considerați combinarea cu alte fișiere.")
    elif valid_chunks > 1000:
        recommendations.append("Fișierul este foarte mare. Considerați împărțirea în fișiere mai mici.")
    
    if avg_quality < 0.5:
        recommendations.append("Calitatea chunk-urilor este scăzută. Verificați formatarea și conținutul.")
    elif avg_quality > 0.8:
        recommendations.append("Chunk-urile au calitate excelentă. Ideal pentru indexare și căutare.")
    
    if language == 'mixed':
        recommendations.append("Conținutul este în limbi mixte. Căutarea hibridă va fi foarte utilă.")
    elif language == 'unknown':
        recommendations.append("Limba nu a putut fi detectată. Verificați codificarea și conținutul textului.")
    
    if source_count == 1:
        recommendations.append("Toate chunk-urile provin din aceeași sursă. Diversificați sursele pentru rezultate mai bune.")
    elif source_count > 10:
        recommendations.append("Multe surse diferite detectate. Excelent pentru diversitatea conținutului.")
    
    if not recommendations:
        recommendations.append("Fișierul este optimizat pentru procesare. Nu sunt necesare ajustări.")
    
    return recommendations

def preview_json_chunks(file_path: str, max_chunks: int = 3) -> Dict[str, Any]:
    """
    Oferă o previzualizare îmbunătățită a chunk-urilor din fișier.
    
    Args:
        file_path: Calea către fișierul JSON
        max_chunks: Numărul maxim de chunk-uri de previzualizat
        
    Returns:
        Dicționar cu previzualizarea îmbunătățită a chunk-urilor
    """
    try:
        if not os.path.exists(file_path):
            return {"error": "Fișierul nu există"}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            return {"error": "JSON-ul nu este un dicționar"}
        
        chunk_pattern = re.compile(r'^chunk_(\d+)"""
Utilitare OPTIMIZATE pentru procesarea fișierelor JSON chunkizate
Versiunea 3.0.0 - Îmbunătățiri pentru căutare și procesare
"""

import json
import os
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)

def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează că fișierul JSON are formatul EXACT specificat cu verificări îmbunătățite.
    
    Formatul acceptat:
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
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Fișierul este gol", 0
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return False, f"Fișierul este prea mare ({file_size // 1024 // 1024}MB). Maximum 100MB.", 0
        
        # Încercăm să încărcăm JSON-ul
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"JSON invalid: {str(e)[:100]}...", 0
        
        # Verificăm că este un dicționar
        if not isinstance(data, dict):
            return False, "JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip", 0
        
        if len(data) == 0:
            return False, "JSON-ul nu conține date", 0
        
        # Căutăm chunk-uri în formatul chunk_X
        chunk_pattern = re.compile(r'^chunk_\d+$')
        chunk_keys = [key for key in data.keys() if chunk_pattern.match(key)]
        
        if len(chunk_keys) == 0:
            # Încercăm să găsim alte patterns comune
            other_keys = list(data.keys())[:10]  # Primele 10 chei pentru debugging
            return False, f"Nu s-au găsit chunk-uri în formatul așteptat (chunk_0, chunk_1, etc.). Chei găsite: {other_keys}", 0
        
        # Validăm structura chunk-urilor - verificăm mai multe pentru siguranță
        valid_chunks = 0
        invalid_chunks = []
        sample_content_lengths = []
        
        for key in chunk_keys[:min(10, len(chunk_keys))]:  # Verificăm până la 10 chunk-uri
            chunk = data[key]
            
            # Verificăm că chunk-ul este un dicționar
            if not isinstance(chunk, dict):
                invalid_chunks.append(f"{key}: nu este dicționar")
                continue
            
            # Verificăm că are EXACT cheile: "metadata" și "chunk"
            required_keys = {"metadata", "chunk"}
            chunk_keys_set = set(chunk.keys())
            
            if not required_keys.issubset(chunk_keys_set):
                missing_keys = required_keys - chunk_keys_set
                invalid_chunks.append(f"{key}: lipsesc cheile {missing_keys}")
                continue
            
            # Verificăm că metadata este STRING
            if not isinstance(chunk["metadata"], str):
                invalid_chunks.append(f"{key}: metadata nu este string (este {type(chunk['metadata']).__name__})")
                continue
            
            if len(chunk["metadata"].strip()) == 0:
                invalid_chunks.append(f"{key}: metadata este string gol")
                continue
            
            # Verificăm că chunk este STRING cu conținut suficient
            if not isinstance(chunk["chunk"], str):
                invalid_chunks.append(f"{key}: chunk nu este string (este {type(chunk['chunk']).__name__})")
                continue
            
            content = chunk["chunk"].strip()
            if len(content) < 10:
                invalid_chunks.append(f"{key}: conținut prea scurt ({len(content)} caractere)")
                continue
            
            sample_content_lengths.append(len(content))
            valid_chunks += 1
        
        if valid_chunks == 0:
            error_details = "; ".join(invalid_chunks[:5])  # Primele 5 erori
            return False, f"Nu s-au găsit chunk-uri valide. Erori: {error_details}", 0
        
        # Verificăm consistența numerotării chunk-urilor
        chunk_numbers = []
        for key in chunk_keys:
            match = re.match(r'chunk_(\d+)', key)
            if match:
                chunk_numbers.append(int(match.group(1)))
        
        chunk_numbers.sort()
        expected_sequence = list(range(len(chunk_numbers)))
        
        # Warning pentru numerotare inconsistentă (nu blochez validarea)
        if chunk_numbers != expected_sequence:
            logger.warning(f"Numerotarea chunk-urilor nu este consecutivă: găsite {chunk_numbers[:10]}..., așteptate {expected_sequence[:10]}...")
        
        # Statistici pentru logging
        avg_length = sum(sample_content_lengths) / len(sample_content_lengths) if sample_content_lengths else 0
        
        logger.info(f"✅ JSON valid găsit cu {len(chunk_keys)} chunk-uri")
        logger.info(f"📊 Validare: {valid_chunks}/{min(10, len(chunk_keys))} chunk-uri verificate cu succes")
        logger.info(f"📏 Lungime medie conținut: {avg_length:.0f} caractere")
        
        return True, f"Valid - {len(chunk_keys)} chunk-uri găsite", len(chunk_keys)
        
    except Exception as e:
        logger.error(f"Eroare la validarea JSON: {str(e)}")
        return False, f"Eroare neașteptată la validare: {str(e)}", 0

def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează fișierul JSON cu îmbunătățiri pentru căutare și indexare optimizată.
    
    Returns:
        Lista de dicționare cu chunk-uri procesate și îmbunătățite
    """
    try:
        # Validăm mai întâi formatul
        is_vali)
        chunk_keys = [key for key in data.keys() if chunk_pattern.match(key)]
        
        if not chunk_keys:
            return {"error": "Nu s-au găsit chunk-uri"}
        
        # Sortăm chunk-urile
        def extract_chunk_number(key):
            match = chunk_pattern.match(key)
            return int(match.group(1)) if match else 0
        
        chunk_keys.sort(key=extract_chunk_number)
        
        # Selectăm chunk-urile pentru previzualizare
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
                
                # Procesăm conținutul pentru previzualizare
                cleaned_content = clean_and_normalize_content(content)
                content_stats = calculate_content_statistics(cleaned_content)
                keywords = extract_advanced_keywords(cleaned_content, 8)
                important_terms = extract_important_terms(cleaned_content, 5)
                language = detect_simple_language(cleaned_content)
                
                # Creăm preview-ul conținutului
                if len(cleaned_content) > 300:
                    preview_content = cleaned_content[:300] + "..."
                else:
                    preview_content = cleaned_content
                
                # Calculăm scorul de calitate
                metadata_enhanced = {
                    'keywords_count': len(keywords),
                    'reading_complexity': content_stats['reading_complexity'],
                    'structural_elements': content_stats['structural_elements'],
                    'important_terms_count': len(important_terms)
                }
                quality_score = calculate_chunk_quality_score(cleaned_content, metadata_enhanced)
                
                preview_chunks.append({
                    'chunk_id': key,
                    'chunk_number': extract_chunk_number(key),
                    'content_preview': preview_content,
                    'content_length': len(cleaned_content),
                    'word_count': content_stats['word_count'],
                    'sentence_count': content_stats['sentence_count'],
                    'metadata': metadata_string,
                    'keywords': keywords[:5],  # Top 5 keywords
                    'important_terms': important_terms[:3],  # Top 3 termeni importanți
                    'language_detected': language,
                    'reading_complexity': content_stats['reading_complexity'],
                    'quality_score': round(quality_score, 3),
                    'has_structural_elements': content_stats['structural_elements'] > 0,
                    'processing_suggestions': generate_chunk_suggestions(cleaned_content, content_stats)
                })
        
        # Calculăm statistici generale pentru previzualizare
        total_quality = sum(chunk['quality_score'] for chunk in preview_chunks)
        avg_quality = total_quality / len(preview_chunks) if preview_chunks else 0
        
        languages = [chunk['language_detected'] for chunk in preview_chunks]
        language_counts = Counter(languages)
        dominant_language = language_counts.most_common(1)[0][0] if language_counts else 'unknown'
        
        result = {
            'file_info': {
                'file_name': os.path.basename(file_path),
                'total_chunks': len(chunk_keys),
                'preview_count': len(preview_chunks),
                'showing': f"{min(len(preview_chunks), max_chunks)} din {len(chunk_keys)} chunk-uri"
            },
            'preview_chunks': preview_chunks,
            'preview_analysis': {
                'average_quality': round(avg_quality, 3),
                'dominant_language': dominant_language,
                'language_distribution': dict(language_counts),
                'total_words_preview': sum(chunk['word_count'] for chunk in preview_chunks),
                'complexity_distribution': Counter(chunk['reading_complexity'] for chunk in preview_chunks)
            },
            'recommendations': generate_preview_recommendations(preview_chunks, avg_quality, dominant_language)
        }
        
        logger.debug(f"👁️ Previzualizare îmbunătățită generată pentru {file_path}: {len(preview_chunks)} chunk-uri")
        return result
        
    except Exception as e:
        logger.error(f"Eroare la previzualizarea fișierului {file_path}: {str(e)}")
        return {"error": f"Eroare la previzualizare: {str(e)}"}

def generate_chunk_suggestions(content: str, stats: Dict[str, Any]) -> List[str]:
    """
    Generează sugestii specifice pentru îmbunătățirea unui chunk.
    """
    suggestions = []
    
    if stats['word_count'] < 20:
        suggestions.append("Chunk foarte scurt - considerați combinarea cu chunk-uri adiacente")
    elif stats['word_count'] > 500:
        suggestions.append("Chunk foarte lung - considerați împărțirea în chunk-uri mai mici")
    
    if stats['sentence_count'] < 2:
        suggestions.append("Adăugați mai multe propoziții pentru context mai bogat")
    
    if stats['reading_complexity'] == 'high':
        suggestions.append("Complexitate mare - excelent pentru conținut tehnic detaliat")
    elif stats['reading_complexity'] == 'low':
        suggestions.append("Complexitate scăzută - considerați adăugarea de detalii")
    
    if stats['structural_elements'] == 0:
        suggestions.append("Adăugați elemente structurale (liste, headers) pentru organizare mai bună")
    
    if not suggestions:
        suggestions.append("Chunk-ul este bine structurat și optimizat")
    
    return suggestions

def generate_preview_recommendations(chunks: List[Dict], avg_quality: float, language: str) -> List[str]:
    """
    Generează recomandări bazate pe previzualizarea chunk-urilor.
    """
    recommendations = []
    
    if avg_quality < 0.4:
        recommendations.append("🔧 Calitatea chunk-urilor este scăzută. Reviziți formatarea și conținutul.")
    elif avg_quality > 0.8:
        recommendations.append("✅ Chunk-urile au calitate excelentă. Fișierul este optimizat pentru căutare.")
    
    word_counts = [chunk['word_count'] for chunk in chunks]
    if word_counts:
        avg_words = sum(word_counts) / len(word_counts)
        if avg_words < 30:
            recommendations.append("📝 Chunk-urile sunt scurte. Considerați combinarea pentru context mai bogat.")
        elif avg_words > 300:
            recommendations.append("📄 Chunk-urile sunt lungi. Considerați împărțirea pentru căutare mai precisă.")
    
    if language == 'mixed':
        recommendations.append("🌐 Conținut multilingv detectat. Activați căutarea hibridă pentru rezultate optime.")
    elif language == 'unknown':
        recommendations.append("❓ Limba nedeterminată. Verificați codificarea textului.")
    
    # Analizăm diversitatea keywords
    all_keywords = []
    for chunk in chunks:
        all_keywords.extend(chunk.get('keywords', []))
    
    unique_keywords = len(set(all_keywords))
    total_keywords = len(all_keywords)
    
    if total_keywords > 0:
        diversity_ratio = unique_keywords / total_keywords
        if diversity_ratio > 0.8:
            recommendations.append("🎯 Diversitate mare de keywords. Excelent pentru căutări variate.")
        elif diversity_ratio < 0.4:
            recommendations.append("🔄 Keywords repetitive. Considerați diversificarea vocabularului.")
    
    if not recommendations:
        recommendations.append("🎉 Fișierul este excelent optimizat pentru sistemul RAG!")
    
    return recommendations

def validate_chunk_structure(chunk_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validează structura EXACTĂ a unui chunk individual cu verificări îmbunătățite.
    
    Args:
        chunk_data: Datele chunk-ului de validat
        
    Returns:
        (is_valid, error_message)
    """
    try:
        if not isinstance(chunk_data, dict):
            return False, "Chunk-ul trebuie să fie un dicționar"
        
        # Verificăm cheile EXACTE
        required_keys = {"metadata", "chunk"}
        chunk_keys = set(chunk_data.keys())
        
        # Verificăm chei lipsă
        missing_keys = required_keys - chunk_keys
        if missing_keys:
            return False, f"Lipsesc cheile obligatorii: {missing_keys}"
        
        # Verificăm chei în plus (opțional - warning, nu eroare)
        extra_keys = chunk_keys - required_keys
        if extra_keys:
            logger.warning(f"Chei suplimentare găsite în chunk: {extra_keys}")
        
        # Validăm metadata - TREBUIE să fie string
        metadata = chunk_data["metadata"]
        if not isinstance(metadata, str):
            return False, f"Metadata trebuie să fie string, nu {type(metadata).__name__}"
        
        metadata_stripped = metadata.strip()
        if len(metadata_stripped) == 0:
            return False, "Metadata nu poate fi string gol"
        
        if len(metadata_stripped) > 1000:
            return False, f"Metadata prea lungă ({len(metadata_stripped)} caractere, maximum 1000)"
        
        # Validăm conținutul - TREBUIE să fie string
        content = chunk_data["chunk"]
        if not isinstance(content, str):
            return False, f"Conținutul chunk-ului trebuie să fie string, nu {type(content).__name__}"
        
        content_stripped = content.strip()
        if len(content_stripped) < 10:
            return False, f"Conținutul chunk-ului este prea scurt ({len(content_stripped)} caractere, minimum 10)"
        
        if len(content_stripped) > 50000:
            logger.warning(f"Chunk foarte lung detectat: {len(content_stripped)} caractere")
        
        # Verificări de calitate suplimentare
        word_count = len(content_stripped.split())
        if word_count < 3:
            return False, f"Chunk-ul conține prea puține cuvinte ({word_count}, minimum 3)"
        
        # Verificăm dacă conținutul nu este doar punctuație
        alphanumeric_chars = sum(1 for c in content_stripped if c.isalnum())
        if alphanumeric_chars < 5:
            return False, "Chunk-ul trebuie să conțină cel puțin 5 caractere alfanumerice"
        
        return True, "Chunk valid cu structura exactă și conținut de calitate"
        
    except Exception as e:
        return False, f"Eroare la validarea chunk-ului: {str(e)}"

def clean_chunk_content(content: str) -> str:
    """
    Curăță și normalizează conținutul unui chunk cu algoritmi îmbunătățiți.
    
    Args:
        content: Conținutul de curățat
        
    Returns:
        Conținutul curățat și optimizat
    """
    if not isinstance(content, str):
        return ""
    
    # 1. Eliminăm spațiile în plus de la început și sfârșit
    content = content.strip()
    
    if not content:
        return ""
    
    # 2. Normalizăm line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    # 3. Eliminăm comentariile HTML și Markdown
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'<!-- image -->', '', content)
    
    # 4. Normalizăm spațiile multiple, dar păstrăm structura paragrafelor
    content = re.sub(r'[ \t]+', ' ', content)  # Spații și tab-uri multiple -> un spațiu
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Line breaks multiple -> două line breaks
    
    # 5. Eliminăm caracterele de control, dar păstrăm \n și \t
    content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\t')
    
    # 6. Normalizăm punctuația pentru consistență
    content = re.sub(r'[""''`]', '"', content)  # Uniformizăm ghilimelele
    content = re.sub(r'[–—]', '-', content)     # Uniformizăm liniuțele
    content = re.sub(r'…', '...', content)      # Uniformizăm ellipsis
    
    # 7. Eliminăm liniile care sunt doar punctuație sau whitespace
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        # Păstrăm linia dacă:
        # - Nu este goală
        # - Nu este doar punctuație
        # - Are cel puțin 2 caractere alfanumerice
        if (line_stripped and 
            not re.match(r'^[^\w]*"""
Utilitare OPTIMIZATE pentru procesarea fișierelor JSON chunkizate
Versiunea 3.0.0 - Îmbunătățiri pentru căutare și procesare
"""

import json
import os
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)

def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează că fișierul JSON are formatul EXACT specificat cu verificări îmbunătățite.
    
    Formatul acceptat:
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
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Fișierul este gol", 0
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return False, f"Fișierul este prea mare ({file_size // 1024 // 1024}MB). Maximum 100MB.", 0
        
        # Încercăm să încărcăm JSON-ul
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"JSON invalid: {str(e)[:100]}...", 0
        
        # Verificăm că este un dicționar
        if not isinstance(data, dict):
            return False, "JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip", 0
        
        if len(data) == 0:
            return False, "JSON-ul nu conține date", 0
        
        # Căutăm chunk-uri în formatul chunk_X
        chunk_pattern = re.compile(r'^chunk_\d+$')
        chunk_keys = [key for key in data.keys() if chunk_pattern.match(key)]
        
        if len(chunk_keys) == 0:
            # Încercăm să găsim alte patterns comune
            other_keys = list(data.keys())[:10]  # Primele 10 chei pentru debugging
            return False, f"Nu s-au găsit chunk-uri în formatul așteptat (chunk_0, chunk_1, etc.). Chei găsite: {other_keys}", 0
        
        # Validăm structura chunk-urilor - verificăm mai multe pentru siguranță
        valid_chunks = 0
        invalid_chunks = []
        sample_content_lengths = []
        
        for key in chunk_keys[:min(10, len(chunk_keys))]:  # Verificăm până la 10 chunk-uri
            chunk = data[key]
            
            # Verificăm că chunk-ul este un dicționar
            if not isinstance(chunk, dict):
                invalid_chunks.append(f"{key}: nu este dicționar")
                continue
            
            # Verificăm că are EXACT cheile: "metadata" și "chunk"
            required_keys = {"metadata", "chunk"}
            chunk_keys_set = set(chunk.keys())
            
            if not required_keys.issubset(chunk_keys_set):
                missing_keys = required_keys - chunk_keys_set
                invalid_chunks.append(f"{key}: lipsesc cheile {missing_keys}")
                continue
            
            # Verificăm că metadata este STRING
            if not isinstance(chunk["metadata"], str):
                invalid_chunks.append(f"{key}: metadata nu este string (este {type(chunk['metadata']).__name__})")
                continue
            
            if len(chunk["metadata"].strip()) == 0:
                invalid_chunks.append(f"{key}: metadata este string gol")
                continue
            
            # Verificăm că chunk este STRING cu conținut suficient
            if not isinstance(chunk["chunk"], str):
                invalid_chunks.append(f"{key}: chunk nu este string (este {type(chunk['chunk']).__name__})")
                continue
            
            content = chunk["chunk"].strip()
            if len(content) < 10:
                invalid_chunks.append(f"{key}: conținut prea scurt ({len(content)} caractere)")
                continue
            
            sample_content_lengths.append(len(content))
            valid_chunks += 1
        
        if valid_chunks == 0:
            error_details = "; ".join(invalid_chunks[:5])  # Primele 5 erori
            return False, f"Nu s-au găsit chunk-uri valide. Erori: {error_details}", 0
        
        # Verificăm consistența numerotării chunk-urilor
        chunk_numbers = []
        for key in chunk_keys:
            match = re.match(r'chunk_(\d+)', key)
            if match:
                chunk_numbers.append(int(match.group(1)))
        
        chunk_numbers.sort()
        expected_sequence = list(range(len(chunk_numbers)))
        
        # Warning pentru numerotare inconsistentă (nu blochez validarea)
        if chunk_numbers != expected_sequence:
            logger.warning(f"Numerotarea chunk-urilor nu este consecutivă: găsite {chunk_numbers[:10]}..., așteptate {expected_sequence[:10]}...")
        
        # Statistici pentru logging
        avg_length = sum(sample_content_lengths) / len(sample_content_lengths) if sample_content_lengths else 0
        
        logger.info(f"✅ JSON valid găsit cu {len(chunk_keys)} chunk-uri")
        logger.info(f"📊 Validare: {valid_chunks}/{min(10, len(chunk_keys))} chunk-uri verificate cu succes")
        logger.info(f"📏 Lungime medie conținut: {avg_length:.0f} caractere")
        
        return True, f"Valid - {len(chunk_keys)} chunk-uri găsite", len(chunk_keys)
        
    except Exception as e:
        logger.error(f"Eroare la validarea JSON: {str(e)}")
        return False, f"Eroare neașteptată la validare: {str(e)}", 0

def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează fișierul JSON cu îmbunătățiri pentru căutare și indexare optimizată.
    
    Returns:
        Lista de dicționare cu chunk-uri procesate și îmbunătățite
    """
    try:
        # Validăm mai întâi formatul
        is_vali, line_stripped) and
            sum(1 for c in line_stripped if c.isalnum()) >= 2):
            cleaned_lines.append(line)
        elif line_stripped == '':
            # Păstrăm liniile goale pentru separarea paragrafelor
            cleaned_lines.append('')
    
    content = '\n'.join(cleaned_lines)
    
    # 8. Eliminăm paragrafele goale multiple
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # 9. Curățăm din nou spațiile de la început și sfârșit
    content = content.strip()
    
    # 10. Verificare finală de calitate
    if len(content) < 10:
        logger.warning(f"Conținut foarte scurt după curățare: '{content[:50]}...'")
    
    return content

def extract_metadata_info(metadata_string: str) -> Dict[str, Any]:
    """
    Extrage informații îmbunătățite din metadata string și le convertește în dict.
    
    Args:
        metadata_string: Metadata originală ca string
        
    Returns:
        Metadata convertită în dict cu informații îmbunătățite
    """
    if not isinstance(metadata_string, str):
        return {"original_source": "Necunoscut", "source": "Necunoscut"}
    
    metadata_dict = {
        "original_source": metadata_string,
        "source": metadata_string,
        "extracted_info": {}
    }
    
    # Pattern matching îmbunătățit pentru diferite formate de metadata
    
    # 1. Format "Source: filename"
    source_match = re.search(r'Source:\s*(.+?)(?:\s*$|\s*\||\s*,)', metadata_string, re.IGNORECASE)
    if source_match:
        document_name = source_match.group(1).strip()
        metadata_dict["document_name"] = document_name
        metadata_dict["source"] = document_name
        metadata_dict["extracted_info"]["document_name"] = document_name
    
    # 2. Detectăm tipul de fișier
    file_extensions = re.findall(r'\.([a-zA-Z0-9]+)', metadata_string)
    if file_extensions:
        metadata_dict["file_type"] = file_extensions[-1].lower()  # Ultima extensie găsită
        metadata_dict["extracted_info"]["file_extensions"] = file_extensions
    
    # 3. Detectăm date
    date_patterns = [
        r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',  # YYYY-MM-DD sau YYYY/MM/DD
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',  # MM-DD-YYYY sau MM/DD/YYYY
        r'\b([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})\b',  # Mar 15, 2024
        r'\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b'   # 15 March 2024
    ]
    
    for pattern in date_patterns:
        dates = re.findall(pattern, metadata_string)
        if dates:
            metadata_dict["extracted_info"]["dates_found"] = dates
            metadata_dict["date_context"] = dates[0]  # Prima dată găsită
            break
    
    # 4. Detectăm autori sau nume
    author_patterns = [
        r'(?:by|author|written by|created by):\s*([A-Za-z\s]+?)(?:\s*$|\s*\||\s*,)',
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'  # Nume proprii (FirstName LastName)
    ]
    
    for pattern in author_patterns:
        authors = re.findall(pattern, metadata_string, re.IGNORECASE)
        if authors:
            metadata_dict["extracted_info"]["potential_authors"] = authors
            metadata_dict["author"] = authors[0]
            break
    
    # 5. Detectăm versiuni sau numere de capitol
    version_patterns = [
        r'\b[vV](\d+(?:\.\d+)*)\b',  # v1.0, V2.1.3
        r'\b(?:version|ver)[\s:]*(\d+(?:\.\d+)*)\b',  # version 1.0, ver: 2.1
        r'\b(?:chapter|ch|cap)[\s:]*(\d+)\b',  # chapter 1, ch: 5
        r'\b(?:section|sec)[\s:]*(\d+(?:\.\d+)*)\b'  # section 1.2
    ]
    
    for pattern in version_patterns:
        versions = re.findall(pattern, metadata_string, re.IGNORECASE)
        if versions:
            metadata_dict["extracted_info"]["versions_or_chapters"] = versions
            metadata_dict["version"] = versions[0]
            break
    
    # 6. Detectăm categorii sau tag-uri
    if '|' in metadata_string or ',' in metadata_string:
        # Probabil conține tag-uri separate prin | sau ,
        separators = ['|', ',']
        for sep in separators:
            if sep in metadata_string:
                parts = [part.strip() for part in metadata_string.split(sep)]
                if len(parts) > 1:
                    metadata_dict["extracted_info"]["tags"] = parts
                    break
    
    # 7. Detectăm URL-uri sau path-uri
    url_patterns = [
        r'https?://[^\s]+',
        r'www\.[^\s]+',
        r'[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+(?:\.[a-zA-Z]{2,})?'
    ]
    
    for pattern in url_patterns:
        urls = re.findall(pattern, metadata_string)
        if urls:
            metadata_dict["extracted_info"]["urls_or_domains"] = urls
            metadata_dict["domain"] = urls[0]
            break
    
    # 8. Calculăm un scor de completitudine pentru metadata
    info_count = len([v for v in metadata_dict["extracted_info"].values() if v])
    metadata_dict["metadata_richness"] = min(1.0, info_count / 5)  # Normalizat la 0-1
    
    # 9. Adăugăm sugestii pentru îmbunătățirea metadata
    suggestions = []
    if "document_name" not in metadata_dict:
        suggestions.append("Adăugați numele documentului")
    if "date_context" not in metadata_dict:
        suggestions.append("Adăugați informații despre dată")
    if "author" not in metadata_dict:
        suggestions.append("Adăugați informații despre autor")
    
    if suggestions:
        metadata_dict["improvement_suggestions"] = suggestions
    
    return metadata_dict

# Funcții helper pentru debugging și testing - ÎMBUNĂTĂȚITE

def test_json_file(file_path: str) -> None:
    """
    Testează complet un fișier JSON cu analiza îmbunătățită și afișează rezultatele detaliate.
    """
    print(f"\n🔍 TESTARE COMPLETĂ FIȘIER: {file_path}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Test validare
    print("📋 VALIDARE FORMAT JSON...")
    is_valid, error_msg, chunks_count = validate_json_format(file_path)
    print(f"✅ Validare: {'VALID' if is_valid else 'INVALID'}")
    if not is_valid:
        print(f"❌ Eroare: {error_msg}")
        return
    
    print(f"📊 Chunk-uri detectate: {chunks_count}")
    
    # Test statistici îmbunătățite
    print("\n📈 STATISTICI DETALIATE...")
    stats = get_json_statistics(file_path)
    if 'error' not in stats:
        print(f"📄 Informații fișier:")
        print(f"   - Nume: {stats['file_info']['file_name']}")
        print(f"   - Dimensiune: {stats['file_info']['file_size_mb']} MB")
        
        print(f"🧩 Analiză chunk-uri:")
        ca = stats['chunk_analysis']
        print(f"   - Total: {ca['total_chunks']}")
        print(f"   - Valide: {ca['valid_chunks']}")
        print(f"   - Numerotare consistentă: {'DA' if ca['chunk_numbering_consistent'] else 'NU'}")
        
        print(f"📝 Statistici conținut:")
        cs = stats['content_statistics']
        print(f"   - Total cuvinte: {cs['total_words']:,}")
        print(f"   - Lungime medie chunk: {cs['average_chunk_length']} caractere")
        print(f"   - Cuvinte medii per chunk: {cs['average_words_per_chunk']}")
        print(f"   - Scor calitate medie: {cs['average_quality_score']}")
        
        print(f"🌐 Analiză metadata:")
        ma = stats['metadata_analysis']
        print(f"   - Surse unice: {ma['unique_metadata_sources']}")
        print(f"   - Limba dominantă: {ma['dominant_language']}")
        print(f"   - Distribuție limbi: {ma['language_distribution']}")
        
        if stats['processing_recommendations']:
            print(f"💡 Recomandări:")
            for rec in stats['processing_recommendations']:
                print(f"   - {rec}")
    
    # Test previzualizare îmbunătățită
    print("\n👁️ PREVIZUALIZARE ÎMBUNĂTĂȚITĂ...")
    preview = preview_json_chunks(file_path, 2)
    if 'error' not in preview:
        pa = preview['preview_analysis']
        print(f"📊 Analiză previzualizare:")
        print(f"   - Calitate medie: {pa['average_quality']}")
        print(f"   - Limba dominantă: {pa['dominant_language']}")
        print(f"   - Total cuvinte preview: {pa['total_words_preview']}")
        
        print(f"🔍 Chunk-uri previzualizate:")
        for i, chunk in enumerate(preview['preview_chunks'], 1):
            print(f"   📄 {chunk['chunk_id']}:")
            print(f"      - Cuvinte: {chunk['word_count']}")
            print(f"      - Calitate: {chunk['quality_score']}")
            print(f"      - Limbă: {chunk['language_detected']}")
            print(f"      - Keywords: {', '.join(chunk['keywords'])}")
            if chunk['processing_suggestions']:
                print(f"      - Sugestii: {'; '.join(chunk['processing_suggestions'])}")
    
    # Test procesare
    print("\n⚙️ TEST PROCESARE OPTIMIZATĂ...")
    try:
        chunks_data = process_json_chunks(file_path)
        print(f"✅ Procesare: {len(chunks_data)} chunk-uri procesate cu succes")
        
        # Analizăm calitatea chunk-urilor procesate
        quality_scores = [chunk['quality_score'] for chunk in chunks_data]
        avg_quality = sum(quality_scores) / len(quality_scores)
        high_quality = len([s for s in quality_scores if s > 0.7])
        
        print(f"📊 Rezultate procesare:")
        print(f"   - Calitate medie: {avg_quality:.3f}")
        print(f"   - Chunk-uri de calitate înaltă: {high_quality}/{len(chunks_data)}")
        
        # Sample din primul chunk procesat
        if chunks_data:
            sample_chunk = chunks_data[0]
            metadata = sample_chunk['metadata']
            print(f"📋 Sample metadata (primul chunk):")
            print(f"   - Keywords: {metadata.get('keywords', 'N/A')[:100]}...")
            print(f"   - Limba detectată: {metadata.get('language_detected', 'N/A')}")
            print(f"   - Complexitate citire: {metadata.get('reading_complexity', 'N/A')}")
            print(f"   - Elemente structurale: {metadata.get('structural_elements', 0)}")
            
    except Exception as e:
        print(f"❌ Eroare la procesare: {str(e)}")
    
    processing_time = time.time() - start_time
    print(f"\n⏱️ TIMP TOTAL PROCESARE: {processing_time:.2f} secunde")
    print("=" * 70)

# Export pentru compatibilitate
__all__ = [
    'validate_json_format',
    'process_json_chunks', 
    'get_json_statistics',
    'preview_json_chunks',
    'validate_chunk_structure',
    'clean_chunk_content',
    'extract_metadata_info',
    'test_json_file',
    'clean_and_normalize_content',
    'extract_advanced_keywords',
    'detect_simple_language',
    'calculate_content_statistics',
    'extract_important_terms',
    'create_content_hash',
    'calculate_chunk_quality_score'
]"""
Utilitare OPTIMIZATE pentru procesarea fișierelor JSON chunkizate
Versiunea 3.0.0 - Îmbunătățiri pentru căutare și procesare
"""

import json
import os
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)

def validate_json_format(file_path: str) -> Tuple[bool, str, int]:
    """
    Validează că fișierul JSON are formatul EXACT specificat cu verificări îmbunătățite.
    
    Formatul acceptat:
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
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Fișierul este gol", 0
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return False, f"Fișierul este prea mare ({file_size // 1024 // 1024}MB). Maximum 100MB.", 0
        
        # Încercăm să încărcăm JSON-ul
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"JSON invalid: {str(e)[:100]}...", 0
        
        # Verificăm că este un dicționar
        if not isinstance(data, dict):
            return False, "JSON-ul trebuie să fie un obiect (dicționar), nu o listă sau alt tip", 0
        
        if len(data) == 0:
            return False, "JSON-ul nu conține date", 0
        
        # Căutăm chunk-uri în formatul chunk_X
        chunk_pattern = re.compile(r'^chunk_\d+$')
        chunk_keys = [key for key in data.keys() if chunk_pattern.match(key)]
        
        if len(chunk_keys) == 0:
            # Încercăm să găsim alte patterns comune
            other_keys = list(data.keys())[:10]  # Primele 10 chei pentru debugging
            return False, f"Nu s-au găsit chunk-uri în formatul așteptat (chunk_0, chunk_1, etc.). Chei găsite: {other_keys}", 0
        
        # Validăm structura chunk-urilor - verificăm mai multe pentru siguranță
        valid_chunks = 0
        invalid_chunks = []
        sample_content_lengths = []
        
        for key in chunk_keys[:min(10, len(chunk_keys))]:  # Verificăm până la 10 chunk-uri
            chunk = data[key]
            
            # Verificăm că chunk-ul este un dicționar
            if not isinstance(chunk, dict):
                invalid_chunks.append(f"{key}: nu este dicționar")
                continue
            
            # Verificăm că are EXACT cheile: "metadata" și "chunk"
            required_keys = {"metadata", "chunk"}
            chunk_keys_set = set(chunk.keys())
            
            if not required_keys.issubset(chunk_keys_set):
                missing_keys = required_keys - chunk_keys_set
                invalid_chunks.append(f"{key}: lipsesc cheile {missing_keys}")
                continue
            
            # Verificăm că metadata este STRING
            if not isinstance(chunk["metadata"], str):
                invalid_chunks.append(f"{key}: metadata nu este string (este {type(chunk['metadata']).__name__})")
                continue
            
            if len(chunk["metadata"].strip()) == 0:
                invalid_chunks.append(f"{key}: metadata este string gol")
                continue
            
            # Verificăm că chunk este STRING cu conținut suficient
            if not isinstance(chunk["chunk"], str):
                invalid_chunks.append(f"{key}: chunk nu este string (este {type(chunk['chunk']).__name__})")
                continue
            
            content = chunk["chunk"].strip()
            if len(content) < 10:
                invalid_chunks.append(f"{key}: conținut prea scurt ({len(content)} caractere)")
                continue
            
            sample_content_lengths.append(len(content))
            valid_chunks += 1
        
        if valid_chunks == 0:
            error_details = "; ".join(invalid_chunks[:5])  # Primele 5 erori
            return False, f"Nu s-au găsit chunk-uri valide. Erori: {error_details}", 0
        
        # Verificăm consistența numerotării chunk-urilor
        chunk_numbers = []
        for key in chunk_keys:
            match = re.match(r'chunk_(\d+)', key)
            if match:
                chunk_numbers.append(int(match.group(1)))
        
        chunk_numbers.sort()
        expected_sequence = list(range(len(chunk_numbers)))
        
        # Warning pentru numerotare inconsistentă (nu blochez validarea)
        if chunk_numbers != expected_sequence:
            logger.warning(f"Numerotarea chunk-urilor nu este consecutivă: găsite {chunk_numbers[:10]}..., așteptate {expected_sequence[:10]}...")
        
        # Statistici pentru logging
        avg_length = sum(sample_content_lengths) / len(sample_content_lengths) if sample_content_lengths else 0
        
        logger.info(f"✅ JSON valid găsit cu {len(chunk_keys)} chunk-uri")
        logger.info(f"📊 Validare: {valid_chunks}/{min(10, len(chunk_keys))} chunk-uri verificate cu succes")
        logger.info(f"📏 Lungime medie conținut: {avg_length:.0f} caractere")
        
        return True, f"Valid - {len(chunk_keys)} chunk-uri găsite", len(chunk_keys)
        
    except Exception as e:
        logger.error(f"Eroare la validarea JSON: {str(e)}")
        return False, f"Eroare neașteptată la validare: {str(e)}", 0

def process_json_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Procesează fișierul JSON cu îmbunătățiri pentru căutare și indexare optimizată.
    
    Returns:
        Lista de dicționare cu chunk-uri procesate și îmbunătățite
    """
    try:
        # Validăm mai întâi formatul
        is_vali