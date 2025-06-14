from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, validator, root_validator
import re
from datetime import datetime
from enum import Enum


class SearchMethodEnum(str, Enum):
    """Metodele de căutare disponibile."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class LanguageEnum(str, Enum):
    """Limbile suportate pentru răspunsuri."""
    ROMANIAN = "ro"
    ENGLISH = "en"
    AUTO = "auto"


class QualityLevelEnum(str, Enum):
    """Nivelurile de calitate pentru chunk-uri."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXCELLENT = "excellent"


class CollectionCreate(BaseModel):
    """Model optimizat pentru crearea unei colecții noi."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'  # Nu permite câmpuri suplimentare
    )
    
    name: str = Field(
        ..., 
        min_length=1, 
        max_length=50,
        description="Numele colecției (doar litere, cifre și underscore)"
    )
    description: Optional[str] = Field(
        None, 
        max_length=500,
        description="Descrierea opțională a colecției"
    )
    enable_hybrid_search: bool = Field(
        default=True,
        description="Activează căutarea hibridă pentru această colecție"
    )
    default_language: LanguageEnum = Field(
        default=LanguageEnum.AUTO,
        description="Limba implicită pentru colecție"
    )
    
    @validator('name')
    def validate_collection_name(cls, v):
        """Validează numele colecției cu reguli îmbunătățite."""
        if not v:
            raise ValueError('Numele colecției nu poate fi gol')
        
        # Verifică doar litere, cifre și underscore
        if not re.match(r'^[a-zA-Z0-9_]+, v):
            raise ValueError('Numele colecției poate conține doar litere, cifre și underscore (_)')
        
        # Nu poate începe cu cifră
        if v[0].isdigit():
            raise ValueError('Numele colecției nu poate începe cu o cifră')
        
        # Nu poate începe sau termina cu underscore
        if v.startswith('_') or v.endswith('_'):
            raise ValueError('Numele colecției nu poate începe sau termina cu underscore')
        
        # Nu poate avea underscore consecutive
        if '__' in v:
            raise ValueError('Numele colecției nu poate avea underscore consecutive')
        
        # Nume rezervate extinse
        reserved_names = {
            'admin', 'system', 'config', 'test', 'default', 'null', 'undefined',
            'api', 'health', 'docs', 'cache', 'temp', 'tmp', 'root', 'user',
            'database', 'db', 'collection', 'document', 'chunk', 'index'
        }
        if v.lower() in reserved_names:
            raise ValueError(f'Numele "{v}" este rezervat și nu poate fi folosit')
        
        return v.lower()  # Normalizăm la lowercase
    
    @validator('description')
    def validate_description(cls, v):
        """Validează descrierea colecției."""
        if v is not None:
            v = v.strip()
            if v == '':
                return None  # Convertim string-uri goale la None
            
            # Verificăm că nu conține doar spații sau caractere speciale
            if re.match(r'^[\s\-_\.,;:!?]*, v):
                raise ValueError('Descrierea trebuie să conțină text valid')
        
        return v


class DocumentDeleteRequest(BaseModel):
    """Model optimizat pentru cererea de ștergere a documentelor."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        extra='forbid'
    )
    
    document_ids: Optional[List[str]] = Field(
        None,
        description="Lista ID-urilor documentelor de șters"
    )
    source: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Sursa fișierului de șters (alternativă la document_ids)"
    )
    force_delete: bool = Field(
        default=False,
        description="Forțează ștergerea chiar dacă există dependențe"
    )
    clear_cache: bool = Field(
        default=True,
        description="Curăță cache-ul după ștergere"
    )
    
    @root_validator
    def validate_delete_request(cls, values):
        """Validează că cel puțin unul dintre parametri este specificat."""
        document_ids = values.get('document_ids')
        source = values.get('source')
        
        if not document_ids and not source:
            raise ValueError('Trebuie specificat fie document_ids fie source')
        
        if document_ids and source:
            raise ValueError('Nu se pot specifica ambii parametri document_ids și source')
        
        return values
    
    @validator('document_ids')
    def validate_document_ids(cls, v):
        """Validează lista de ID-uri de documente."""
        if v is not None:
            if len(v) == 0:
                raise ValueError('Lista document_ids nu poate fi goală')
            
            if len(v) > 100:
                raise ValueError('Nu se pot șterge mai mult de 100 de documente odată')
            
            # Verifică că ID-urile nu sunt goale și au format valid
            for doc_id in v:
                if not doc_id or not doc_id.strip():
                    raise ValueError('ID-urile documentelor nu pot fi goale')
                
                # Verificăm lungimea ID-ului
                if len(doc_id.strip()) > 255:
                    raise ValueError('ID-ul documentului este prea lung (maximum 255 caractere)')
        
        return v


class GenerateRequest(BaseModel):
    """Model optimizat pentru cererea de generare a răspunsurilor."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    query: str = Field(
        ..., 
        min_length=3, 
        max_length=2000,
        description="Întrebarea utilizatorului"
    )
    top_k_docs: Optional[int] = Field(
        10,  # Creștem default de la 5 la 10
        ge=1, 
        le=20,
        description="Numărul de documente relevante de recuperat"
    )
    temperature: Optional[float] = Field(
        0.3,  # Creștem de la 0.2 la 0.3
        ge=0.0, 
        le=1.0,
        description="Temperatura pentru generarea răspunsului (0.0 = conservator, 1.0 = creativ)"
    )
    max_output_tokens: Optional[int] = Field(
        1500,  # Creștem de la 1024 la 1500
        ge=100,
        le=3000,  # Creștem limita maximă
        description="Numărul maxim de tokeni pentru răspuns"
    )
    use_hybrid_search: Optional[bool] = Field(
        True,  # Activăm hybrid search implicit
        description="Activează căutarea hibridă (semantic + keyword)"
    )
    similarity_threshold: Optional[float] = Field(
        0.15,  # Threshold mai permisiv
        ge=0.0,
        le=1.0,
        description="Pragul minim de similaritate pentru rezultate"
    )
    include_sources: Optional[bool] = Field(
        True,
        description="Include sursele documentelor în răspuns"
    )
    language: Optional[LanguageEnum] = Field(
        LanguageEnum.ROMANIAN,
        description="Limba pentru răspuns"
    )
    search_method: Optional[SearchMethodEnum] = Field(
        SearchMethodEnum.HYBRID,
        description="Metoda de căutare preferată"
    )
    enable_query_expansion: Optional[bool] = Field(
        True,
        description="Activează expandarea automată a termenilor din query"
    )
    quality_filter: Optional[QualityLevelEnum] = Field(
        None,
        description="Filtrează rezultatele după nivel minim de calitate"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validează interogarea utilizatorului cu verificări îmbunătățite."""
        if not v or not v.strip():
            raise ValueError('Interogarea nu poate fi goală')
        
        # Elimină spațiile multiple și caracterele de control
        cleaned_query = re.sub(r'\s+', ' ', v.strip())
        
        # Verifică că nu este doar punctuație
        alphanumeric_chars = sum(1 for c in cleaned_query if c.isalnum())
        if alphanumeric_chars < 2:
            raise ValueError('Interogarea trebuie să conțină cel puțin 2 caractere alfanumerice')
        
        # Verifică că nu este spam (același caracter repetat)
        if len(set(cleaned_query.replace(' ', ''))) < 3:
            raise ValueError('Interogarea pare să fie spam sau invalidă')
        
        # Verifică lungimea cuvintelor
        words = cleaned_query.split()
        if len(words) == 1 and len(words[0]) < 3:
            raise ValueError('Interogarea este prea scurtă. Folosiți cel puțin 3 caractere sau mai multe cuvinte')
        
        return cleaned_query
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """Validează temperatura pentru generare."""
        if v is not None:
            # Rotunjim la 2 zecimale pentru consistență
            return round(v, 2)
        return v
    
    @validator('top_k_docs')
    def validate_top_k_docs(cls, v):
        """Validează numărul de documente de recuperat."""
        if v is not None and v <= 0:
            raise ValueError('top_k_docs trebuie să fie un număr pozitiv')
        return v
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        """Validează pragul de similaritate."""
        if v is not None:
            if v < 0.0 or v > 1.0:
                raise ValueError('similarity_threshold trebuie să fie între 0.0 și 1.0')
            return round(v, 3)
        return v
    
    @root_validator
    def validate_request_consistency(cls, values):
        """Validări globale pentru consistența cererii."""
        temperature = values.get('temperature', 0.3)
        max_tokens = values.get('max_output_tokens', 1500)
        top_k = values.get('top_k_docs', 10)
        
        # Pentru temperaturi mari, sugerăm mai mulți tokeni
        if temperature > 0.7 and max_tokens < 1000:
            values['max_output_tokens'] = 1000
        
        # Pentru query-uri complexe, sugerăm mai multe documente
        query_length = len(values.get('query', ''))
        if query_length > 500 and top_k < 15:
            values['top_k_docs'] = 15
        
        return values


class QueryResponse(BaseModel):
    """Model optimizat pentru răspunsul la o interogare."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    query: str = Field(..., description="Interogarea originală")
    answer: str = Field(..., description="Răspunsul generat")
    documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Documentele folosite pentru generarea răspunsului"
    )
    query_time_ms: Optional[int] = Field(
        None,
        description="Timpul de procesare în milisecunde"
    )
    model_used: Optional[str] = Field(
        None,
        description="Modelul AI folosit pentru generare"
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Scorul de încredere pentru răspuns"
    )
    search_method: Optional[SearchMethodEnum] = Field(
        None,
        description="Metoda de căutare folosită"
    )
    total_chunks_searched: Optional[int] = Field(
        None,
        ge=0,
        description="Numărul total de chunk-uri căutate"
    )
    relevance_scores: Optional[List[float]] = Field(
        None,
        description="Scorurile de relevanță pentru documentele returnate"
    )
    query_expansion_terms: Optional[List[str]] = Field(
        None,
        description="Termenii adăugați prin expandarea query-ului"
    )
    processing_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata suplimentară despre procesare"
    )


class DocumentInfo(BaseModel):
    """Model optimizat pentru informațiile despre un document."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Optional[str] = Field(None, description="ID-ul unic al documentului")
    source: str = Field(..., description="Sursa documentului")
    content_length: int = Field(..., ge=0, description="Lungimea conținutului")
    chunk_count: int = Field(..., ge=1, description="Numărul de chunk-uri")
    created_at: str = Field(..., description="Data creării")
    file_type: str = Field(default="json_chunked", description="Tipul fișierului")
    language_detected: Optional[str] = Field(None, description="Limba detectată")
    quality_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Scorul de calitate al documentului"
    )
    keywords_count: Optional[int] = Field(
        None, 
        ge=0, 
        description="Numărul de keywords unice"
    )
    word_count: Optional[int] = Field(
        None, 
        ge=0, 
        description="Numărul total de cuvinte"
    )
    processing_version: Optional[str] = Field(
        None,
        description="Versiunea algoritmului de procesare folosit"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadate suplimentare"
    )


class CollectionInfo(BaseModel):
    """Model optimizat pentru informațiile despre o colecție."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(..., description="Numele colecției")
    description: Optional[str] = Field(None, description="Descrierea colecției")
    document_count: int = Field(default=0, ge=0, description="Numărul de documente")
    total_chunks: int = Field(default=0, ge=0, description="Numărul total de chunk-uri")
    created_at: str = Field(..., description="Data creării")
    last_updated: Optional[str] = Field(None, description="Ultima actualizare")
    size_bytes: Optional[int] = Field(None, ge=0, description="Dimensiunea în bytes")
    hybrid_search_enabled: bool = Field(
        default=True, 
        description="Dacă căutarea hibridă este activă"
    )
    default_language: Optional[LanguageEnum] = Field(
        None,
        description="Limba implicită a colecției"
    )
    quality_distribution: Optional[Dict[str, int]] = Field(
        None,
        description="Distribuția calității chunk-urilor"
    )
    language_distribution: Optional[Dict[str, int]] = Field(
        None,
        description="Distribuția limbilor în colecție"
    )
    optimization_level: Optional[str] = Field(
        None,
        description="Nivelul de optimizare al colecției"
    )


class SearchRequest(BaseModel):
    """Model pentru cererile de căutare fără generare de răspuns."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        extra='forbid'
    )
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Query-ul de căutare"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Numărul de rezultate de returnat"
    )
    search_method: SearchMethodEnum = Field(
        default=SearchMethodEnum.HYBRID,
        description="Metoda de căutare"
    )
    similarity_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Pragul minim de similaritate"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadatele în rezultate"
    )
    include_scores: bool = Field(
        default=True,
        description="Include scorurile de similaritate"
    )


class HealthCheckResponse(BaseModel):
    """Model optimizat pentru răspunsul de health check."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    status: str = Field(..., description="Statusul aplicației")
    timestamp: str = Field(..., description="Timestamp-ul verificării")
    version: Optional[str] = Field(None, description="Versiunea aplicației")
    database_status: str = Field(..., description="Statusul bazei de date")
    ai_model_status: str = Field(..., description="Statusul modelului AI")
    uptime_seconds: Optional[int] = Field(None, ge=0, description="Timpul de funcționare")
    memory_usage: Optional[Dict[str, Any]] = Field(
        None,
        description="Informații despre utilizarea memoriei"
    )
    cache_status: Optional[Dict[str, Any]] = Field(
        None,
        description="Statusul cache-ului"
    )
    optimization_features: Optional[List[str]] = Field(
        None,
        description="Lista optimizărilor active"
    )
    performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Metrici de performanță"
    )


class ErrorResponse(BaseModel):
    """Model optimizat pentru răspunsurile de eroare."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    error: str = Field(..., description="Tipul erorii")
    message: str = Field(..., description="Mesajul de eroare")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Detalii suplimentare despre eroare"
    )
    timestamp: str = Field(..., description="Timestamp-ul erorii")
    request_id: Optional[str] = Field(None, description="ID-ul cererii")
    error_code: Optional[str] = Field(None, description="Codul de eroare")
    suggestions: Optional[List[str]] = Field(
        None,
        description="Sugestii pentru rezolvarea erorii"
    )
    documentation_link: Optional[str] = Field(
        None,
        description="Link către documentație pentru această eroare"
    )


class CacheInfo(BaseModel):
    """Model pentru informațiile despre cache."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    current_size: int = Field(..., ge=0, description="Dimensiunea actuală a cache-ului")
    max_size: int = Field(..., ge=0, description="Dimensiunea maximă a cache-ului")
    hit_rate: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Rata de hit a cache-ului"
    )
    total_hits: Optional[int] = Field(None, ge=0, description="Total hit-uri")
    total_misses: Optional[int] = Field(None, ge=0, description="Total miss-uri")
    memory_usage_bytes: Optional[int] = Field(
        None, 
        ge=0, 
        description="Utilizarea memoriei în bytes"
    )
    oldest_entry_age: Optional[int] = Field(
        None, 
        ge=0, 
        description="Vârsta celei mai vechi intrări în secunde"
    )
    cache_efficiency: Optional[str] = Field(
        None,
        description="Evaluarea eficienței cache-ului"
    )


class ProcessingStats(BaseModel):
    """Model pentru statisticile de procesare."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    total_files_processed: int = Field(..., ge=0)
    total_chunks_created: int = Field(..., ge=0)
    total_processing_time: float = Field(..., ge=0.0)
    average_processing_time_per_file: float = Field(..., ge=0.0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    error_count: int = Field(..., ge=0)
    quality_distribution: Dict[str, int] = Field(...)
    language_distribution: Dict[str, int] = Field(...)
    optimization_metrics: Dict[str, float] = Field(...)


# Funcții helper pentru validare și creare de răspunsuri - OPTIMIZATE

def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validează extensia unui fișier cu verificări îmbunătățite."""
    if not filename:
        return False
    
    # Normalizăm numele fișierului
    filename = filename.strip().lower()
    
    # Verificăm că fișierul are extensie
    if '.' not in filename:
        return False
    
    extension = filename.split('.')[-1]
    
    # Verificăm că extensia nu este goală
    if not extension:
        return False
    
    # Normalizăm extensiile permise
    normalized_allowed = [ext.lower().lstrip('.') for ext in allowed_extensions]
    
    return extension in normalized_allowed


def sanitize_filename(filename: str) -> str:
    """Sanitizează numele unui fișier pentru siguranță cu algoritm îmbunătățit."""
    if not filename:
        return "unknown_file"
    
    # Eliminăm path-ul dacă există
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Eliminăm caracterele periculoase și înlocuim cu underscore
    dangerous_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(dangerous_chars, '_', filename)
    
    # Eliminăm underscore-urile multiple
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Eliminăm underscore de la început și sfârșit
    sanitized = sanitized.strip('_')
    
    # Dacă nu avem nimic valid, returnăm un nume default
    if not sanitized or sanitized == '_':
        return "unknown_file"
    
    # Limitează lungimea, păstrând extensia
    if len(sanitized) > 255:
        if '.' in sanitized:
            name, ext = sanitized.rsplit('.', 1)
            max_name_length = 255 - len(ext) - 1  # -1 pentru punct
            sanitized = name[:max_name_length] + '.' + ext
        else:
            sanitized = sanitized[:255]
    
    return sanitized


def create_success_response(
    data: Any, 
    message: str = "Success",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Creează un răspuns de succes standardizat și îmbunătățit."""
    response = {
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0"
    }
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def create_error_response(
    error: str, 
    message: str, 
    details: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None,
    error_code: Optional[str] = None
) -> ErrorResponse:
    """Creează un răspuns de eroare standardizat și îmbunătățit."""
    return ErrorResponse(
        error=error,
        message=message,
        details=details or {},
        timestamp=datetime.utcnow().isoformat(),
        error_code=error_code,
        suggestions=suggestions or [],
        documentation_link=None  # Poate fi setat la un link valid către documentație
    )


def validate_query_complexity(query: str) -> Dict[str, Any]:
    """
    Analizează complexitatea unei interogări și returnează metrici.
    
    Args:
        query: Interogarea de analizat
        
    Returns:
        Dicționar cu metrici de complexitate
    """
    if not query:
        return {
            "complexity_level": "invalid",
            "word_count": 0,
            "estimated_processing_time": 0,
            "recommended_top_k": 5
        }
    
    words = query.split()
    word_count = len(words)
    char_count = len(query)
    
    # Calculăm complexitatea pe baza mai multor factori
    complexity_score = 0
    
    # Factor 1: Lungimea
    if word_count > 20:
        complexity_score += 3
    elif word_count > 10:
        complexity_score += 2
    elif word_count > 5:
        complexity_score += 1
    
    # Factor 2: Cuvinte complexe (mai lungi de 6 caractere)
    complex_words = sum(1 for word in words if len(word) > 6)
    complexity_score += min(complex_words, 3)
    
    # Factor 3: Întrebări multiple (mai multe semne de întrebare)
    question_marks = query.count('?')
    if question_marks > 1:
        complexity_score += 2
    
    # Factor 4: Conjuncții și disjuncții
    logical_words = ['și', 'sau', 'dar', 'însă', 'and', 'or', 'but']
    logical_count = sum(1 for word in logical_words if word.lower() in query.lower())
    complexity_score += min(logical_count, 2)
    
    # Factor 5: Termeni tehnici
    technical_indicators = ['cum să', 'cum pot', 'ce este', 'de ce', 'când', 'unde']
    technical_count = sum(1 for indicator in technical_indicators if indicator in query.lower())
    complexity_score += technical_count
    
    # Determinăm nivelul de complexitate
    if complexity_score >= 8:
        complexity_level = "very_high"
        estimated_time = 5000  # 5 secunde
        recommended_top_k = 15
    elif complexity_score >= 6:
        complexity_level = "high"
        estimated_time = 3000  # 3 secunde
        recommended_top_k = 12
    elif complexity_score >= 4:
        complexity_level = "medium"
        estimated_time = 2000  # 2 secunde
        recommended_top_k = 10
    elif complexity_score >= 2:
        complexity_level = "low"
        estimated_time = 1000  # 1 secundă
        recommended_top_k = 8
    else:
        complexity_level = "very_low"
        estimated_time = 500   # 0.5 secunde
        recommended_top_k = 5
    
    return {
        "complexity_level": complexity_level,
        "complexity_score": complexity_score,
        "word_count": word_count,
        "char_count": char_count,
        "complex_words": complex_words,
        "logical_operators": logical_count,
        "technical_indicators": technical_count,
        "estimated_processing_time": estimated_time,
        "recommended_top_k": recommended_top_k,
        "recommended_temperature": 0.3 if complexity_score > 5 else 0.2
    }


def optimize_request_parameters(request: GenerateRequest) -> GenerateRequest:
    """
    Optimizează automat parametrii unei cereri pe baza complexității query-ului.
    
    Args:
        request: Cererea originală
        
    Returns:
        Cererea optimizată
    """
    complexity_info = validate_query_complexity(request.query)
    
    # Ajustăm parametrii pe baza complexității
    if complexity_info["complexity_level"] in ["high", "very_high"]:
        # Pentru query-uri complexe
        if request.top_k_docs < complexity_info["recommended_top_k"]:
            request.top_k_docs = complexity_info["recommended_top_k"]
        
        if request.max_output_tokens < 2000:
            request.max_output_tokens = 2000
        
        if request.temperature < 0.3:
            request.temperature = 0.3
        
        # Activăm căutarea hibridă pentru rezultate mai bune
        request.use_hybrid_search = True
        request.enable_query_expansion = True
        
    elif complexity_info["complexity_level"] == "very_low":
        # Pentru query-uri simple, optimizăm pentru viteză
        if request.top_k_docs > 8:
            request.top_k_docs = 8
        
        if request.max_output_tokens > 1000:
            request.max_output_tokens = 1000
    
    return request


# Export pentru compatibilitate
__all__ = [
    'SearchMethodEnum',
    'LanguageEnum', 
    'QualityLevelEnum',
    'CollectionCreate',
    'DocumentDeleteRequest',
    'GenerateRequest',
    'QueryResponse',
    'DocumentInfo',
    'CollectionInfo',
    'SearchRequest',
    'HealthCheckResponse',
    'ErrorResponse',
    'CacheInfo',
    'ProcessingStats',
    'validate_file_extension',
    'sanitize_filename',
    'create_success_response',
    'create_error_response',
    'validate_query_complexity',
    'optimize_request_parameters'
]