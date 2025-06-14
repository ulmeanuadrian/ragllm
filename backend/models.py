from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field, validator, root_validator
import re


class CollectionCreate(BaseModel):
    """Model pentru crearea unei colecții noi."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        validate_assignment=True
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
    
    @validator('name')
    def validate_collection_name(cls, v):
        """Validează numele colecției."""
        if not v:
            raise ValueError('Numele colecției nu poate fi gol')
        
        # Verifică doar litere, cifre și underscore
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Numele colecției poate conține doar litere, cifre și underscore (_)')
        
        # Nu poate începe cu cifră
        if v[0].isdigit():
            raise ValueError('Numele colecției nu poate începe cu o cifră')
        
        # Nume rezervate
        reserved_names = {'admin', 'system', 'config', 'test', 'default', 'null', 'undefined'}
        if v.lower() in reserved_names:
            raise ValueError(f'Numele "{v}" este rezervat și nu poate fi folosit')
        
        return v.lower()  # Normalizăm la lowercase
    
    @validator('description')
    def validate_description(cls, v):
        """Validează descrierea colecției."""
        if v is not None and v.strip() == '':
            return None  # Convertim string-uri goale la None
        return v


class DocumentDeleteRequest(BaseModel):
    """Model pentru cererea de ștergere a documentelor."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True
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
            
            # Verifică că ID-urile nu sunt goale
            for doc_id in v:
                if not doc_id or not doc_id.strip():
                    raise ValueError('ID-urile documentelor nu pot fi goale')
        
        return v


class GenerateRequest(BaseModel):
    """Model pentru cererea de generare a răspunsurilor."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    query: str = Field(
        ..., 
        min_length=3, 
        max_length=2000,
        description="Întrebarea utilizatorului"
    )
    top_k_docs: Optional[int] = Field(
        5, 
        ge=1, 
        le=20,
        description="Numărul de documente relevante de recuperat"
    )
    temperature: Optional[float] = Field(
        0.2, 
        ge=0.0, 
        le=1.0,
        description="Temperatura pentru generarea răspunsului (0.0 = conservator, 1.0 = creativ)"
    )
    max_output_tokens: Optional[int] = Field(
        1024,
        ge=100,
        le=2048,
        description="Numărul maxim de tokeni pentru răspuns"
    )
    use_web_search: Optional[bool] = Field(
        False,
        description="Activare căutare web (momentan neimplementat)"
    )
    include_sources: Optional[bool] = Field(
        True,
        description="Include sursele documentelor în răspuns"
    )
    language: Optional[str] = Field(
        "ro",
        regex=r'^(ro|en)$',
        description="Limba pentru răspuns (ro/en)"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validează interogarea utilizatorului."""
        if not v or not v.strip():
            raise ValueError('Interogarea nu poate fi goală')
        
        # Elimină spațiile multiple și caracterele de control
        cleaned_query = re.sub(r'\s+', ' ', v.strip())
        
        # Verifică că nu este doar punctuație
        if re.sub(r'[^\w\s]', '', cleaned_query).strip() == '':
            raise ValueError('Interogarea trebuie să conțină cel puțin un cuvânt')
        
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
    
    @root_validator
    def validate_request_consistency(cls, values):
        """Validări globale pentru consistența cererii."""
        temperature = values.get('temperature', 0.2)
        max_tokens = values.get('max_output_tokens', 1024)
        
        # Pentru temperaturi mari, sugerăm mai mulți tokeni
        if temperature > 0.7 and max_tokens < 512:
            values['max_output_tokens'] = 512
        
        return values


class QueryResponse(BaseModel):
    """Model pentru răspunsul la o interogare."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
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


class DocumentInfo(BaseModel):
    """Model pentru informațiile despre un document."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(..., description="ID-ul unic al documentului")
    source: str = Field(..., description="Sursa documentului")
    content_length: int = Field(..., ge=0, description="Lungimea conținutului")
    chunk_count: int = Field(..., ge=1, description="Numărul de chunk-uri")
    created_at: str = Field(..., description="Data creării")
    file_type: str = Field(default="json_chunked", description="Tipul fișierului")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadate suplimentare"
    )


class CollectionInfo(BaseModel):
    """Model pentru informațiile despre o colecție."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(..., description="Numele colecției")
    description: Optional[str] = Field(None, description="Descrierea colecției")
    document_count: int = Field(default=0, ge=0, description="Numărul de documente")
    total_chunks: int = Field(default=0, ge=0, description="Numărul total de chunk-uri")
    created_at: str = Field(..., description="Data creării")
    last_updated: Optional[str] = Field(None, description="Ultima actualizare")
    size_bytes: Optional[int] = Field(None, ge=0, description="Dimensiunea în bytes")


class HealthCheckResponse(BaseModel):
    """Model pentru răspunsul de health check."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    status: str = Field(..., description="Statusul aplicației")
    timestamp: str = Field(..., description="Timestamp-ul verificării")
    version: Optional[str] = Field(None, description="Versiunea aplicației")
    database_status: str = Field(..., description="Statusul bazei de date")
    ai_model_status: str = Field(..., description="Statusul modelului AI")
    uptime_seconds: Optional[int] = Field(None, ge=0, description="Timpul de funcționare")


class ErrorResponse(BaseModel):
    """Model pentru răspunsurile de eroare."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    error: str = Field(..., description="Tipul erorii")
    message: str = Field(..., description="Mesajul de eroare")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Detalii suplimentare despre eroare"
    )
    timestamp: str = Field(..., description="Timestamp-ul erorii")
    request_id: Optional[str] = Field(None, description="ID-ul cererii")


# Funcții helper pentru validare
def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validează extensia unui fișier."""
    if not filename:
        return False
    
    extension = filename.lower().split('.')[-1] if '.' in filename else ''
    return extension in [ext.lower().lstrip('.') for ext in allowed_extensions]


def sanitize_filename(filename: str) -> str:
    """Sanitizează numele unui fișier pentru siguranță."""
    if not filename:
        return "unknown_file"
    
    # Elimină caracterele periculoase
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limitează lungimea
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:250] + ('.' + ext if ext else '')
    
    return sanitized


def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Creează un răspuns de succes standardizat."""
    return {
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }


def create_error_response(error: str, message: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """Creează un răspuns de eroare standardizat."""
    return ErrorResponse(
        error=error,
        message=message,
        details=details or {},
        timestamp=__import__('datetime').datetime.utcnow().isoformat()
    )