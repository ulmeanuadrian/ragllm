import os
import time
import hashlib
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from collections import OrderedDict

# Configurare logger
logger = logging.getLogger("rag_api")

# Încărcăm variabilele de mediu
load_dotenv()

# Configurăm API key pentru Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY nu este setat în fișierul .env")

genai.configure(api_key=GOOGLE_API_KEY)

# Singleton pentru a evita crearea mai multor instanțe ale modelului
_gemini_model_instances = {}

# Cache pentru răspunsuri pentru a evita apeluri repetate pentru aceeași interogare
# Folosim OrderedDict pentru a păstra ordinea inserării și a putea elimina cele mai vechi intrări
from collections import OrderedDict
_response_cache = OrderedDict()
MAX_CACHE_SIZE = 100  # Limităm dimensiunea cache-ului pentru a evita consumul excesiv de memorie

class GeminiGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Inițializează generatorul Gemini.
        
        Args:
            model_name: Numele modelului Gemini de utilizat (implicit: gemini-2.5-flash)
        """
        self.model_name = model_name
        
        # Folosim modelul din cache dacă există deja
        global _gemini_model_instances
        if model_name not in _gemini_model_instances:
            print(f"Crearea unei noi instanțe pentru modelul {model_name}")
            _gemini_model_instances[model_name] = genai.GenerativeModel(model_name)
            self._is_initialized = False
        else:
            print(f"Folosirea instanței existente pentru modelul {model_name}")
            self._is_initialized = True
            
        self.model = _gemini_model_instances[model_name]
        
        if not self._is_initialized:
            self.initialize()
    
    def initialize(self):
        """Inițializează modelul pentru a fi pregătit de utilizare."""
        if not self._is_initialized:
            try:
                # Facem o interogare simplă pentru a încărca modelul
                _ = self.model.generate_content("Testare inițializare model")
                self._is_initialized = True
                print(f"Modelul {self.model_name} a fost inițializat cu succes.")
            except Exception as e:
                print(f"Eroare la inițializarea modelului: {e}")
                raise
    
    def _generate_cache_key(self, query: str, document_contents: List[str], temperature: float = 0.2) -> str:
        """
        Generează o cheie unică pentru cache bazată pe interogare și conținutul documentelor.
        
        Args:
            query: Interogarea utilizatorului
            document_contents: Lista conținuturilor documentelor
            temperature: Temperatura pentru generare
            
        Returns:
            Cheia pentru cache
        """
        # Normalizăm interogarea
        query_normalized = query.lower().strip()
        
        # Creăm un hash din interogare și conținutul documentelor
        hash_input = query_normalized
        
        # Adăugăm fingerprint-uri pentru conținutul documentelor
        for content in document_contents:
            # Folosim doar primele 100 de caractere pentru eficiență
            content_sample = content[:100].strip()
            # Adăugăm un mini-hash al conținutului
            content_hash = hashlib.md5(content_sample.encode()).hexdigest()[:8]
            hash_input += f"|{content_hash}"
        
        # Adăugăm temperatura pentru a diferenția răspunsurile generate cu temperaturi diferite
        hash_input += f"|temp:{temperature}"
        
        # Generăm hash-ul final
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]], 
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        top_k: int = 40,
        top_p: float = 0.95,
        use_web_search: bool = False,
    ) -> str:
        """
        Generează un răspuns la o întrebare folosind documentele recuperate.
        
        Args:
            query: Întrebarea utilizatorului
            context_docs: Lista de documente relevante
            temperature: Temperatura pentru generare (valoare mai mică = răspunsuri mai consistente)
            max_output_tokens: Numărul maxim de tokeni pentru răspuns
            top_k: Parametrul top_k pentru generare
            top_p: Parametrul top_p pentru generare
            use_web_search: Dacă să se folosească căutarea web (ignorat în această versiune)
            
        Returns:
            Răspunsul generat
        """
        # Extragem conținutul documentelor - limităm la primele 5 pentru a evita depășirea limitei de tokeni
        document_contents = self._extract_document_contents(context_docs[:5])
        
        # Verificăm dacă avem documente
        if not document_contents:
            logger.warning("Nu s-au găsit documente relevante pentru interogare", extra={"query": query})
            return "Nu am găsit informații relevante pentru a răspunde la această întrebare."
        
        # Generăm un hash pentru interogare și documente pentru cache
        cache_key = self._generate_cache_key(query, document_contents, temperature)
        
        # Verificăm dacă răspunsul este în cache
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.info(f"Răspuns găsit în cache pentru interogarea", 
                       extra={"query": query[:50], "cache_hit": True})
            return cached_response
        
        # Construim prompt-ul pentru Gemini
        prompt = self._build_prompt(query, document_contents)
        
        try:
            # Obținem modelul Gemini
            model = self._get_gemini_model()
            
            # Generăm răspunsul cu măsurarea timpului
            start_time = time.time()
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            generation_time = time.time() - start_time
            
            # Logăm metricile de performanță
            logger.info(f"Generare răspuns Gemini finalizată", 
                       extra={
                           "duration": f"{generation_time:.2f}s", 
                           "duration_ms": int(generation_time * 1000),
                           "query_length": len(query),
                           "doc_count": len(context_docs),
                           "temperature": temperature
                       })
            
            # Extragem textul răspunsului
            answer = response.text
            
            # Salvăm răspunsul în cache
            self._add_to_cache(cache_key, answer)
            
            return answer
        except Exception as e:
            logger.error(f"Eroare la generarea răspunsului cu Gemini", 
                        extra={"error": str(e), "query": query[:50]})
            return f"A apărut o eroare la generarea răspunsului: {str(e)}"
    
    def _extract_document_contents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Extrage conținutul documentelor.
        
        Args:
            documents: Lista de documente
        
        Returns:
            Lista de conținuturi ale documentelor
        """
        contents = []
        for doc in documents:
            if isinstance(doc, dict):
                content = doc.get("content", "")
            else:
                content = getattr(doc, "content", "")
            
            contents.append(content)
        
        return contents
    
    def _build_prompt(self, query: str, document_contents: List[str]) -> str:
        """
        Construiește prompt-ul pentru Gemini.
        
        Args:
            query: Întrebarea utilizatorului
            document_contents: Conținuturile documentelor
        
        Returns:
            Prompt-ul pentru Gemini
        """
        prompt = f"""
        Ești un asistent AI expert care răspunde la întrebări bazate pe informațiile din documentele furnizate.
        
        ÎNTREBARE: {query}
        
        CONTEXT DIN DOCUMENTE:
        """
        for i, content in enumerate(document_contents):
            prompt += f"\n\nDocument {i+1}:\n{content}"
        
        prompt += f"""
        
        Instrucțiuni:
        1. Răspunde DOAR pe baza informațiilor din documentele furnizate.
        2. Dacă informația necesară nu se găsește în documente, spune clar că nu poți răspunde la întrebare bazat pe documentele disponibile.
        3. Nu inventa informații care nu sunt prezente în documente.
        4. Citează sursele documentelor relevante în răspunsul tău.
        5. Răspunde în limba română, folosind un ton profesional și clar.
        
        RĂSPUNS:
        """
        
        return prompt
            
    def _add_to_cache(self, key: str, value: str) -> None:
        """
        Adaugă un răspuns în cache cu limitarea dimensiunii cache-ului.
        
        Args:
            key: Cheia pentru cache
            value: Valoarea de stocat (răspunsul generat)
        """
        global _response_cache, MAX_CACHE_SIZE
        
        # Dacă cache-ul a atins dimensiunea maximă, eliminăm cea mai veche intrare
        if len(_response_cache) >= MAX_CACHE_SIZE:
            # OrderedDict păstrează ordinea inserării, deci prima cheie este cea mai veche
            _response_cache.popitem(last=False)  # last=False înseamnă FIFO (first in, first out)
        
        # Adăugăm noul răspuns în cache
        _response_cache[key] = value
    
    def _extract_document_content(self, doc) -> tuple:
        """
        Extrage conținutul și metadatele dintr-un document, indiferent de format.
        
        Args:
            doc: Documentul din care se extrage conținutul (dict sau Document)
            
        Returns:
            Tuple conținând (conținut, sursă)
        """
        # Gestionăm diferite tipuri de documente în mod unificat
        if isinstance(doc, dict):
            content = doc.get("content", "")
            metadata = doc.get("meta", {})
        else:  # Presupunem că este un obiect Document din Haystack
            content = getattr(doc, "content", "")
            metadata = getattr(doc, "meta", {})
            
        # Extragem sursa din metadata
        if isinstance(metadata, dict):
            source = metadata.get("source", "Necunoscut")
        else:
            source = "Necunoscut"
            
        return content, source
