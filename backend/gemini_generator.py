import os
import time
import hashlib
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from functools import lru_cache
import google.generativeai as genai
from dotenv import load_dotenv

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

# Cache pentru răspunsuri - optimizat cu LRU
_response_cache = OrderedDict()
MAX_CACHE_SIZE = 100  # Limităm dimensiunea cache-ului


class GeminiGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Inițializează generatorul Gemini.
        
        Args:
            model_name: Numele modelului Gemini de utilizat (implicit: gemini-2.5-flash)
        """
        self.model_name = model_name
        self._initialize_model()
    
    def _initialize_model(self):
        """Inițializează modelul pentru a fi pregătit de utilizare."""
        global _gemini_model_instances
        
        if self.model_name not in _gemini_model_instances:
            logger.info(f"Crearea unei noi instanțe pentru modelul {self.model_name}")
            try:
                model_instance = genai.GenerativeModel(self.model_name)
                # Test inițializare cu o interogare simplă
                _ = model_instance.generate_content("Test inițializare")
                _gemini_model_instances[self.model_name] = model_instance
                logger.info(f"Modelul {self.model_name} a fost inițializat cu succes.")
            except Exception as e:
                logger.error(f"Eroare la inițializarea modelului: {e}")
                raise
        else:
            logger.info(f"Folosirea instanței existente pentru modelul {self.model_name}")
            
        self.model = _gemini_model_instances[self.model_name]
    
    @staticmethod
    def _generate_cache_key(query: str, document_contents: List[str], temperature: float = 0.2) -> str:
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
        hash_input = f"{query_normalized}|temp:{temperature}"
        
        # Adăugăm fingerprint-uri pentru conținutul documentelor
        for content in document_contents:
            # Folosim doar primele 100 de caractere pentru eficiență
            content_sample = content[:100].strip()
            content_hash = hashlib.md5(content_sample.encode()).hexdigest()[:8]
            hash_input += f"|{content_hash}"
        
        # Generăm hash-ul final
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    @staticmethod
    def _extract_document_contents(documents: List[Dict[str, Any]]) -> List[str]:
        """
        Extrage conținutul documentelor în mod optimizat.
        
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
            
            if content.strip():  # Ignorăm conținutul gol
                contents.append(content.strip())
        
        return contents
    
    @staticmethod
    def _build_prompt(query: str, document_contents: List[str]) -> str:
        """
        Construiește prompt-ul pentru Gemini în mod optimizat.
        
        Args:
            query: Întrebarea utilizatorului
            document_contents: Conținuturile documentelor
        
        Returns:
            Prompt-ul pentru Gemini
        """
        # Limităm numărul de documente pentru a evita depășirea limitei de tokeni
        max_docs = 5
        limited_contents = document_contents[:max_docs]
        
        prompt = f"""Ești un asistent AI expert care răspunde la întrebări bazate pe informațiile din documentele furnizate.

ÎNTREBARE: {query}

CONTEXT DIN DOCUMENTE:"""
        
        for i, content in enumerate(limited_contents, 1):
            # Limităm lungimea fiecărui document
            max_content_length = 2000
            truncated_content = content[:max_content_length]
            if len(content) > max_content_length:
                truncated_content += "..."
            
            prompt += f"\n\nDocument {i}:\n{truncated_content}"
        
        prompt += f"""

Instrucțiuni:
1. Răspunde DOAR pe baza informațiilor din documentele furnizate.
2. Dacă informația necesară nu se găsește în documente, spune clar că nu poți răspunde la întrebare bazat pe documentele disponibile.
3. Nu inventa informații care nu sunt prezente în documente.
4. Citează sursele documentelor relevante în răspunsul tău.
5. Răspunde în limba română, folosind un ton profesional și clar.
6. Structurează răspunsul în paragrafe clare și ușor de înțeles.

RĂSPUNS:"""
        
        return prompt
    
    @staticmethod
    def _add_to_cache(key: str, value: str) -> None:
        """
        Adaugă un răspuns în cache cu gestionare optimizată a memoriei.
        
        Args:
            key: Cheia pentru cache
            value: Valoarea de stocat (răspunsul generat)
        """
        global _response_cache, MAX_CACHE_SIZE
        
        # Dacă cache-ul a atins dimensiunea maximă, eliminăm cea mai veche intrare
        if len(_response_cache) >= MAX_CACHE_SIZE:
            # OrderedDict păstrează ordinea inserării, deci prima cheie este cea mai veche
            _response_cache.popitem(last=False)  # FIFO (first in, first out)
        
        # Adăugăm noul răspuns în cache
        _response_cache[key] = value
        logger.debug(f"Răspuns adăugat în cache pentru cheia: {key[:16]}...")
    
    @staticmethod
    def _get_from_cache(key: str) -> Optional[str]:
        """
        Obține un răspuns din cache.
        
        Args:
            key: Cheia pentru cache
            
        Returns:
            Răspunsul din cache sau None dacă nu există
        """
        global _response_cache
        
        if key in _response_cache:
            # Mutăm la sfârșit pentru LRU behavior
            value = _response_cache.pop(key)
            _response_cache[key] = value
            logger.debug(f"Cache hit pentru cheia: {key[:16]}...")
            return value
        
        logger.debug(f"Cache miss pentru cheia: {key[:16]}...")
        return None
    
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
        # Validări input
        if not query or not query.strip():
            logger.warning("Interogare goală primită")
            return "Interogarea nu poate fi goală."
        
        if not context_docs:
            logger.warning("Nu s-au găsit documente relevante pentru interogare", 
                         extra={"query": query[:50]})
            return "Nu am găsit informații relevante pentru a răspunde la această întrebare."
        
        # Extragem conținutul documentelor - limităm la primele 5 pentru a evita depășirea limitei de tokeni
        document_contents = self._extract_document_contents(context_docs[:5])
        
        if not document_contents:
            logger.warning("Documentele nu conțin conținut valid", 
                         extra={"query": query[:50]})
            return "Documentele furnizate nu conțin informații valide."
        
        # Generăm un hash pentru interogare și documente pentru cache
        cache_key = self._generate_cache_key(query, document_contents, temperature)
        
        # Verificăm dacă răspunsul este în cache
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.info("Răspuns găsit în cache", 
                       extra={"query": query[:50], "cache_hit": True})
            return cached_response
        
        # Construim prompt-ul pentru Gemini
        prompt = self._build_prompt(query, document_contents)
        
        try:
            # Generăm răspunsul cu măsurarea timpului
            start_time = time.time()
            
            generation_config = {
                "temperature": max(0.0, min(1.0, temperature)),  # Clamp la [0,1]
                "top_p": max(0.0, min(1.0, top_p)),
                "top_k": max(1, min(100, top_k)),
                "max_output_tokens": max(100, min(2048, max_output_tokens)),
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            generation_time = time.time() - start_time
            
            # Logăm metricile de performanță
            logger.info("Generare răspuns Gemini finalizată", 
                       extra={
                           "duration": f"{generation_time:.2f}s", 
                           "duration_ms": int(generation_time * 1000),
                           "query_length": len(query),
                           "doc_count": len(context_docs),
                           "temperature": temperature,
                           "prompt_length": len(prompt)
                       })
            
            # Extragem textul răspunsului
            if not response or not response.text:
                logger.error("Răspuns gol primit de la Gemini")
                return "Nu am putut genera un răspuns. Încercați din nou."
            
            answer = response.text.strip()
            
            # Validăm că răspunsul nu este prea scurt
            if len(answer) < 10:
                logger.warning("Răspuns prea scurt generat de Gemini", 
                             extra={"answer_length": len(answer)})
                return "Răspunsul generat este prea scurt. Încercați să reformulați întrebarea."
            
            # Salvăm răspunsul în cache
            self._add_to_cache(cache_key, answer)
            
            return answer
            
        except Exception as e:
            logger.error("Eroare la generarea răspunsului cu Gemini", 
                        extra={"error": str(e), "query": query[:50]})
            return f"A apărut o eroare la generarea răspunsului. Vă rugăm să încercați din nou."
    
    @staticmethod
    def clear_cache() -> int:
        """
        Curăță cache-ul de răspunsuri.
        
        Returns:
            Numărul de intrări șterse din cache
        """
        global _response_cache
        cleared_count = len(_response_cache)
        _response_cache.clear()
        logger.info(f"Cache-ul Gemini a fost curățat: {cleared_count} intrări șterse")
        return cleared_count
    
    @staticmethod
    def get_cache_info() -> Dict[str, Any]:
        """
        Obține informații despre cache-ul de răspunsuri.
        
        Returns:
            Dicționar cu informații despre cache
        """
        global _response_cache, MAX_CACHE_SIZE
        
        return {
            "current_size": len(_response_cache),
            "max_size": MAX_CACHE_SIZE,
            "keys_sample": list(_response_cache.keys())[:5],  # Primele 5 chei pentru debug
            "memory_usage_estimate": sum(len(k) + len(v) for k, v in _response_cache.items())
        }