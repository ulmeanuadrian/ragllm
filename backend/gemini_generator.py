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
import re
from collections import Counter

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

# Dicționar general de sinonime și traduceri pentru diverse domenii
GENERAL_SYNONYMS = {
    # Termeni generali de căutare
    'cum': ['how', 'how to', 'în ce mod', 'în ce fel', 'modalitate'],
    'de ce': ['why', 'pentru ce', 'motivul', 'cauza', 'reason'],
    'ce': ['what', 'care', 'which', 'ce anume'],
    'unde': ['where', 'în care', 'location', 'locația'],
    'când': ['when', 'momentul când', 'timpul'],
    'cine': ['who', 'care persoană'],
    'cât': ['how much', 'how many', 'câte', 'cantitate'],
    
    # Termeni tehnici generali
    'cod': ['code', 'script', 'program', 'coding', 'programare'],
    'script': ['script', 'cod', 'code', 'file', 'fișier'],
    'functii': ['functions', 'funcții', 'methods', 'metode'],
    'funcții': ['functions', 'functii', 'methods', 'metode'],
    'componente': ['components', 'parts', 'părți', 'elemente', 'elements'],
    'structura': ['structure', 'organizare', 'organization', 'structuring'],
    'organizare': ['organization', 'structura', 'structure', 'organize'],
    'configurare': ['configuration', 'config', 'setup', 'setare'],
    'instalare': ['installation', 'install', 'setup', 'installing'],
    'folosire': ['usage', 'using', 'utilizare', 'use'],
    'utilizare': ['usage', 'using', 'folosire', 'use'],
    
    # Termeni de dezvoltare
    'aplicatie': ['application', 'app', 'aplicație', 'program'],
    'aplicație': ['application', 'app', 'aplicatie', 'program'],
    'proiect': ['project', 'proyect', 'aplicație', 'application'],
    'dezvoltare': ['development', 'developing', 'creating', 'dev'],
    'crearea': ['creating', 'creation', 'making', 'building'],
    'construire': ['building', 'construction', 'creating'],
    
    # Termeni de fișiere și directoare
    'fisiere': ['files', 'fișiere', 'documents', 'documente'],
    'fișiere': ['files', 'fisiere', 'documents', 'documente'],
    'directoare': ['directories', 'folders', 'dosare', 'foldere'],
    'foldere': ['folders', 'directories', 'directoare'],
    'dosare': ['folders', 'directories', 'directoare', 'foldere'],
    
    # Termeni de documentație
    'documentatie': ['documentation', 'docs', 'documentație'],
    'documentație': ['documentation', 'docs', 'documentatie'],
    'ghid': ['guide', 'tutorial', 'manual'],
    'tutorial': ['tutorial', 'guide', 'ghid', 'walkthrough'],
    'exemplu': ['example', 'sample', 'demo', 'demonstration'],
    'exemple': ['examples', 'samples', 'demos', 'demonstrations'],
    
    # Termeni de proces
    'pas': ['step', 'phase', 'stage', 'etapă'],
    'pasi': ['steps', 'phases', 'stages', 'etape'],
    'pași': ['steps', 'phases', 'stages', 'etape'],
    'etapa': ['stage', 'phase', 'step', 'pas'],
    'etape': ['stages', 'phases', 'steps', 'pași'],
    'proces': ['process', 'procedure', 'procedură'],
    'procedura': ['procedure', 'process', 'proces'],
    'procedură': ['procedure', 'process', 'proces'],
    
    # Termeni de calitate și optimizare
    'optimizare': ['optimization', 'optimize', 'îmbunătățire'],
    'imbunatatire': ['improvement', 'optimization', 'îmbunătățire'],
    'îmbunătățire': ['improvement', 'optimization', 'imbunatatire'],
    'performanta': ['performance', 'performanță', 'speed', 'viteza'],
    'performanță': ['performance', 'performanta', 'speed', 'viteza'],
    'calitate': ['quality', 'standard', 'nivel'],
    'eficienta': ['efficiency', 'eficiență', 'productive'],
    'eficiență': ['efficiency', 'eficienta', 'productive'],
    
    # Termeni de erori și probleme
    'eroare': ['error', 'mistake', 'greșeală', 'bug'],
    'erori': ['errors', 'mistakes', 'greșeli', 'bugs'],
    'problema': ['problem', 'issue', 'problemă'],
    'probleme': ['problems', 'issues', 'probleme'],
    'solutie': ['solution', 'fix', 'soluție'],
    'soluție': ['solution', 'fix', 'solutie'],
    'rezolvare': ['solving', 'resolution', 'fix'],
    
    # Termeni de testare
    'testare': ['testing', 'test', 'verificare'],
    'test': ['test', 'testing', 'verificare'],
    'verificare': ['verification', 'check', 'testing'],
    'validare': ['validation', 'verify', 'confirmare'],
    
    # Termeni de management
    'gestionare': ['management', 'managing', 'handling'],
    'management': ['management', 'gestionare', 'administrare'],
    'administrare': ['administration', 'management', 'gestionare'],
    'control': ['control', 'management', 'supervision'],
}

# Stop words în română și engleză (expandate)
STOP_WORDS_RO = {
    'și', 'în', 'la', 'de', 'cu', 'pe', 'din', 'pentru', 'este', 'sunt', 'a', 'al', 'ale',
    'că', 'să', 'se', 'nu', 'mai', 'dar', 'sau', 'dacă', 'când', 'cum', 'unde', 'care',
    'această', 'acest', 'aceasta', 'acesta', 'unei', 'unui', 'îi', 'le', 'o', 'un',
    'am', 'ai', 'au', 'avea', 'fi', 'fost', 'fiind', 'va', 'vor', 'avea', 'aș', 'ar',
    'către', 'despre', 'după', 'înainte', 'asupra', 'printre', 'între', 'sub', 'peste',
    'foarte', 'mult', 'puțin', 'destul', 'aproape', 'departe', 'aici', 'acolo', 'undeva'
}

STOP_WORDS_EN = {
    'and', 'in', 'to', 'of', 'with', 'on', 'from', 'for', 'is', 'are', 'the', 'a', 'an',
    'that', 'this', 'it', 'be', 'have', 'has', 'do', 'does', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'must', 'shall', 'at', 'by', 'up', 'as', 'if',
    'what', 'when', 'where', 'who', 'why', 'how', 'which', 'than', 'or', 'but', 'not',
    'was', 'were', 'been', 'being', 'had', 'having', 'about', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before'
}

ALL_STOP_WORDS = STOP_WORDS_RO.union(STOP_WORDS_EN)


class GeminiGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Inițializează generatorul Gemini optimizat pentru utilizare generală.
        
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
    def expand_query_terms(query: str) -> List[str]:
        """
        Expandează termenii din interogare cu sinonime și traduceri generale.
        
        Args:
            query: Interogarea originală
            
        Returns:
            Lista de termeni expandați
        """
        # Normalizăm query-ul
        query_lower = query.lower().strip()
        
        # Extragem cuvintele folosind regex pentru a păstra și caracterele speciale
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Lista de termeni expandați
        expanded_terms = set(words)  # Începem cu cuvintele originale
        
        # Adăugăm sinonime și traduceri din dicționarul general
        for word in words:
            if word in GENERAL_SYNONYMS:
                expanded_terms.update(GENERAL_SYNONYMS[word])
            
            # Căutăm și în valorile dicționarului (reverse lookup)
            for key, values in GENERAL_SYNONYMS.items():
                if word in values:
                    expanded_terms.add(key)
                    expanded_terms.update(values)
        
        # Eliminăm stop words
        expanded_terms = [term for term in expanded_terms if term not in ALL_STOP_WORDS]
        
        # Eliminăm cuvintele prea scurte
        expanded_terms = [term for term in expanded_terms if len(term) > 2]
        
        # Eliminăm duplicatele și sortăm pentru consistență
        expanded_terms = sorted(list(set(expanded_terms)))
        
        logger.debug(f"Query expandat: '{query}' -> {len(expanded_terms)} termeni: {expanded_terms[:10]}...")
        return expanded_terms
    
    @staticmethod
    def calculate_advanced_similarity(query: str, content: str) -> Dict[str, float]:
        """
        Calculează multiple tipuri de similaritate pentru o evaluare mai precisă.
        
        Args:
            query: Interogarea utilizatorului
            content: Conținutul documentului
            
        Returns:
            Dicționar cu diferite scoruri de similaritate
        """
        # 1. KEYWORD SIMILARITY - pe baza cuvintelor expandate
        query_terms = set(GeminiGenerator.expand_query_terms(query))
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        content_words = {word for word in content_words if len(word) > 2 and word not in ALL_STOP_WORDS}
        
        if not query_terms or not content_words:
            keyword_score = 0.0
        else:
            intersection = query_terms.intersection(content_words)
            union = query_terms.union(content_words)
            keyword_score = len(intersection) / len(union) if union else 0.0
        
        # 2. PHRASE SIMILARITY - căutare de fraze exacte
        phrase_score = 0.0
        query_phrases = [phrase.strip() for phrase in query.split(',') if len(phrase.strip()) > 3]
        content_lower = content.lower()
        
        for phrase in query_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in content_lower:
                # Bonus mai mare pentru fraze mai lungi
                phrase_bonus = min(0.3, len(phrase) / 50)
                phrase_score += phrase_bonus
        
        phrase_score = min(1.0, phrase_score)  # Cap la 1.0
        
        # 3. TERM FREQUENCY - cât de des apar termenii cheie
        tf_score = 0.0
        for term in query_terms:
            if term in content_lower:
                # Calculăm frecvența relativă
                count = content_lower.count(term)
                tf_score += min(0.1, count / 100)  # Normalizăm
        
        tf_score = min(1.0, tf_score)
        
        # 4. POSITION SCORE - termenii la începutul documentului sunt mai importanți
        position_score = 0.0
        content_start = content_lower[:500]  # Primele 500 caractere
        
        for term in query_terms:
            if term in content_start:
                position_score += 0.1
        
        position_score = min(1.0, position_score)
        
        # 5. LENGTH BONUS - documentele cu lungime moderată sunt preferate
        length_score = 0.0
        content_length = len(content)
        if 200 <= content_length <= 3000:  # Lungime optimă
            length_score = 0.2
        elif 100 <= content_length < 200 or 3000 < content_length <= 5000:
            length_score = 0.1
        
        scores = {
            'keyword': keyword_score,
            'phrase': phrase_score,
            'term_frequency': tf_score,
            'position': position_score,
            'length': length_score
        }
        
        # Calculăm scorul final combinat cu ponderi
        weights = {
            'keyword': 0.4,      # 40% - cel mai important
            'phrase': 0.25,      # 25% - fraze exacte sunt importante
            'term_frequency': 0.2, # 20% - frecvența termilor
            'position': 0.1,     # 10% - poziția în document
            'length': 0.05       # 5% - lungimea documentului
        }
        
        final_score = sum(scores[key] * weights[key] for key in scores)
        scores['final'] = min(1.0, final_score)
        
        if scores['final'] > 0.1:  # Log doar pentru potriviri semnificative
            logger.debug(f"Similaritate: {scores['final']:.3f} (k:{scores['keyword']:.2f}, p:{scores['phrase']:.2f}, tf:{scores['term_frequency']:.2f})")
        
        return scores
    
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
        for content in document_contents[:5]:  # Luăm doar primele 5 pentru performanță
            # Folosim doar primele 200 de caractere pentru eficiență
            content_sample = content[:200].strip()
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
            
            if content and content.strip():  # Ignorăm conținutul gol
                contents.append(content.strip())
        
        return contents
    
    @staticmethod
    def _build_enhanced_prompt(query: str, document_contents: List[str]) -> str:
        """
        Construiește un prompt îmbunătățit și general pentru Gemini.
        
        Args:
            query: Întrebarea utilizatorului
            document_contents: Conținuturile documentelor
        
        Returns:
            Prompt-ul pentru Gemini
        """
        # Creștem numărul de documente pentru context mai bogat
        max_docs = 10  # Creștem de la 5 la 10
        limited_contents = document_contents[:max_docs]
        
        # Expandăm termenii din query pentru context
        expanded_terms = GeminiGenerator.expand_query_terms(query)
        
        prompt = f"""Ești un asistent AI expert specializat în analiza documentelor și răspunsul la întrebări în limba română. Ai acces la un set de documente care pot conține informații în română sau engleză.

ÎNTREBARE UTILIZATOR: {query}

TERMENI CHEIE IDENTIFICAȚI: {', '.join(expanded_terms[:15])}

CONTEXT DIN DOCUMENTE:"""
        
        for i, content in enumerate(limited_contents, 1):
            # Creștem lungimea permisă pentru fiecare document
            max_content_length = 4000  # Creștem pentru mai mult context
            truncated_content = content[:max_content_length]
            if len(content) > max_content_length:
                truncated_content += "...[content truncated]"
            
            # Calculăm relevanța pentru acest document
            similarity_scores = GeminiGenerator.calculate_advanced_similarity(query, content)
            relevance = similarity_scores['final']
            
            prompt += f"\n\n--- DOCUMENT {i} (Relevanță: {relevance:.1%}) ---\n{truncated_content}"
        
        prompt += f"""

--- INSTRUCȚIUNI PENTRU RĂSPUNS ---
🎯 OBIECTIV: Răspunde în limba română la întrebarea utilizatorului bazându-te EXCLUSIV pe informațiile din documentele de mai sus.

📋 REGULI OBLIGATORII:
1. ✅ Răspunde în română, într-un mod clar, profesional și detaliat
2. 📚 Folosește DOAR informațiile din documentele furnizate
3. 🔍 Dacă găsești informații relevante, explică-le cu exemple concrete din documente
4. ❌ Dacă informațiile nu sunt suficiente, spune clar: "Pe baza documentelor disponibile, nu pot oferi informații complete despre..."
5. 🏗️ Structurează răspunsul logic: introducere → puncte principale → exemple → concluzie
6. 💡 Oferă sfaturi practice când informațiile permit
7. 🔗 Fă conexiuni între informații din diferite documente când este relevant
8. 📊 Folosește bullet points sau numerotare pentru claritate când este necesar
9. ⚡ Fii concis dar complet - evită informații redundante

🎨 STRUCTURA RĂSPUNSULUI:
- Începe cu un rezumat scurt (1-2 propoziții)
- Dezvoltă punctele principale cu detalii din documente
- Încheie cu o concluzie practică

⚠️ NU INVENTA informații care nu sunt în documente!
⚠️ NU folosi cunoștințe generale - doar ce este în context!

RĂSPUNS DETALIAT ÎN ROMÂNĂ:"""
        
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
        _response_cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'access_count': 0
        }
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
            # Actualizăm statisticile de acces
            cached_item = _response_cache.pop(key)
            cached_item['access_count'] += 1
            cached_item['last_access'] = time.time()
            
            # Reintroducem la sfârșit pentru LRU behavior
            _response_cache[key] = cached_item
            
            logger.debug(f"Cache hit pentru cheia: {key[:16]}... (accesat de {cached_item['access_count']} ori)")
            return cached_item['value']
        
        logger.debug(f"Cache miss pentru cheia: {key[:16]}...")
        return None
    
    def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]], 
        temperature: float = 0.3,  # Creștem de la 0.2 la 0.3 pentru mai multă creativitate
        max_output_tokens: int = 1500,  # Creștem pentru răspunsuri mai detaliate
        top_k: int = 40,
        top_p: float = 0.95,
        use_web_search: bool = False,
    ) -> str:
        """
        Generează un răspuns la o întrebare folosind documentele recuperate.
        OPTIMIZAT pentru orice tip de conținut JSON chunkizat.
        
        Args:
            query: Întrebarea utilizatorului
            context_docs: Lista de documente relevante
            temperature: Temperatura pentru generare (0.3 pentru echilibru creativitate/precizie)
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
            return "Nu am găsit informații relevante pentru a răspunde la această întrebare. Încercați să reformulați întrebarea sau să verificați dacă documentele din colecție conțin informațiile căutate."
        
        # Extragem conținutul documentelor - creștem limita
        document_contents = self._extract_document_contents(context_docs[:10])  # Creștem de la 5 la 10
        
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
        prompt = self._build_enhanced_prompt(query, document_contents)
        
        try:
            # Generăm răspunsul cu măsurarea timpului
            start_time = time.time()
            
            generation_config = {
                "temperature": max(0.0, min(1.0, temperature)),  # Clamp la [0,1]
                "top_p": max(0.0, min(1.0, top_p)),
                "top_k": max(1, min(100, top_k)),
                "max_output_tokens": max(200, min(3000, max_output_tokens)),  # Creștem limita maximă
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
                           "prompt_length": len(prompt),
                           "temperature": temperature,
                           "model": self.model_name
                       })
            
            # Extragem textul răspunsului
            if not response or not response.text:
                logger.error("Răspuns gol primit de la Gemini")
                return "Nu am putut genera un răspuns. Acest lucru se poate întâmpla dacă întrebarea este prea complexă sau documentele nu conțin suficiente informații relevante. Încercați să reformulați întrebarea."
            
            answer = response.text.strip()
            
            # Validăm că răspunsul nu este prea scurt
            if len(answer) < 20:  # Creștem pragul de la 10 la 20
                logger.warning("Răspuns prea scurt generat de Gemini", 
                             extra={"answer_length": len(answer), "answer_preview": answer[:100]})
                return f"Răspunsul generat este prea scurt. Pe baza documentelor disponibile, am găsit informații limitate. Încercați să reformulați întrebarea pentru a obține un răspuns mai detaliat.\n\nRăspuns parțial: {answer}"
            
            # Verificăm dacă răspunsul conține informații utile
            if any(phrase in answer.lower() for phrase in ["nu pot", "nu am", "nu există", "nu găsesc"]):
                logger.info("Răspuns cu informații limitate generat")
                # Nu returnăm eroare, ci lăs