# Aplicație RAG (Retrieval-Augmented Generation)

Această aplicație implementează un sistem RAG (Retrieval-Augmented Generation) pentru procesarea documentelor și generarea de răspunsuri bazate pe conținutul acestora.

## Caracteristici

- Procesare documente în formate multiple (PDF, DOCX, TXT, JSON)
- Fragmentare inteligentă a documentelor
- Căutare semantică folosind embeddings
- Generare de răspunsuri folosind Gemini AI
- API RESTful cu FastAPI
- Interfață web simplă

## Tehnologii utilizate

- Python
- farm-haystack
- FastAPI
- Qdrant (vector database)
- Gemini AI
- Sentence Transformers

## Instalare

1. Clonează repository-ul
2. Creează un mediu virtual Python: `python -m venv venv`
3. Activează mediul virtual:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Instalează dependențele: `pip install -r requirements.txt`
5. Creează un fișier `.env` și adaugă cheia API pentru Gemini: `GOOGLE_API_KEY=your_api_key`

## Utilizare

1. Pornește serverul Qdrant (dacă nu rulează deja)
2. Rulează aplicația backend: `python backend/rag_api.py`
3. Accesează interfața web din browser

## Structura proiectului

- `backend/` - Codul pentru backend-ul API
- `frontend/` - Interfața web simplă
- `storage/` - Fișiere pentru stocarea datelor
- `requirements.txt` - Dependențele Python necesare
