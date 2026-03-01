#RAG_PUBMED — Retrieval Augmented Generation for PubMed

___

## Project Overview

This repository implements a **Retrieval-Augmented Generation (RAG)** based semantic search system for PubMed articles.

The system:

- Fetches articles from PubMed
- Preprocesses and chunks text
- Generates embeddings
- Stores them in a FAISS vector database
- Provides:
  - Streamlit UI for semantic search
  -  LLM-based summarization using Ollama

___

# End-to-End System Flow

---

## 1️⃣ Data Collection — Fetching from PubMed

**Scripts:**
- `backend/fetch_pubmed.py`
- `fetch_pubmed_articles_batch.py`

**Process:**
- Uses NCBI E-utilities (`esearch`, `efetch`)
- Retrieves:
  - Title
  - Abstract
  - PMID
  - PubMed URL
  - Saves data 

___

## 2️⃣ Preprocessing & Chunking

**Scripts:**
- `backend/text_processing.py`
- `backend/embeddings.py`
- `index_from_json.py`
- `index_articles.py`

**Methods Used:**

### ➤ LangChain RecursiveCharacterTextSplitter
- Configurable chunk size
- Overlapping chunks

### ➤ NLTK Sentence Chunking
- Min 2 sentences
- Max 5 sentences
- Overlap: 1 sentence

**Output:**
data/processed/pubmed_clean.json


___

## 3️⃣ Embedding Generation

**File:**
backend/embeddings.py

**Model Used:**
sentence-transformers/all-MiniLM-L6-v2

- 384-dimensional embeddings
- Encodes each text chunk into dense vectors

  ## 4️⃣ Vector Database — FAISS

**File:**

backend/vector_store.py

## 5️⃣ Retrieval Pipeline (RAG)

**File:**

backend/rag_pipeline.py

### Query Strategy:

- Dynamic query expansion
- Multiple embedding searches
- Distance threshold filtering
- Deduplication by (pmid, chunk)
- Top-k ranking

Semantic similarity is prioritized over keyword matching.



