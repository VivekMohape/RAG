# RAG QA System 

This repository contains a **Retrieval-Augmented Generation (RAG)** question-answering system built with **Python** and **Streamlit**.  
It demonstrates an end-to-end RAG pipeline — including **document chunking**, **embedding-based retrieval**, and **context-grounded answer generation** — using **Hugging Face SentenceTransformers** for embeddings and **Groq OSS models** for generation.

You can explore the fully deployed **live app**, upload `.txt` or `.md` files, and test the retrieval pipeline interactively.

---

## Live Resources
- **Live Streamlit App:** [https://opensource-rag.streamlit.app/](https://opensource-rag.streamlit.app/)  
- **Video Explanation:** [https://drive.google.com/file/d/1MDoYCCrZvP0PDsfj_x4br5E0hM32zuhA/view?usp=drivesdk](https://drive.google.com/file/d/1MDoYCCrZvP0PDsfj_x4br5E0hM32zuhA/view?usp=drivesdk)

---

## Project Goals
Implements the following assignment brief:

> “Build a minimal retrieval-augmented QA service over a small document set. Include: chunking strategy, embedding store, answer generation, and a short README with trade-offs and cost/latency numbers.”

### Core Features
- Hybrid **chunking** strategy (sentence + newline + bullet aware)
- **In-memory embedding store** for semantic retrieval (upgrade-ready for vector DBs)
- **Answer generation** using Groq OSS models (`openai/gpt-oss-120b`, `llama-3.3-70b`, `mixtral-8x7b`)
- **Extractive fallback** when no API key is provided
- Streamlit-based **interactive interface** for upload, retrieval, and querying
- Built-in **latency metrics** and debug views for analysis

---

## Project Structure
| File | Purpose |
|------|----------|
| `app.py` | Streamlit frontend (UI + configuration + evaluation) |
| `main.py` | RAG core logic — chunking, embedding, retrieval, answer generation |
| `evaluation.py` | Optional script for computing EM/F1/Faithfulness metrics |
| `requirements.txt` | Dependencies and deployment setup |

---

## Chunking Strategy
- Splits documents into sentence-level units using punctuation, newlines, and bullet detection.  
- Groups text into ~250-word chunks (adjustable from the UI).  
- Designed for structured text such as resumes, reports, or press releases.  
- Balances context coverage with retrieval precision for small datasets.

---

## Embedding Store
- **Default:** In-memory embeddings cached within Streamlit runtime (ideal for small datasets).  
- **Models Supported:**
  - `BAAI/bge-small-en-v1.5`
  - `sentence-transformers/all-MiniLM-L6-v2`
- **Upgrade Path:** Integrate Chroma, Pinecone, or Weaviate for persistence and scalability.  
- Embeddings are precomputed once per document and reused for subsequent queries.

---

## Answer Generation
- If `GROQ_API_KEY` is set, the app uses **Groq OSS models** (`openai/gpt-oss-120b` by default) for context-grounded generation.  
- Answers are generated strictly based on retrieved document context.  
- Without a Groq key, the app performs **extractive summarization** from retrieved chunks.  
- Sidebar controls allow model selection, top-k tuning, and chunk-size configuration.

---

## Trade-offs, Cost & Latency (approx.)
> Approximate figures for light usage with small text inputs.

| Component | Latency | Cost (est.) | Notes |
|------------|----------|-------------|-------|
| **Embedding (BGE/MiniLM)** | 50–300 ms | ~$0 (local) or $0.001 / 1K tokens | Cached after first run |
| **Groq LLM (OSS 120B)** | 200–800 ms | ~$0.0005–0.005 / 1K tokens | Dependent on query length |
| **End-to-End** | 300 ms – 2 s | — | Includes embedding + retrieval + generation |

**Trade-offs**
- In-memory embeddings: simple, fast, and ephemeral.  
- Groq based OpenAI OSS models: cost-effective and open-source but require API key.  
- Chunk size: smaller improves precision; larger improves context completeness.

---

