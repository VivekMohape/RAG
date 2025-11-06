# RAG Streamlit Demo (Groq OSS 20B)
This repository contains a small Retrieval-Augmented Generation (RAG) demo built with Streamlit.
It uses Hugging Face sentence-transformers for embeddings and Groq OSS 20B for answer generation (if you provide a Groq API key).

## Goals
Build a retrieval-augmented QA service over a small doc set. Includes:
- Chunking strategy
- In-memory embedding store (with option to upgrade)
- Answer generation using Groq OSS 20B (preferred) or extractive fallback
- Evaluation set (10 Q/A) with EM/F1/Faithfulness metrics

## Project structure
- `app.py` — Streamlit front-end (chat + eval)
- `main.py` — RAG core: chunking, embedding precompute, retrieval, extractive answer
- `evaluation.py` — Small QA dataset + metrics (EM/F1/Faithfulness)
- `requirements.txt`, `Dockerfile`

## Chunking strategy
- Documents are split into sentence-based chunks and grouped into ~150-word chunks.
- Rationale: 150 words balances context usefulness vs retrieval granularity for small docsets. For long documents, consider 200-400 words or semantic sentence groups determined by embedding similarity.

## Embedding store
- Default: in-memory embeddings cached in Streamlit process (suitable for <=~200 small docs).
- Upgrade path: Chroma, Pinecone, Weaviate for persistent, scalable vector stores.

## Answer generation
- If you supply `GROQ_API_KEY` in Streamlit sidebar, `app.py` will call Groq OSS 20B (`llama-3-oss-20b`) for generative answers using retrieved context.
- Fallback: extractive answer using simple heuristics from retrieved chunks.

## Evaluation
- `evaluation.py` includes a 10-question QA set and computes EM, F1, and a crude faithfulness heuristic (overlap-based).

## Trade-offs, cost & latency (estimates)
> These are approximate numbers to help planning. Actual results depend on model provider pricing, hardware, and request batching.

**Embedding (bge-small-en-v1.5 or all-MiniLM-L6-v2)**
- Latency: ~50-300 ms per request locally (model download first-run cost ~30-90s).
- Cost: If hosted via Hugging Face Inference endpoint, expect $0.0006 - $0.003 per 1K tokens (varies); local inference cost = compute/hardware cost.

**Groq OSS 20B (inference via Groq cloud)**
- Latency: ~200-800 ms per request for a single-turn chat (depends on model, batch, and provider).
- Cost: Groq pricing varies; OSS models are cheaper but you still pay per-token inference on their cloud. Estimate: $0.0005 - $0.005 per 1K tokens (very rough).
- Note: Using OSS model on Groq often yields similar quality to larger hosted models at lower cost, but may have differences in instruction-following.

**Overall example (small demo)**
- End-to-end latency (embedding + retrieval + generation): ~300 ms - 2s for small inputs when cached embeddings and a responsive LLM endpoint are used.
- Monthly cost (light usage, 1k queries/month): likely <$5-$50 depending on provider and tokens per query.

## Run locally
1. Create venv and install:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run:
   ```
   streamlit run app.py
   ```
3. Open `http://localhost:8501`

## Deploy
- Streamlit Cloud: push to GitHub, set `GROQ_API_KEY` as secret.
- Docker: build image via `docker build -t rag-streamlit .` and run.

## Notes & Next steps
- For production, replace in-memory store with a vector DB and add authentication.
- Add better prompt engineering and answer calibration for faithfulness (e.g., source attribution, confidence scoring).

## License
MIT
