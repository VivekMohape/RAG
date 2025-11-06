import re
from typing import List, Dict, Set
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class DocumentChunk:
    def __init__(self, doc_id, doc_title, text, chunk_id):
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.text = text
        self.chunk_id = chunk_id
        self.words = self._extract_words(text)
        self.word_freq = Counter(self.words)
        self.embedding = None

    def _extract_words(self, text):
        t = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in t.split() if len(w) > 2]


class RAGSystem:
    def __init__(self, documents: List[Dict], hf_model=None, chunk_size_words=250):
        self.documents = documents
        self.chunk_size = chunk_size_words
        self.hf_model = hf_model
        self.chunks = []
        self._process_documents()

    def _process_documents(self):
        for doc in self.documents:
            chunks = self._chunk_document(doc)
            self.chunks.extend(chunks)
        print(f"[DEBUG] Created {len(self.chunks)} chunks total")
        for i, chunk in enumerate(self.chunks[:3]):
            print(f"[DEBUG] Chunk {i+1} preview: {chunk.text[:100]}...")

    def _chunk_document(self, doc, max_words=None):
        if max_words is None:
            max_words = self.chunk_size
        content = doc.get("content", "")
        
        if not content.strip():
            return []
        
        # Multi-strategy sentence splitting for robustness
        sentences = []
        
        # Strategy 1: Split on sentence endings with punctuation
        parts = re.split(r"(?<=[.!?])\s+", content)
        
        # Strategy 2: If very few splits, also split on newlines and bullets
        if len(parts) <= 2 and len(content.split()) > max_words:
            # Split on newlines, bullets, dashes, and sentence endings
            parts = re.split(r"(?<=[.!?])\s+|\n+|(?:^|\n)\s*[•\-\*]\s+", content)
        
        # Clean up the parts
        for part in parts:
            part = part.strip()
            if part:
                # If a single part is still too large, force split it
                if len(part.split()) > max_words * 2:
                    words = part.split()
                    for i in range(0, len(words), max_words):
                        sub_chunk = " ".join(words[i:i+max_words])
                        if sub_chunk.strip():
                            sentences.append(sub_chunk)
                else:
                    sentences.append(part)
        
        # Final fallback: if still empty or just one huge block
        if not sentences:
            sentences = [content]
        
        # Now group sentences into chunks of ~max_words
        chunks = []
        current = []
        wc = 0
        
        for s in sentences:
            if not s.strip():
                continue
                
            words = len(s.split())
            
            # Start new chunk if adding this would exceed limit
            if wc + words > max_words and current:
                text = " ".join(current)
                chunks.append(
                    DocumentChunk(
                        doc["id"],
                        doc.get("title", ""),
                        text,
                        f"{doc['id']}_chunk_{len(chunks)}"
                    )
                )
                current = [s]
                wc = words
            else:
                current.append(s)
                wc += words
        
        # Don't forget the last chunk
        if current:
            text = " ".join(current)
            chunks.append(
                DocumentChunk(
                    doc["id"],
                    doc.get("title", ""),
                    text,
                    f"{doc['id']}_chunk_{len(chunks)}"
                )
            )
        
        print(f"[DEBUG] Doc '{doc.get('title', 'untitled')}': {len(chunks)} chunks created")
        return chunks

    def precompute_embeddings(self, model):
        if model is None:
            return
        texts = [c.text for c in self.chunks]
        if not texts:
            print("[DEBUG] No texts to embed!")
            return
        embs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        for c, e in zip(self.chunks, embs):
            c.embedding = e
        print(f"[DEBUG] Embedded {len(embs)} chunks, vector dim: {len(embs[0]) if len(embs) > 0 else 'None'}")

    def retrieve(self, query, top_k=3):
        # Check if query is an embedding vector (numpy array, list, or has shape attribute)
        is_embedding = isinstance(query, (list, tuple, np.ndarray)) or (
            hasattr(query, 'shape') and query.shape is not None
        )
        
        if is_embedding:
            q_emb = np.array(query)
            scored = []
            for c in self.chunks:
                if c.embedding is not None:
                    sim = float(cosine_similarity([q_emb], [c.embedding])[0][0])
                    scored.append((c, sim))
                else:
                    # If chunks don't have embeddings, skip or score as 0
                    scored.append((c, 0.0))
            scored.sort(key=lambda x: x[1], reverse=True)
            print(f"[DEBUG] Semantic retrieval: top score = {scored[0][1]:.3f if scored else 'N/A'}")
            return scored[:top_k]
        
        # Fallback: keyword overlap (BM25-like)
        q_words = set(re.sub(r"[^\w\s]", " ", str(query).lower()).split())
        q_words = {w for w in q_words if len(w) > 2}
        
        scored = []
        for c in self.chunks:
            inter = len(q_words & set(c.words))
            union = len(q_words | set(c.words)) if (q_words or c.words) else 1
            score = inter / union if union > 0 else 0.0
            scored.append((c, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] Keyword retrieval: top score = {scored[0][1]:.3f if scored else 'N/A'}")
        return scored[:top_k]

    def generate_answer(self, query, chunks, llm_gen_fn=None):
        if llm_gen_fn:
            context = "\n\n".join([f"[{i+1}] {c.text}" for i, (c, _) in enumerate(chunks)])
            if not context.strip():
                return "No relevant context found in documents."
            return llm_gen_fn(query, context)
        
        # Simple extractive fallback
        if not chunks:
            return "No relevant info found."
        
        context = " ".join([c.text for c, _ in chunks])
        
        # Try to extract sentences
        sentences = re.split(r"(?<=[.!?])\s+", context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            # No sentence structure, return truncated context
            return context[:400].strip()
        
        # Rank sentences by keyword overlap with query
        q_words = set(re.sub(r"[^\w\s]", " ", query.lower()).split())
        q_words = {w for w in q_words if len(w) > 2}
        
        ranked = []
        for s in sentences:
            s_words = set(re.sub(r"[^\w\s]", " ", s.lower()).split())
            s_words = {w for w in s_words if len(w) > 2}
            overlap = len(q_words & s_words)
            ranked.append((s.strip(), overlap))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 2-3 sentences
        result = " ".join([s for s, _ in ranked[:3] if s])
        return result if result else context[:400].strip()
