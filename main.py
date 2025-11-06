import re
from typing import List, Dict
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class DocumentChunk:
    """Represents a text chunk from a document."""

    def __init__(self, doc_id, doc_title, text, chunk_id):
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.text = text
        self.chunk_id = chunk_id
        self.words = self._extract_words(text)
        self.word_freq = Counter(self.words)
        self.embedding = None

    def _extract_words(self, text):
        """Normalize and tokenize text."""
        t = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in t.split() if len(w) > 2]


class RAGSystem:
    """Retrieval-Augmented Generation (RAG) system for document QA."""

    def __init__(self, documents: List[Dict], hf_model=None, chunk_size_words=250):
        self.documents = documents
        self.chunk_size = chunk_size_words
        self.hf_model = hf_model
        self.chunks = []
        self._process_documents()

    # ---------------- Document Processing ----------------
    def _process_documents(self):
        """Split all docs into chunks."""
        for doc in self.documents:
            chunks = self._chunk_document(doc)
            self.chunks.extend(chunks)

        print(f"[DEBUG] Created {len(self.chunks)} total chunks.")
        for i, c in enumerate(self.chunks[:3]):
            print(f"[DEBUG] Chunk {i+1} preview: {c.text[:120]}...")

    def _chunk_document(self, doc, max_words=None):
        """Robustly split a document into ~max_words chunks."""
        if max_words is None:
            max_words = self.chunk_size
        content = doc.get("content", "")

        if not content.strip():
            print(f"[DEBUG] Skipping empty doc: {doc.get('title', 'untitled')}")
            return []

        # Step 1: split on punctuation and newlines
        parts = re.split(r"(?<=[.!?])\s+|\n+|(?:^|\n)\s*[•\-\*]\s+", content)
        parts = [p.strip() for p in parts if len(p.strip()) > 0]

        # Step 2: if still too long, force split by words
        sentences = []
        for part in parts:
            if len(part.split()) > max_words * 2:
                words = part.split()
                for i in range(0, len(words), max_words):
                    sub = " ".join(words[i:i + max_words]).strip()
                    if sub:
                        sentences.append(sub)
            else:
                sentences.append(part)

        # Step 3: assemble into chunks of ~max_words
        chunks = []
        current, wc = [], 0
        for s in sentences:
            w = len(s.split())
            if wc + w > max_words and current:
                text = " ".join(current)
                chunks.append(DocumentChunk(doc["id"], doc.get("title", ""), text, f"{doc['id']}_chunk_{len(chunks)}"))
                current, wc = [s], w
            else:
                current.append(s)
                wc += w

        if current:
            text = " ".join(current)
            chunks.append(DocumentChunk(doc["id"], doc.get("title", ""), text, f"{doc['id']}_chunk_{len(chunks)}"))

        print(f"[DEBUG] Doc '{doc.get('title', 'untitled')}': {len(chunks)} chunks created.")
        return chunks

    # ---------------- Embeddings ----------------
    def precompute_embeddings(self, model):
        """Compute embeddings for all chunks."""
        if model is None:
            print("[DEBUG] No embedding model provided; skipping.")
            return
        texts = [c.text for c in self.chunks]
        if not texts:
            print("[DEBUG] No texts to embed!")
            return

        print(f"[DEBUG] Computing embeddings for {len(texts)} chunks...")
        embs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        for c, e in zip(self.chunks, embs):
            c.embedding = e
        print(f"[DEBUG] Embedded {len(embs)} chunks. Vector dim: {len(embs[0]) if len(embs) else 'N/A'}")

    # ---------------- Retrieval ----------------
    def retrieve(self, query, top_k=3):
        """Retrieve top-k relevant chunks."""
        if not self.chunks:
            print("[DEBUG] No chunks to retrieve from.")
            return []

        # Semantic retrieval if embedding given
        is_embedding = isinstance(query, (list, np.ndarray)) or (
            hasattr(query, "shape") and query.shape is not None
        )

        if is_embedding:
            q_emb = np.array(query)
            scored = []
            for c in self.chunks:
                if c.embedding is not None:
                    sim = float(cosine_similarity([q_emb], [c.embedding])[0][0])
                    scored.append((c, sim))
                else:
                    scored.append((c, 0.0))
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored:
                print(f"[DEBUG] Semantic retrieval top score: {scored[0][1]:.3f}")
            return scored[:top_k]

        # Fallback keyword retrieval
        q_words = set(re.sub(r"[^\w\s]", " ", str(query).lower()).split())
        q_words = {w for w in q_words if len(w) > 2}
        scored = []
        for c in self.chunks:
            inter = len(q_words & set(c.words))
            union = len(q_words | set(c.words)) if (q_words or c.words) else 1
            score = inter / union if union > 0 else 0.0
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            print(f"[DEBUG] Keyword retrieval top score: {scored[0][1]:.3f}")
        return scored[:top_k]

    # ---------------- Answer generation ----------------
    def generate_answer(self, query, chunks, llm_gen_fn=None):
        """Generate answer using LLM or extractive fallback."""
        if not chunks:
            return "No relevant information found."

        # Use LLM if provided
        if llm_gen_fn:
            context = "\n\n".join([f"[{i+1}] {c.text}" for i, (c, _) in enumerate(chunks)])
            if not context.strip():
                return "No relevant context found in documents."
            return llm_gen_fn(query, context)

        # Extractive fallback
        context = " ".join([c.text for c, _ in chunks])
        sentences = re.split(r"(?<=[.!?])\s+", context)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return context[:400]

        q_words = set(re.sub(r"[^\w\s]", " ", query.lower()).split())
        q_words = {w for w in q_words if len(w) > 2}
        ranked = []
        for s in sentences:
            s_words = set(re.sub(r"[^\w\s]", " ", s.lower()).split())
            overlap = len(q_words & s_words)
            ranked.append((s.strip(), overlap))
        ranked.sort(key=lambda x: x[1], reverse=True)
        result = " ".join([s for s, _ in ranked[:3] if s])
        return result if result else context[:400]
