import re, time
from typing import List, Dict, Set
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
except Exception:
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
        t = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in t.split() if len(w)>2]
class RAGSystem:
    def __init__(self, documents: List[Dict], hf_model=None, chunk_size_words=150):
        self.documents = documents
        self.chunk_size = chunk_size_words
        self.hf_model = hf_model
        self.chunks = []
        self._process_documents()
    def _process_documents(self):
        for doc in self.documents:
            self.chunks.extend(self._chunk_document(doc))
    def _chunk_document(self, doc, max_words=None):
        if max_words is None:
            max_words = self.chunk_size
        content = doc.get('content','')
        sentences = re.findall(r'[^.!?]+[.!?]+', content)
        if not sentences:
            sentences = [content]
        chunks=[]
        current=[]
        wc=0
        for s in sentences:
            s=s.strip()
            words=len(s.split())
            if wc+words>max_words and current:
                text=' '.join(current)
                chunks.append(DocumentChunk(doc['id'], doc.get('title',''), text, f"{doc['id']}_chunk_{len(chunks)}"))
                current=[s]; wc=words
            else:
                current.append(s); wc+=words
        if current:
            text=' '.join(current)
            chunks.append(DocumentChunk(doc['id'], doc.get('title',''), text, f"{doc['id']}_chunk_{len(chunks)}"))
        return chunks
    def precompute_embeddings(self, model):
        if model is None:
            return
        texts = [c.text for c in self.chunks]
        embs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        for c,e in zip(self.chunks, embs):
            c.embedding = e
    def retrieve(self, query, top_k=3):
        # If query is embedding vector
        if isinstance(query, (list,tuple)) or (hasattr(query,'shape') and getattr(query,'shape') is not None):
            q_emb = np.array(query)
            scored=[]
            for c in self.chunks:
                if c.embedding is not None:
                    sim = float(cosine_similarity([q_emb],[c.embedding])[0][0])
                    scored.append((c,sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        # else keyword overlap
        q_words = set(re.sub(r'[^\w\s]',' ', query.lower()).split())
        scored=[]
        for c in self.chunks:
            inter = len(q_words & set(c.words))
            union = len(q_words | set(c.words)) if q_words or c.words else 1
            score = inter/union
            scored.append((c,score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    def generate_answer(self, query, chunks, llm_gen_fn=None):
        if llm_gen_fn:
            context='\n\n'.join([f"[{i+1}] {c.text}" for i,(c,_) in enumerate(chunks)])
            return llm_gen_fn(query, context)
        # simple extractive
        if not chunks:
            return "No relevant info found."
        context=' '.join([c.text for c,_ in chunks])
        sentences = re.findall(r'[^.!?]+[.!?]+', context)
        if not sentences:
            return context[:400]
        q_words=set(re.sub(r'[^\w\s]',' ', query.lower()).split())
        ranked=[]
        for s in sentences:
            s_words=set(re.sub(r'[^\w\s]',' ', s.lower()).split())
            ranked.append((s.strip(), len(q_words & s_words)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ' '.join([s for s,_ in ranked[:2]])
