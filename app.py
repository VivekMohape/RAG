import streamlit as st
import os, time, json
from main import RAGSystem
from evaluation import run_evaluation, load_eval_dataset

st.set_page_config(page_title="RAG QA (Groq OSS 120B)", layout="wide")
st.title("Retrieval-Augmented QA — Groq OSS 120B (demo)")

# Sidebar configuration
st.sidebar.header("Configuration")
GROQ_KEY = st.sidebar.text_input(
    "Groq API Key (optional for generation)",
    type="password",
    value=os.getenv("GROQ_API_KEY", "")
)
use_embeddings = st.sidebar.checkbox("Use embeddings (semantic retrieval)", True)
hf_model = st.sidebar.selectbox("Embedding model (HF)", ["bge-small-en-v1.5", "all-MiniLM-L6-v2"])
top_k = st.sidebar.slider("Top retrieved chunks", 1, 5, 3)

# File uploader
uploads = st.file_uploader("Upload .txt/.md docs (multiple)", accept_multiple_files=True, type=["txt", "md"])

# Default sample docs
SAMPLE_DOCS = [
    {
        "id": "doc1",
        "title": "Product A Spec",
        "content": "Product A is a compact device... It supports features X, Y, Z. Pricing starts at $199. Warranty 1 year."
    },
    {
        "id": "doc2",
        "title": "Installation Guide",
        "content": "To install Product A, first unbox. Then connect power. Follow safety guidelines..."
    },
    {
        "id": "doc3",
        "title": "Troubleshooting",
        "content": "Common issues include network, power, and firmware. For network reset, hold button 10s. For firmware, visit example.com/firmware."
    }
]

# --- Persistent document handling ---
if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = []

if uploads:
    new_docs = []
    for f in uploads:
        text = f.read().decode("utf-8", errors="ignore")
        new_docs.append({
            "id": f.name + "_" + str(int(time.time())),
            "title": f.name,
            "content": text
        })
    st.session_state["uploaded_docs"] = new_docs

# Use uploaded docs if available
if st.session_state["uploaded_docs"]:
    documents = st.session_state["uploaded_docs"]
    st.success(f"✅ Using {len(documents)} uploaded document(s).")
else:
    documents = SAMPLE_DOCS
    st.info("📄 Using default sample documents (Product A Spec, etc.).")

# Cached Hugging Face model loader
@st.cache_resource
def load_hf_model(name):
    from sentence_transformers import SentenceTransformer
    if name.lower() == "bge-small-en-v1.5":
        name = "BAAI/bge-small-en-v1.5"
    elif name.lower() == "all-minilm-l6-v2":
        name = "sentence-transformers/all-MiniLM-L6-v2"
    st.info(f"Loading embedding model: {name} ... (first run may take ~1 min)")
    return SentenceTransformer(name)

hf = load_hf_model(hf_model) if use_embeddings else None

# Safe re-init of RAG system
def init_rag(docs, hf_model):
    rag = RAGSystem(docs, hf_model=hf_model)
    if hf_model:
        rag.precompute_embeddings(hf_model)
    return rag

if "rag" not in st.session_state or len(st.session_state.get("rag_docs", [])) != len(documents):
    with st.spinner("Initializing RAG system..."):
        st.session_state["rag"] = init_rag(documents, hf)
        st.session_state["rag_docs"] = documents

rag = st.session_state["rag"]
st.sidebar.success(f"{len(rag.chunks)} chunks ready")

# Tabs: Chat / Evaluation / Docs / About
tabs = st.tabs(["Chat", "Evaluation", "Docs", "About"])

# ---------------- Debug View ----------------
with st.expander("🧩 Debug: View created chunks"):
    for i, c in enumerate(rag.chunks[:10], 1):
        st.markdown(f"**Chunk {i}** ({len(c.text.split())} words):")
        st.text(c.text[:400])

# ---------------- Chat Tab ----------------
with tabs[0]:
    st.header("Ask a question")
    q = st.text_input("Question")

    if st.button("Get Answer") and q:
        t0 = time.time()
        q_emb = None
        if use_embeddings and hf:
            q_emb = hf.encode(q, normalize_embeddings=True)

        chunks = rag.retrieve(q_emb if q_emb is not None else q, top_k=top_k)

        if not chunks:
            st.warning("No relevant context found.")
        else:
            with st.expander("🔍 Retrieved Chunks (context)"):
                for i, (c, s) in enumerate(chunks, 1):
                    st.markdown(f"**[{i}] {c.doc_title}** (score: {s:.3f})")
                    st.write(c.text[:600])

        def llm_gen(q, context):
            if not context.strip():
                return "No relevant context found."
            prompt = f"""
You are a helpful assistant that summarizes relevant details about the topic using the information in the context below. 
If needed, use your general knowledge to fill small gaps, but prioritize facts from the document.

Context:
{context}

Question: {q}

Answer:
"""
            try:
                if GROQ_KEY:
                    from groq import Groq
                    client = Groq(api_key=GROQ_KEY)
                    r = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=400
                    )
                    return r.choices[0].message.content.strip()
            except Exception as e:
                st.warning(f"Groq call failed: {e}")
            return rag.generate_answer(q, chunks, llm_gen_fn=None)

        answer = rag.generate_answer(q, chunks, llm_gen_fn=llm_gen)
        latency = (time.time() - t0) * 1000
        st.metric("Latency (ms)", f"{latency:.0f}")
        st.markdown("**Answer:**")
        st.write(answer)

# ---------------- Evaluation Tab ----------------
with tabs[1]:
    st.header("Run evaluation on small QA set")
    if st.button("Run eval"):
        eval_set = load_eval_dataset()
        results = run_evaluation(rag, eval_set, top_k=top_k)
        st.json(results)
        st.download_button(
            "Download results (json)",
            data=json.dumps(results, indent=2),
            file_name="eval_results.json"
        )

# ---------------- Docs Tab ----------------
with tabs[2]:
    st.header("Documents")
    for d in documents:
        with st.expander(d["title"]):
            st.write(d["content"][:4000])

# ---------------- About Tab ----------------
with tabs[3]:
    st.header("About & Tips")
    st.markdown("""
**Chunking:** Documents are split into ~250-word chunks for efficient semantic retrieval.

**Embeddings:** Cached locally in memory (suitable for <500 docs). Use Chroma or Pinecone for larger datasets.

**Answer generation:** Uses Groq OSS 120B if `GROQ_API_KEY` is provided; otherwise falls back to extractive answers.
""")
