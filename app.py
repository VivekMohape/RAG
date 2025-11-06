import streamlit as st
import os, time, json
from main import RAGSystem
from evaluation import run_evaluation, load_eval_dataset

st.set_page_config(page_title="RAG QA (Groq OSS 20B)", layout="wide")
st.title("Retrieval-Augmented QA — Groq OSS 20B (demo)")

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

# File uploads
uploads = st.file_uploader("Upload .txt/.md docs (multiple)", accept_multiple_files=True, type=["txt", "md"])

# Default fallback documents
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

# Load uploaded or sample docs
documents = []
if uploads:
    for f in uploads:
        text = f.read().decode('utf-8', errors='ignore')
        documents.append({
            "id": f"name_{int(time.time())}",
            "title": f.name,
            "content": text
        })

if not documents:
    documents = SAMPLE_DOCS
    st.info("Using sample document set (upload files to override).")

# ✅ Cached Hugging Face model loader (safe for Streamlit Cloud)
@st.cache_resource
def load_hf_model(name):
    from sentence_transformers import SentenceTransformer
    # Auto-fix model paths for cloud
    if name.lower() == "bge-small-en-v1.5":
        name = "BAAI/bge-small-en-v1.5"
    elif name.lower() == "all-minilm-l6-v2":
        name = "sentence-transformers/all-MiniLM-L6-v2"

    st.info(f"Loading embedding model: {name} ... (first time may take ~1 min)")
    return SentenceTransformer(name)


hf = load_hf_model(hf_model) if use_embeddings else None


# ✅ Initialize RAG safely (avoid unhashable cache issue)
def init_rag(docs, hf_model):
    rag = RAGSystem(docs, hf_model=hf_model)
    if hf_model:
        rag.precompute_embeddings(hf_model)
    return rag


if "rag" not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        st.session_state["rag"] = init_rag(documents, hf)

rag = st.session_state["rag"]
st.sidebar.success(f"{len(rag.chunks)} chunks loaded")


# Tabs: Chat / Evaluation / Docs / About
tabs = st.tabs(["Chat", "Evaluation", "Docs", "About"])

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

        # Define LLM generator function
        def llm_gen(q, context):
            prompt = f"""Use the context to answer the question concisely.

Context:
{context}

Question: {q}
Answer:"""
            try:
                if GROQ_KEY:
                    from groq import Groq
                    client = Groq(api_key=GROQ_KEY)
                    r = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=400
                    )
                    return r.choices[0].message.content
            except Exception as e:
                st.warning(f"Groq call failed: {e}")

            # fallback extractive answer
            return rag.generate_answer(q, chunks, llm_gen_fn=None)

        answer = rag.generate_answer(q, chunks, llm_gen_fn=llm_gen)
        latency = (time.time() - t0) * 1000
        st.metric("Latency (ms)", f"{latency:.0f}")
        st.markdown("**Answer:**")
        st.write(answer)

        with st.expander("Retrieved chunks (context)"):
            for i, (c, score) in enumerate(chunks, 1):
                st.markdown(f"**[{i}] {c.doc_title}** (score: {score:.3f})")
                st.write(c.text)

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
        with st.expander(d['title']):
            st.write(d['content'][:4000])

# ---------------- About Tab ----------------
with tabs[3]:
    st.header("About & Tips")
    st.markdown("""
    **Chunking**: Documents are split into ~150-word semantic chunks for balance between recall and context clarity.

    **Embeddings**: Cached in memory for speed (suitable for < 500 docs). Use Chroma or Pinecone for large datasets.

    **Answer generation**: Uses Groq OSS 20B if API key is set; otherwise uses extractive fallback.
    """)
