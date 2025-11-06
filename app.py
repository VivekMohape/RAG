import streamlit as st
import os, time, json
from main import RAGSystem
from evaluation import run_evaluation, load_eval_dataset

st.set_page_config(page_title="RAG QA (Groq OSS 20B)", layout="wide")
st.title("Retrieval-Augmented QA — Groq OSS 20B (demo)")

# Sidebar config
st.sidebar.header("Configuration")
GROQ_KEY = st.sidebar.text_input(
    "Groq API Key (or leave empty for local demo)",
    type="password",
    value=os.getenv("GROQ_API_KEY", "")
)
use_embeddings = st.sidebar.checkbox("Use embeddings (semantic retrieval)", True)
hf_model = st.sidebar.selectbox("Embedding model (HF)", ["bge-small-en-v1.5", "all-MiniLM-L6-v2"])
top_k = st.sidebar.slider("Top retrieved chunks", 1, 5, 3)

# File uploads
uploads = st.file_uploader("Upload .txt/.md docs (multiple)", accept_multiple_files=True, type=["txt", "md"])

# Load default small docset if no uploads
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


# ✅ Cached model loader (auto-fixes BGE path for Streamlit Cloud)
@st.cache_resource
def load_hf_model(name):
    from sentence_transformers import SentenceTransformer

    # Automatically map to the correct HF repo ID for reliability
    if name.lower() == "bge-small-en-v1.5":
        name = "BAAI/bge-small-en-v1.5"
    elif name.lower() == "all-minilm-l6-v2":
        name = "sentence-transformers/all-MiniLM-L6-v2"

    st.info(f"Loading embedding model: {name} ... (first time may take up to 1 min)")
    return SentenceTransformer(name)


hf = load_hf_model(hf_model) if use_embeddings else None


@st.cache_resource
def init_rag(docs, hf_model):
    rag = RAGSystem(docs, hf_model=hf_model)
    if hf_model:
        rag.precompute_embeddings(hf_model)
    return rag


rag = init_rag(documents, hf)
st.sidebar.success(f"{len(rag.chunks)} chunks")

tabs = st.tabs(["Chat", "Evaluation", "Docs", "About"])

# Chat Tab
with tabs[0]:
    st.header("Ask a question")
    q = st.text_input("Question")

    if st.button("Get Answer") and q:
        t0 = time.time()
        q_emb = None

        if use_embeddings and hf:
            q_emb = hf.encode(q, normalize_embeddings=True)

        chunks = rag.retrieve(q_emb if q_emb is not None else q, top_k=top_k)

        def llm_gen(q, context):
            prompt = f"""Use the context to answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"""
            try:
                if GROQ_KEY:
                    from groq import Groq
                    client = Groq(api_key=GROQ_KEY)
                    r = client.chat.completions.create(
                        model="llama-3-oss-20b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=400
                    )
                    return r.choices[0].message.content
            except Exception as e:
                st.warning(f"Groq call failed: {e}")
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

# Evaluation Tab
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

# Documents Tab
with tabs[2]:
    st.header("Documents")
    for d in documents:
        with st.expander(d['title']):
            st.write(d['content'][:4000])

# About Tab
with tabs[3]:
    st.header("About & Tips")
    st.markdown("""
    **Chunking**: Documents are chunked into ~150-word semantic units for a balance of context and recall.

    **Embedding store**: Cached in memory (fast for small datasets). Use Chroma or Pinecone for larger projects.

    **Answer generation**: Groq OSS 20B for generative answers if `GROQ_API_KEY` is set, else fallback to extractive answers.
    """)
