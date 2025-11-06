import streamlit as st
import os, time, json
from main import RAGSystem
from evaluation import run_evaluation, load_eval_dataset

st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("Retrieval-Augmented QA System")

# Sidebar configuration
st.sidebar.header("Configuration")
GROQ_KEY = st.sidebar.text_input(
    "Groq API Key (optional for generation)",
    type="password",
    value=os.getenv("GROQ_API_KEY", "")
)

# Model selection for Groq
groq_model = st.sidebar.selectbox(
    "Groq Model",
    ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    help="Select which Groq model to use for answer generation"
)

use_embeddings = st.sidebar.checkbox("Use embeddings (semantic retrieval)", True)
hf_model = st.sidebar.selectbox("Embedding model (HF)", ["bge-small-en-v1.5", "all-MiniLM-L6-v2"])
top_k = st.sidebar.slider("Top retrieved chunks", 1, 10, 3)
chunk_size = st.sidebar.slider("Chunk size (words)", 100, 500, 250)

# File uploader
uploads = st.file_uploader(
    "Upload .txt/.md docs (multiple)", 
    accept_multiple_files=True, 
    type=["txt", "md"],
    help="Upload your documents to query against"
)

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

# --- Document handling with better persistence ---
if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = []
    st.session_state["doc_hash"] = ""

if uploads:
    new_docs = []
    doc_names = []
    for f in uploads:
        text = f.read().decode("utf-8", errors="ignore")
        new_docs.append({
            "id": f.name.replace(" ", "_") + "_" + str(int(time.time())),
            "title": f.name,
            "content": text
        })
        doc_names.append(f.name)
    
    # Check if docs changed
    new_hash = "_".join(sorted(doc_names))
    if new_hash != st.session_state["doc_hash"]:
        st.session_state["uploaded_docs"] = new_docs
        st.session_state["doc_hash"] = new_hash
        # Force RAG reinit
        if "rag" in st.session_state:
            del st.session_state["rag"]

# Use uploaded docs if available
if st.session_state["uploaded_docs"]:
    documents = st.session_state["uploaded_docs"]
    st.success(f"✅ Using {len(documents)} uploaded document(s)")
    
    # Show document preview
    with st.expander("📄 Document preview"):
        for doc in documents:
            word_count = len(doc['content'].split())
            st.write(f"**{doc['title']}** - {word_count} words")
else:
    documents = SAMPLE_DOCS
    st.info("📄 Using default sample documents")

# Cached Hugging Face model loader
@st.cache_resource
def load_hf_model(name):
    from sentence_transformers import SentenceTransformer
    model_map = {
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"
    }
    model_name = model_map.get(name.lower(), name)
    with st.spinner(f"Loading embedding model: {model_name}..."):
        return SentenceTransformer(model_name)

hf = load_hf_model(hf_model) if use_embeddings else None

# Initialize RAG system
def init_rag(docs, hf_model, chunk_sz):
    rag = RAGSystem(docs, hf_model=hf_model, chunk_size_words=chunk_sz)
    if hf_model:
        with st.spinner("Computing embeddings..."):
            rag.precompute_embeddings(hf_model)
    return rag

# Use session state to avoid reinitializing unnecessarily
rag_key = f"rag_{len(documents)}_{chunk_size}_{use_embeddings}"
if "rag" not in st.session_state or st.session_state.get("rag_key") != rag_key:
    with st.spinner("Initializing RAG system..."):
        st.session_state["rag"] = init_rag(documents, hf, chunk_size)
        st.session_state["rag_key"] = rag_key

rag = st.session_state["rag"]
st.sidebar.success(f"✅ {len(rag.chunks)} chunks indexed")

# Tabs: Chat / Evaluation / Docs / About
tabs = st.tabs(["💬 Chat", "📊 Evaluation", "📚 Documents", "ℹ️ About"])

# ---------------- Debug View ----------------
with st.sidebar.expander("🔍 Debug Info"):
    st.write(f"**Total chunks:** {len(rag.chunks)}")
    st.write(f"**Embeddings computed:** {rag.chunks[0].embedding is not None if rag.chunks else False}")
    st.write(f"**Chunk size setting:** {chunk_size} words")
    if rag.chunks:
        avg_chunk_size = sum(len(c.text.split()) for c in rag.chunks) / len(rag.chunks)
        st.write(f"**Avg chunk size:** {avg_chunk_size:.0f} words")

# ---------------- Chat Tab ----------------
with tabs[0]:
    st.header("Ask a question")
    q = st.text_input("Question", placeholder="e.g., What is the price of Product A?")

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_btn = st.button("🔍 Get Answer", type="primary")
    
    if ask_btn and q:
        t0 = time.time()
        
        # Get query embedding if using semantic search
        q_emb = None
        if use_embeddings and hf:
            with st.spinner("Encoding query..."):
                q_emb = hf.encode(q, normalize_embeddings=True)

        # Retrieve relevant chunks
        with st.spinner("Retrieving relevant chunks..."):
            chunks = rag.retrieve(q_emb if q_emb is not None else q, top_k=top_k)

        if not chunks or all(score < 0.01 for _, score in chunks):
            st.warning("⚠️ No relevant context found for your question.")
            st.stop()

        # Show retrieved chunks
        with st.expander(f"🔍 Retrieved {len(chunks)} Chunks (click to expand)", expanded=False):
            for i, (c, s) in enumerate(chunks, 1):
                st.markdown(f"**[{i}] {c.doc_title}** (relevance: {s:.3f})")
                st.text(c.text[:600])
                st.divider()

        # Answer generation function
        def llm_gen(q, context):
            if not context.strip():
                return "No relevant context found."
            
            prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Be concise and accurate.

Context:
{context}

Question: {q}

Answer (be specific and concise):"""
            
            try:
                if GROQ_KEY:
                    from groq import Groq
                    client = Groq(api_key=GROQ_KEY)
                    with st.spinner(f"Generating answer with {groq_model}..."):
                        r = client.chat.completions.create(
                            model=groq_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1,
                            max_tokens=400
                        )
                    return r.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"❌ Groq API error: {e}")
                st.info("Falling back to extractive answer...")
            
            # Fallback to extractive
            return rag.generate_answer(q, chunks, llm_gen_fn=None)

        # Generate answer
        answer = rag.generate_answer(q, chunks, llm_gen_fn=llm_gen if GROQ_KEY else None)
        latency = (time.time() - t0) * 1000
        
        # Display results
        col1, col2 = st.columns([3, 1])
        with col2:
            st.metric("⏱️ Latency", f"{latency:.0f} ms")
        
        st.markdown("### 💡 Answer")
        st.markdown(f"> {answer}")
        
        if not GROQ_KEY:
            st.info("💡 Add a Groq API key in the sidebar for better generative answers!")

# ---------------- Evaluation Tab ----------------
with tabs[1]:
    st.header("📊 Evaluation on QA Dataset")
    st.write("Test the RAG system on a predefined set of questions.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        eval_btn = st.button("▶️ Run Evaluation", type="primary")
    
    if eval_btn:
        eval_set = load_eval_dataset()
        with st.spinner(f"Evaluating on {len(eval_set)} questions..."):
            results = run_evaluation(rag, eval_set, top_k=top_k)
        
        # Show aggregate metrics
        st.subheader("Aggregate Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Exact Match (EM)", f"{results['aggregate']['EM']:.2%}")
        with col2:
            st.metric("F1 Score", f"{results['aggregate']['F1']:.2%}")
        with col3:
            st.metric("Faithfulness", f"{results['aggregate']['Faith']:.2%}")
        
        # Show per-example results
        with st.expander("📋 Per-Example Results"):
            st.json(results)
        
        # Download button
        st.download_button(
            "⬇️ Download Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name=f"eval_results_{int(time.time())}.json",
            mime="application/json"
        )

# ---------------- Docs Tab ----------------
with tabs[2]:
    st.header("📚 Loaded Documents")
    for i, d in enumerate(documents, 1):
        with st.expander(f"{i}. {d['title']}"):
            word_count = len(d['content'].split())
            st.caption(f"Word count: {word_count}")
            st.text_area("Content preview", d["content"][:2000], height=200, disabled=True)

# ---------------- About Tab ----------------
with tabs[3]:
    st.header("ℹ️ About This System")
    st.markdown("""
### How It Works

**1. Document Chunking**
- Documents are split into chunks of ~{} words
- Uses sentence boundaries when possible
- Falls back to forced splits for unstructured text

**2. Semantic Retrieval** 
- Embedding models: BGE-small or MiniLM
- Cosine similarity for semantic search
- Fallback to keyword matching if embeddings disabled

**3. Answer Generation**
- **With Groq API**: Uses LLM (Llama 3.3 70B or others) for natural answers
- **Without API**: Extractive answers from retrieved chunks

### Tips for Best Results

✅ **Upload well-formatted documents** with clear sentences  
✅ **Use embeddings** for better semantic understanding  
✅ **Adjust chunk size** based on your document structure  
✅ **Add Groq API key** for high-quality generative answers  

### Limitations

- In-memory storage (suitable for <500 documents)
- Simple extractive fallback without API key
- No persistent storage across sessions

### Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: Sentence Transformers (HuggingFace)
- **LLM**: Groq Cloud (Llama 3.3 70B)
- **Retrieval**: Cosine similarity / BM25-like

""".format(chunk_size))
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit, HuggingFace, and Groq")
