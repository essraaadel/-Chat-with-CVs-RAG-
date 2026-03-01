

"""
app.py
------
Streamlit HR CV Assistant
  - HuggingFace embeddings (local, free)
  - Gemini 2.5 Flash LLM (free tier)
  - Qdrant vector DB (local or cloud)
  - Source chunk panel shown alongside every answer

Run:first: venv\Scripts\activate
second :python -m streamlit run app.py
"""

import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

from clients import (
    get_qdrant, get_gemini, embed_texts,
    COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
)
from ingest import extract_text, chunk_text
from chat import rag, retrieve
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR CV Assistant", page_icon="🧑‍💼", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

/* Brand header */
.brand {
    background: linear-gradient(120deg, #060d1a 0%, #0a1628 60%, #071020 100%);
    border: 1px solid #152540;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.brand-icon  { font-size: 2.4rem; }
.brand-title { color: #e2ecff; font-size: 1.75rem; font-weight: 700; margin: 0; }
.brand-meta  { color: #3d6090; font-size: 0.82rem; margin: 0; }
.brand-badge {
    margin-left: auto;
    background: #0d2040;
    border: 1px solid #1a3a6a;
    border-radius: 20px;
    padding: 4px 14px;
    color: #4a88d8;
    font-size: 0.78rem;
    font-weight: 500;
}

/* Chat bubbles */
.bubble-user {
    background: #102040;
    color: #c8deff;
    padding: 0.8rem 1.1rem;
    border-radius: 16px 16px 4px 16px;
    margin: 5px 0 5px 18%;
    font-size: 0.94rem;
    line-height: 1.55;
    border: 1px solid #1a3560;
}

/* Bot bubble wrapper — targets the Streamlit block inside */
.bubble-bot-outer + div,
div:has(> .bubble-bot-outer) {
    background: #07111f;
    border: 1px solid #0f2240;
    border-radius: 16px 16px 16px 4px;
    margin: 5px 18% 5px 0;
    padding: 0.8rem 1.1rem;
    font-size: 0.94rem;
    line-height: 1.65;
}

/* Style markdown elements inside bot bubble */
[data-testid="stMarkdownContainer"] p  { color: #b0c8e8; margin: 0.3rem 0; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #e2ecff; margin: 0.6rem 0 0.3rem 0; font-family: "Outfit", sans-serif; }
[data-testid="stMarkdownContainer"] strong { color: #6aabff; }
[data-testid="stMarkdownContainer"] em     { color: #8ab8e8; }
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol     { color: #b0c8e8; padding-left: 1.2rem; margin: 0.4rem 0; }
[data-testid="stMarkdownContainer"] li     { margin: 0.2rem 0; }
[data-testid="stMarkdownContainer"] code   { background: #0d2040; color: #5aadff; padding: 2px 6px; border-radius: 4px; font-family: "JetBrains Mono", monospace; font-size: 0.85em; }
[data-testid="stMarkdownContainer"] pre    { background: #060e1c; border: 1px solid #1a3050; border-radius: 8px; padding: 0.8rem; overflow-x: auto; }
[data-testid="stMarkdownContainer"] pre code { background: none; padding: 0; color: #7dc8ff; }
[data-testid="stMarkdownContainer"] table  { border-collapse: collapse; width: 100%; margin: 0.6rem 0; }
[data-testid="stMarkdownContainer"] th     { background: #0d2040; color: #4a9eff; padding: 0.4rem 0.8rem; border: 1px solid #1a3560; font-size: 0.85rem; }
[data-testid="stMarkdownContainer"] td     { color: #8ab8e8; padding: 0.35rem 0.8rem; border: 1px solid #0f2035; font-size: 0.84rem; }
[data-testid="stMarkdownContainer"] tr:nth-child(even) td { background: #060e1c; }
[data-testid="stMarkdownContainer"] blockquote { border-left: 3px solid #2563a8; padding-left: 0.8rem; color: #6a8ab0; margin: 0.4rem 0; font-style: italic; }
[data-testid="stMarkdownContainer"] hr     { border: none; border-top: 1px solid #1a3050; margin: 0.6rem 0; }

/* Source chunk cards */
.chunk-wrap {
    background: #060e1c;
    border: 1px solid #102035;
    border-left: 3px solid #2563a8;
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.7rem;
}
.chunk-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.45rem;
}
.chunk-name   { color: #3a82f0; font-weight: 600; font-size: 0.83rem; }
.chunk-badge  { background: #0d2040; color: #2a62b8; padding: 2px 9px; border-radius: 12px; font-size: 0.72rem; }
.chunk-sub    { color: #1e3a60; font-size: 0.7rem; margin-bottom: 0.4rem; }
.chunk-text   { color: #4a6a90; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; white-space: pre-wrap; word-break: break-word; line-height: 1.5; }

/* Candidate pills */
.pill { display:inline-block; background:#0a1c34; color:#3a82f0; padding:3px 11px; border-radius:20px; font-size:0.79rem; margin:2px 2px; border:1px solid #13304a; }

/* Stats */
.stat { background:#07111f; border:1px solid #0f2035; border-radius:10px; padding:0.7rem; text-align:center; }
.stat-n { font-size:1.7rem; font-weight:700; color:#3a82f0; }
.stat-l { font-size:0.72rem; color:#1e3860; margin-top:1px; }

/* Suggestion buttons */
.stButton > button {
    background: #07111f !important;
    color: #3a6090 !important;
    border: 1px solid #102035 !important;
    border-radius: 8px !important;
    font-size: 0.81rem !important;
    padding: 0.38rem 0.8rem !important;
    font-family: 'Outfit', sans-serif !important;
    text-align: left !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #0d1e35 !important;
    color: #6aabff !important;
    border-color: #2563a8 !important;
}

/* Empty state */
.empty-state {
    color: #1a3050;
    font-size: 0.86rem;
    padding: 1.8rem;
    background: #050c18;
    border-radius: 10px;
    border: 1px dashed #0f2035;
    text-align: center;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand">
  <div class="brand-icon">🧑‍💼</div>
  <div>
    <p class="brand-title">HR CV Assistant</p>
    <p class="brand-meta">Gemini 2.5 Flash · HuggingFace Embeddings · Qdrant</p>
  </div>
  <div class="brand-badge">RAG Pipeline</div>
</div>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
for k, v in [("messages", []), ("last_chunks", []), ("qdrant", None), ("gemini", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource(show_spinner="Loading models...")
def load_clients():
    """Load clients once and cache for the session."""
    return get_qdrant(), get_gemini()


try:
    if st.session_state.qdrant is None:
        st.session_state.qdrant, st.session_state.gemini = load_clients()
except Exception as e:
    st.error(f"❌ Failed to load clients: {e}")
    st.stop()


def get_candidates() -> dict[str, int]:
    try:
        results, _ = st.session_state.qdrant.scroll(
            collection_name=COLLECTION_NAME, with_payload=True, limit=10_000
        )
        counts: dict[str, int] = {}
        for p in results:
            n = p.payload["candidate"]
            counts[n] = counts.get(n, 0) + 1
        return counts
    except Exception:
        return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Upload CVs")
    uploaded = st.file_uploader("PDF · DOCX · TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded and st.button("⚡ Index CVs", use_container_width=True):
        qdrant = st.session_state.qdrant

        # Get already-indexed filenames
        existing, _ = qdrant.scroll(collection_name=COLLECTION_NAME, with_payload=True, limit=10_000)
        indexed_files = {p.payload["filename"] for p in existing}

        prog = st.progress(0)
        total_added = 0

        for i, f in enumerate(uploaded):
            prog.progress((i + 1) / len(uploaded))
            if f.name in indexed_files:
                continue

            suffix = os.path.splitext(f.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            try:
                text = extract_text(tmp_path)
                if not text:
                    st.warning(f"⚠️ No text in {f.name}")
                    continue

                # Save to /data
                data_dir = os.path.join(os.path.dirname(__file__), "data")
                os.makedirs(data_dir, exist_ok=True)
                with open(os.path.join(data_dir, f.name), "wb") as out, open(tmp_path, "rb") as src:
                    out.write(src.read())

                candidate = os.path.splitext(f.name)[0]
                chunks    = chunk_text(text)
                embeds    = embed_texts(chunks)

                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={
                            "text": chunk, "candidate": candidate,
                            "filename": f.name, "chunk_index": idx,
                            "total_chunks": len(chunks)
                        }
                    )
                    for idx, (chunk, emb) in enumerate(zip(chunks, embeds))
                ]
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                total_added += len(points)
            finally:
                os.unlink(tmp_path)

        prog.empty()
        st.success(f"✅ {total_added} chunks indexed!") if total_added else st.info("ℹ️ Already indexed")
        st.rerun()

    st.markdown("---")
    st.markdown("### 👥 Candidates")
    candidates = get_candidates()

    if candidates:
        for name in sorted(candidates):
            st.markdown(f'<span class="pill">👤 {name}</span>', unsafe_allow_html=True)
        st.markdown(f"<br><small style='color:#1a3050'>{len(candidates)} candidates · {sum(candidates.values())} chunks</small>", unsafe_allow_html=True)
        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat"><div class="stat-n">{len(candidates)}</div><div class="stat-l">candidates</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat"><div class="stat-n">{sum(candidates.values())}</div><div class="stat-l">chunks</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("<small style='color:#1a3050'>No CVs indexed yet</small>", unsafe_allow_html=True)

    st.markdown("")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_chunks = []
        st.rerun()


# ── Main Columns ──────────────────────────────────────────────────────────────
chat_col, src_col = st.columns([3, 2], gap="medium")

# ── Chat Column ───────────────────────────────────────────────────────────────

def run_rag(prompt: str):
    """Run the full RAG pipeline for a given prompt and update session state."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not candidates:
        answer = "⚠️ No CVs indexed yet. Upload some using the sidebar."
        st.session_state.last_chunks = []
    else:
        with st.spinner("Searching CVs and generating answer..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            result  = rag(prompt, st.session_state.qdrant, st.session_state.gemini, history)
            answer  = result["answer"]
            st.session_state.last_chunks = result["chunks"]

    st.session_state.messages.append({"role": "assistant", "content": answer})


with chat_col:
    st.markdown("### 💬 Chat")

    # Suggested questions — only shown when chat is empty
    if not st.session_state.messages:
        st.markdown("<small style='color:#1a3050'>Suggested questions:</small>", unsafe_allow_html=True)
        for sug in [
            "Who has the most ML experience?",
            "Which candidates know Python and SQL?",
            "Compare candidates for a senior backend role",
            "Who has a Computer Science degree?",
            "Rank all candidates for a data science role",
        ]:
            if st.button(sug, key=f"s_{sug}"):
                run_rag(sug)   # ← now actually calls the pipeline
                st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # User bubble — plain text, right-aligned HTML
            safe_content = msg["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            st.markdown(
                f'<div class="bubble-user">🧑 {safe_content}</div>',
                unsafe_allow_html=True
            )
        else:
            # Bot response — use st.container with CSS class for styling + native markdown
            with st.container():
                st.markdown('<div class="bubble-bot-outer">', unsafe_allow_html=True)
                st.markdown(f"🤖 {msg['content']}")
                st.markdown('</div>', unsafe_allow_html=True)

    # Manual chat input
    if prompt := st.chat_input("Ask about your candidates..."):
        run_rag(prompt)   # ← same function, consistent behavior
        st.rerun()


# ── Source Panel ──────────────────────────────────────────────────────────────
with src_col:
    st.markdown("### 📎 Source Chunks")
    chunks = st.session_state.last_chunks

    if not chunks:
        st.markdown("""
        <div class="empty-state">
            Source chunks will appear here<br>after you ask a question.<br><br>
            <span style='font-size:1.4rem'>🔍</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"<small style='color:#1a3050'>{len(chunks)} chunks · sorted by relevance score</small>", unsafe_allow_html=True)
        st.markdown("")


        for i, chunk in enumerate(chunks, 1):
            # Safety: ensure text is always a plain string before rendering
            raw_text = chunk["text"]
            if isinstance(raw_text, dict):
               raw_text = raw_text.get("text", str(raw_text))
            safe = str(raw_text).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

            # Safety: ensure section is always a string
            section = chunk.get("section", "general")
            if isinstance(section, dict):
                section = "general"

            st.markdown(f"""
            <div class="chunk-wrap">
               <div class="chunk-meta">
                 <span class="chunk-name">👤 {chunk['candidate']}</span>
                 <span class="chunk-badge">score {chunk['score']}</span>
               </div>
               <div class="chunk-sub">📂 {section} · chunk {chunk['chunk_index']} · {chunk['filename']}</div>
               <div class="chunk-text">{safe}</div>
           </div>""", unsafe_allow_html=True)
