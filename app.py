import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
def load_css():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #18191d 0%, #1e2028 50%, #16171b 100%);
    min-height: 100vh;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] { background: transparent; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.25); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.45); }

@keyframes shimmer {
    0%   { background-position: 0% center; }
    50%  { background-position: 100% center; }
    100% { background-position: 0% center; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-8px); }
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.78); }
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideInMsg {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Header ── */
.ctx-header {
    text-align: center;
    padding: 2.4rem 1rem 1.2rem;
    animation: fadeSlideDown 0.45s ease both;
}
.ctx-logo-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-bottom: 6px;
}
.ctx-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #94a3b8, #cbd5e1, #e2e8f0, #cbd5e1, #94a3b8);
    background-size: 280% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 6s linear infinite;
    letter-spacing: -1.5px;
    line-height: 1.1;
}
.ctx-subtitle {
    color: #52525b;
    font-size: 0.93rem;
    margin-top: 5px;
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* ── Upload zone ── */
.upload-zone {
    animation: fadeSlideUp 0.45s 0.1s ease both;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 20px;
    padding: 1.4rem 1.6rem 1.1rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.3s, box-shadow 0.3s;
    backdrop-filter: blur(10px);
}
.upload-zone:hover {
    border-color: rgba(148,163,184,0.35);
    box-shadow: 0 6px 32px rgba(0,0,0,0.45);
}
.upload-label {
    font-size: 0.77rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1.5px dashed rgba(148,163,184,0.25) !important;
    border-radius: 14px !important;
    transition: border-color 0.25s, background 0.25s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(148,163,184,0.55) !important;
    background: rgba(148,163,184,0.04) !important;
}
[data-testid="stFileUploader"] * { color: #71717a !important; }
[data-testid="stFileUploader"] svg { fill: #94a3b8 !important; }

/* ── Ingest button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #334155 0%, #475569 100%) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 18px !important;
    width: 100% !important;
    transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.2s !important;
    box-shadow: 0 3px 16px rgba(0,0,0,0.35) !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #3d4f63 0%, #536070 100%) !important;
    transform: scale(1.025) !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.45), 0 0 0 1px rgba(148,163,184,0.3) !important;
}
[data-testid="stButton"] > button:active { transform: scale(0.98) !important; }

/* ── Suggestion chip buttons ── */
div.st-key-chip_row [data-testid="stButton"] > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    border-radius: 20px !important;
    box-shadow: none !important;
    width: auto !important;
    transform: none !important;
}
div.st-key-chip_row [data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(148,163,184,0.45) !important;
    color: #cbd5e1 !important;
    transform: none !important;
    box-shadow: none !important;
}
div.st-key-chip_row [data-testid="stHorizontalBlock"] {
    justify-content: center !important;
    gap: 8px !important;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 14px 4px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 8px;
}
.status-pill.active {
    background: rgba(148,163,184,0.08);
    border: 1px solid rgba(148,163,184,0.2);
    color: #94a3b8;
}
.status-pill.empty {
    background: rgba(161,161,170,0.06);
    border: 1px solid rgba(161,161,170,0.2);
    color: #a1a1aa;
}
.pulse-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #94a3b8;
    animation: pulse-dot 1.8s ease-in-out infinite;
    flex-shrink: 0;
}
.pulse-dot.warn { background: #71717a; }

/* ── File pills ── */
.file-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(148,163,184,0.07);
    border: 1px solid rgba(148,163,184,0.18);
    color: #94a3b8;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    margin: 3px 3px 0 0;
    animation: fadeIn 0.3s ease;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    animation: slideInMsg 0.28s ease both;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    backdrop-filter: blur(8px);
}
[data-testid="stChatMessage"] p {
    color: #d4d4d8 !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 2.5rem 2rem 1.5rem;
    animation: fadeIn 0.5s ease both;
}
.empty-icon {
    font-size: 3.2rem;
    animation: float 3s ease-in-out infinite;
    display: block;
    margin-bottom: 1rem;
}
.empty-state h3 { color: #52525b; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.3rem; }
.empty-state p  { color: #3f3f46; font-size: 0.85rem; margin: 0; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    border-radius: 14px !important;
    color: #d4d4d8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(148,163,184,0.5) !important;
    box-shadow: 0 0 0 3px rgba(148,163,184,0.08) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3f3f46 !important; }

/* ── Misc ── */
.stAlert { border-radius: 12px !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
.stSpinner > div { border-top-color: #94a3b8 !important; }
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: #52525b !important;
    border-radius: 8px !important;
}
</style>""", unsafe_allow_html=True)

load_css()

# ─── Constants ──────────────────────────────────────────────────────────────────
PERSIST_DIR = "chroma-db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ─── RAG Helpers (logic unchanged from main.py / create_database.py) ────────────
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model_name="mistral-small-2603", temperature=0.1)


@st.cache_resource(show_spinner=False)
def get_vectorstore_client():
    """Cached Chroma client. Previously a plain Chroma(...) was opened from disk
    on every single question and on every status-badge render — that disk I/O
    was a large chunk of the response latency. Caching it means the DB is only
    opened once per session; ingest_pdf() below calls .clear() on this to force
    a fresh client after new documents are added."""
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding_model())


def load_vectorstore():
    """Return the cached vectorstore, or None if it's empty/unavailable."""
    try:
        vs = get_vectorstore_client()
        if vs._collection.count() == 0:
            return None
        return vs
    except Exception:
        return None


def get_ingested_filenames() -> set:
    """Return filenames already stored in ChromaDB (survives restarts)."""
    try:
        vs = get_vectorstore_client()
        results = vs._collection.get(include=["metadatas"])
        names = set()
        for meta in results.get("metadatas", []):
            if meta and "source_filename" in meta:
                names.add(meta["source_filename"])
        return names
    except Exception:
        return set()


def ingest_pdf(uploaded_file) -> tuple[bool, str, int]:
    """Chunk, embed and store a PDF. Skips if already in the DB."""
    try:
        if uploaded_file.name in get_ingested_filenames():
            return False, f"{uploaded_file.name} is already in the database.", 0

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        docs = PyPDFLoader(tmp_path).load()
        os.unlink(tmp_path)

        if not docs:
            return False, "Could not extract any text from the PDF.", 0

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        ).split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source_filename"] = uploaded_file.name

        Chroma.from_documents(
            documents=chunks,
            embedding=get_embedding_model(),
            persist_directory=PERSIST_DIR,
        )
        # Force a fresh cached client next time so the new chunks are visible —
        # this is now safe because get_vectorstore_client IS @st.cache_resource
        # (unlike the old load_vectorstore, which wasn't, and crashed here).
        get_vectorstore_client.clear()
        return True, f"{uploaded_file.name}", len(chunks)

    except Exception as e:
        return False, f"Error: {str(e)}", 0



RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the context provided. "
    "If you don't know the answer, just say you don't know. Do not try to make up an answer."
)


# Common abbreviation → full form map (domain-agnostic + CS/AI focused)
# Add more here as needed for your specific documents
_ABBREV_MAP = {
    # Search algorithms
    "bfs": "Breadth-First Search",
    "dfs": "Depth-First Search",
    "ucs": "Uniform Cost Search",
    "dls": "Depth-Limited Search",
    "ids": "Iterative Deepening Search",
    "iddfs": "Iterative Deepening Depth-First Search",
    "rbfs": "Recursive Best-First Search",
    # AI / ML
    "ai": "Artificial Intelligence",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "nn": "Neural Network",
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "lstm": "Long Short-Term Memory",
    "rl": "Reinforcement Learning",
    "llm": "Large Language Model",
    "rag": "Retrieval Augmented Generation",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Machine",
    "pca": "Principal Component Analysis",
    # CS general
    "oop": "Object-Oriented Programming",
    "api": "Application Programming Interface",
    "db": "Database",
    "sql": "Structured Query Language",
    "os": "Operating System",
    "cpu": "Central Processing Unit",
    "gpu": "Graphics Processing Unit",
    "ram": "Random Access Memory",
    "cli": "Command Line Interface",
    "gui": "Graphical User Interface",
    "oop": "Object Oriented Programming",
    # Data structures
    "bst": "Binary Search Tree",
    "avl": "AVL Tree",
    "dp": "Dynamic Programming",
}


def expand_abbreviations(query: str) -> str:
    """
    Replace known abbreviations in the query with their full forms so the
    embedding model can match them against document text that uses the full term.

    Only expands whole words (word-boundary match) to avoid false positives.
    Case-insensitive. Leaves unknown terms untouched.

    Examples:
        "explain UCS"        -> "explain Uniform Cost Search"
        "what is BFS"        -> "what is Breadth-First Search"
        "tell me about NLP"  -> "tell me about Natural Language Processing"
        "A* search"          -> "A* search"  (not an abbreviation, untouched)
    """
    import re as _re
    result = query
    for abbr, full in _ABBREV_MAP.items():
        # Match the abbreviation as a whole word, case-insensitive
        pattern = r'\b' + _re.escape(abbr) + r'\b'
        result = _re.sub(pattern, full, result, flags=_re.IGNORECASE)
    return result


def clean_query(query: str) -> str:
    """
    1. Expand abbreviations to full forms
    2. Strip conversational filler from the front
    so the embedding model focuses on the actual topic.
    """
    import re as _re

    # Step 0 — expand abbreviations first
    query = expand_abbreviations(query.strip())

    # Strip trailing punctuation noise
    query = query.rstrip(".!?,;").strip()

    # Pass 1 — remove leading interjections / softeners (repeatable)
    softeners = r'^(?:(?:okay|ok|so|like|well|hey|hmm+|umm+|alright|right)[\s,]*)*'
    query = _re.sub(softeners, '', query, flags=_re.IGNORECASE).strip()

    # Pass 2 — remove leading action/politeness phrases
    action = (
        r'^(?:'
        r'(?:can you|could you|would you|please)\s*|'
        r'(?:what|how) about\s*|'
        r'tell me (?:about|regarding|on|more about|something about)\s*|'
        r'explain\s*(?:to me\s*)?(?:what\s+is\s+|about\s+)?|'
        r'describe\s*(?:to me\s*)?|'
        r'what (?:is|are|was|were|do you know about|can you tell me about)\s*|'
        r'how (?:does|do|did|would)\s*|'
        r'give me (?:information|details|info|an overview|a summary) (?:on|about)\s*|'
        r'i (?:want|need|would like) (?:to know about|to understand|info on|information (?:on|about)|details (?:about|on))\s*|'
        r'can you (?:give me|tell me|show me|provide)\s*(?:information|details|info|an overview)?\s*(?:on|about|regarding)?\s*'
        r')+'
    )
    query = _re.sub(action, '', query, flags=_re.IGNORECASE).strip()

    # Fall back to original if cleaning stripped everything
    return query if len(query) > 3 else query


def retrieve_sources(query: str):
    """Clean the query first, then run vector search."""
    vs = load_vectorstore()
    if vs is None:
        return None
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    return retriever.invoke(clean_query(query))


def build_messages(query: str, source_docs: list):
    context = "\n\n".join(d.page_content for d in source_docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ])
    return prompt.format_prompt(context=context, question=query).to_messages()


# ─── Session State ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = list(get_ingested_filenames())

# ─── Header ─────────────────────────────────────────────────────────────────────
# IMPORTANT: this raw SVG string must have NO leading blank line and NO leading
# indentation on its lines. When it gets substituted into the f-string below,
# any leading whitespace on the line that follows it turns into a markdown
# "indented code block" trigger (blank line + 4-space indent), which is exactly
# what broke the <span class="ctx-title"> rendering — it was being displayed as
# literal text instead of parsed HTML. .strip() removes the stray leading/
# trailing newline so no blank line gets introduced downstream.
LOGO_SVG = """<svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="2" y="2" width="48" height="48" rx="14" fill="url(#bg)"/>
<rect x="2" y="2" width="48" height="48" rx="14" fill="url(#sheen)" opacity="0.5"/>
<path d="M17 16.5C17 14.567 18.567 13 20.5 13H29.5L35 18.5V35.5C35 37.433 33.433 39 31.5 39H20.5C18.567 39 17 37.433 17 35.5V16.5Z" fill="white" fill-opacity="0.92"/>
<path d="M29.5 13V16.5C29.5 17.6046 30.3954 18.5 31.5 18.5H35L29.5 13Z" fill="white" fill-opacity="0.55"/>
<rect x="20.5" y="23" width="10" height="1.8" rx="0.9" fill="url(#lineGrad)"/>
<rect x="20.5" y="27.2" width="10" height="1.8" rx="0.9" fill="url(#lineGrad)"/>
<rect x="20.5" y="31.4" width="6.5" height="1.8" rx="0.9" fill="url(#lineGrad)"/>
<defs>
<linearGradient id="bg" x1="2" y1="2" x2="50" y2="50" gradientUnits="userSpaceOnUse">
<stop stop-color="#8b5cf6"/><stop offset="0.55" stop-color="#6366f1"/><stop offset="1" stop-color="#4338ca"/>
</linearGradient>
<linearGradient id="sheen" x1="2" y1="2" x2="30" y2="30" gradientUnits="userSpaceOnUse">
<stop stop-color="white" stop-opacity="0.35"/><stop offset="1" stop-color="white" stop-opacity="0"/>
</linearGradient>
<linearGradient id="lineGrad" x1="20.5" y1="23" x2="30.5" y2="23" gradientUnits="userSpaceOnUse">
<stop stop-color="#7c3aed"/><stop offset="1" stop-color="#4f46e5"/>
</linearGradient>
</defs>
</svg>""".strip()

# Flush-left, no leading spaces on any line — avoids any markdown
# indented-code-block misinterpretation regardless of what gets substituted in.
header_html = (
    '<div class="ctx-header">'
    '<div class="ctx-logo-row">'
    f'{LOGO_SVG}'
    '<span class="ctx-title">DocuLens AI</span>'
    '</div>'
    '<div class="ctx-subtitle">LangChain-Powered RAG Document Intelligence</div>'
    '</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

# ─── Upload Zone ─────────────────────────────────────────────────────────────────
_, col_c, _ = st.columns([1, 4, 1])
with col_c:
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown('<div class="upload-label">📄 Upload PDF Documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#64748b;font-size:0.8rem;margin:-0.6rem 0 0.9rem 0;">'
        'Drop one or more PDFs below, then hit Ingest to add them to the knowledge base.'
        '</div>',
        unsafe_allow_html=True,
    )

    up_col, btn_col = st.columns([4, 1])
    with up_col:
        main_uploaded_files = st.file_uploader(
            "PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="main_uploader",
        )
    with btn_col:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        ingest_clicked = st.button("⚡ Ingest", use_container_width=True)

    # Status pill
    vs_check = load_vectorstore()
    if vs_check is not None:
        count = vs_check._collection.count()
        st.markdown(
            f'<div class="status-pill active"><span class="pulse-dot"></span>'
            f'Database active &nbsp;·&nbsp; {count} chunks indexed</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill empty"><span class="pulse-dot warn"></span>'
            'Database empty — ingest a PDF to get started</div>',
            unsafe_allow_html=True,
        )

    # Ingested file pills
    if st.session_state.ingested_files:
        pills_html = "".join(
            f'<span class="file-pill">📎 {f}</span>'
            for f in st.session_state.ingested_files
        )
        st.markdown(
            f'<div style="margin-top:8px;flex-wrap:wrap;">{pills_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)  # close upload-zone

    # Ingest logic
    if ingest_clicked:
        if not main_uploaded_files:
            st.warning("Select at least one PDF file first.")
        else:
            for uf in main_uploaded_files:
                with st.spinner(f"Processing {uf.name}…"):
                    ok, msg, n_chunks = ingest_pdf(uf)
                if ok:
                    st.session_state.ingested_files.append(uf.name)
                    st.success(f"✅ **{msg}** ingested — {n_chunks} chunks added")
                elif "already in the database" in msg:
                    st.warning(f"⚠️ {msg}")
                else:
                    st.error(f"❌ {msg}")
            st.rerun()

# ─── Chat Area ───────────────────────────────────────────────────────────────────
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
_, chat_col, _ = st.columns([1, 4, 1])

with chat_col:
    # Clear chat button (top-right corner)
    if st.session_state.messages:
        cl_col1, cl_col2 = st.columns([6, 1])
        with cl_col2:
            if st.button("🗑️", help="Clear chat history"):
                st.session_state.messages = []
                st.rerun()

    # Chat history in scrollable container
    chat_container = st.container(height=480)
    chip_question = None  # set below if a suggestion chip is clicked

    with chat_container:
        if not st.session_state.messages:
            # ── Empty state with clickable example chips ──
            st.markdown(
                '<div class="empty-state" style="padding-bottom:0;">'
                '<span class="empty-icon">🧩</span>'
                '<h3>Ask your first question</h3>'
                '<p>Upload a PDF above, then type anything below.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            suggestions = [
                "What is this document about?",
                "Summarize the key points",
                "What are the main findings?",
            ]
            with st.container(key="chip_row"):
                chip_cols = st.columns(len(suggestions))
                for col, suggestion in zip(chip_cols, suggestions):
                    with col:
                        if st.button(suggestion, key=f"chip_{suggestion}"):
                            chip_question = suggestion
        else:
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                sources = msg.get("sources", [])

                with st.chat_message(role):
                    st.markdown(content)

                    # Optional, collapsed by default — no pills cluttering the
                    # reply itself, just an expander for anyone who wants to
                    # check where the answer came from.
                    if role == "assistant" and sources:
                        with st.expander(f"📚 {len(sources)} source chunk(s)"):
                            for i, doc in enumerate(sources, 1):
                                fname = doc.metadata.get(
                                    "source_filename",
                                    doc.metadata.get("source", "Unknown"),
                                )
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**Chunk {i}** · `{fname}` · Page {page}")
                                st.markdown(
                                    f"<p style='font-size:0.8rem;color:#94a3b8;'>"
                                    f"{doc.page_content[:500]}…</p>",
                                    unsafe_allow_html=True,
                                )
                                if i < len(sources):
                                    st.divider()

    # ── Chat Input (also handles a clicked suggestion chip the same way) ──
    typed_input = st.chat_input("Ask anything about your documents…")
    question_to_process = typed_input or chip_question

    if question_to_process:
        st.session_state.messages.append({"role": "user", "content": question_to_process})

        with chat_container:
            with st.chat_message("assistant"):
                sources = retrieve_sources(question_to_process)

                if sources is None:
                    answer = "No documents found. Please upload and ingest a PDF first."
                    st.markdown(answer)
                    sources = []
                else:
                    # Streaming: tokens appear as soon as the model produces
                    # them instead of the UI sitting blank until the entire
                    # answer is ready. This is the main fix for the "feels
                    # slow" complaint — actual total latency barely changes,
                    # but the perceived wait drops a lot since text starts
                    # showing almost immediately.
                    messages = build_messages(question_to_process, sources)
                    llm = get_llm()

                    def token_stream():
                        for chunk in llm.stream(messages):
                            if chunk.content:
                                yield chunk.content

                    answer = st.write_stream(token_stream())

                    if sources:
                        with st.expander(f"📚 {len(sources)} source chunk(s)"):
                            for i, doc in enumerate(sources, 1):
                                fname = doc.metadata.get(
                                    "source_filename",
                                    doc.metadata.get("source", "Unknown"),
                                )
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**Chunk {i}** · `{fname}` · Page {page}")
                                st.markdown(
                                    f"<p style='font-size:0.8rem;color:#94a3b8;'>"
                                    f"{doc.page_content[:500]}…</p>",
                                    unsafe_allow_html=True,
                                )
                                if i < len(sources):
                                    st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()