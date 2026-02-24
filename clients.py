"""
clients.py
----------
Shared client initialization for Qdrant, Gemini, and HuggingFace embeddings.
All other modules import from here so config lives in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "hr_cvs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # fast, accurate, 384-dim, runs locally
VECTOR_DIM      = 384
GEMINI_MODEL    = "gemini-2.5-flash"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100

# ── Qdrant ────────────────────────────────────────────────────────────────────

def get_qdrant():
    """
    Returns a Qdrant client.
    - If QDRANT_URL is set in .env → connects to Qdrant Cloud (HTTP, no grpcio)
    - Otherwise → uses local on-disk storage in ./qdrant_db
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_key = os.getenv("QDRANT_API_KEY", "").strip()

    if qdrant_url:
        # Cloud mode
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key or None, prefer_grpc=False)
        print(f"🌐 Connected to Qdrant Cloud: {qdrant_url}")
    else:
        # Local on-disk mode
        qdrant_path = os.path.join(os.path.dirname(__file__), "qdrant_db")
        client = QdrantClient(path=qdrant_path)
        print(f"💾 Using local Qdrant at: {qdrant_path}")

    # Ensure collection exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"✅ Created collection: '{COLLECTION_NAME}'")

    return client


# ── HuggingFace Embeddings ────────────────────────────────────────────────────

_embedder = None   # module-level cache so model loads only once

def get_embedder():
    """Load the sentence-transformers model (cached after first call)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"📦 Loading embedding model: {EMBEDDING_MODEL} (first run downloads ~130MB)")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Embedding model ready")
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns list of float vectors."""
    embedder = get_embedder()
    vectors = embedder.encode(texts, show_progress_bar=len(texts) > 10, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


# ── Gemini ────────────────────────────────────────────────────────────────────

def get_gemini():
    """Return a configured Gemini GenerativeModel."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)
