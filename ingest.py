"""
ingest.py
---------
Load CVs → extract text → chunk → embed (HuggingFace local) → store in Qdrant.

Supported: PDF, DOCX, TXT

Usage:
  python ingest.py                  → index all CVs in /data
  python ingest.py list             → list indexed candidates
  python ingest.py delete <name>    → remove a candidate
"""

import os
import uuid
import fitz        # PyMuPDF
import docx
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from clients import get_qdrant, embed_texts, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Text Extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def extract_text_from_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs).strip()


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    fn  = {".pdf": extract_text_from_pdf, ".docx": extract_text_from_docx, ".txt": extract_text_from_txt}
    if ext not in fn:
        print(f"  ⚠️  Unsupported format: {ext}")
        return ""
    return fn[ext](path)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Overlapping chunks that prefer to break at newlines (preserves CV sections)."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            bp = text.rfind("\n", start, end)
            if bp > start:
                end = bp
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_indexed_filenames(qdrant) -> set[str]:
    results, _ = qdrant.scroll(collection_name=COLLECTION_NAME, with_payload=True, limit=10_000)
    return {p.payload["filename"] for p in results}


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest_cvs():
    os.makedirs(DATA_DIR, exist_ok=True)

    qdrant  = get_qdrant()
    indexed = get_indexed_filenames(qdrant)
    files   = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".docx", ".txt"))]

    if not files:
        print("📭 No CV files found in /data. Add PDFs, DOCXs, or TXTs and run again.")
        return

    print(f"📂 {len(files)} file(s) found\n")
    total_added = 0

    for filename in files:
        path           = os.path.join(DATA_DIR, filename)
        candidate_name = os.path.splitext(filename)[0]

        print(f"📄 {filename}")

        if filename in indexed:
            print("   ℹ️  Already indexed — skipping.\n")
            continue

        text = extract_text(path)
        if not text:
            print("   ⚠️  No text extracted — skipping.\n")
            continue

        print(f"   ✅ {len(text):,} characters")

        chunks     = chunk_text(text)
        embeddings = embed_texts(chunks)   # local HuggingFace model, no API key needed

        print(f"   ✂️  {len(chunks)} chunks | 🔢 embeddings done")

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text":         chunk,
                    "candidate":    candidate_name,
                    "filename":     filename,
                    "chunk_index":  i,
                    "total_chunks": len(chunks),
                }
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        total_added += len(points)
        print(f"   💾 {len(points)} vectors stored\n")

    print(f"✅ Done! {total_added} new chunks added.")
    print(f"📊 Total vectors in DB: {qdrant.count(COLLECTION_NAME).count}")


def list_candidates():
    qdrant = get_qdrant()
    results, _ = qdrant.scroll(collection_name=COLLECTION_NAME, with_payload=True, limit=10_000)

    if not results:
        print("📭 No candidates indexed yet.")
        return

    counts: dict[str, int] = {}
    for p in results:
        n = p.payload["candidate"]
        counts[n] = counts.get(n, 0) + 1

    print(f"\n👥 {len(counts)} candidates:")
    for name, n in sorted(counts.items()):
        print(f"   • {name}  ({n} chunks)")


def delete_candidate(name: str):
    qdrant = get_qdrant()
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[FieldCondition(key="candidate", match=MatchValue(value=name))])
    )
    print(f"🗑️  Deleted all chunks for: {name}")


if __name__ == "__main__":
    import sys
    cmds = {"list": list_candidates}
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "list":
            list_candidates()
        elif cmd == "delete" and len(sys.argv) > 2:
            delete_candidate(sys.argv[2])
        else:
            print("Usage:\n  python ingest.py\n  python ingest.py list\n  python ingest.py delete <name>")
    else:
        ingest_cvs()
