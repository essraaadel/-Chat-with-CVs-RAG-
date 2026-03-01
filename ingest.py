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


# ── Section Detection ────────────────────────────────────────────────────────

# All known CV section headers — extend this list freely with your own
CV_SECTION_HEADERS = [
    # Summary / Profile
    "summary", "objective", "profile", "about me", "about", "personal statement",
    "professional summary", "career objective", "overview", "introduction",
    # Experience
    "experience", "work experience", "employment", "employment history",
    "professional experience", "career history", "work history",
    "internship", "internships", "job experience", "practical experience","Training Experience","Professional Training","Training"
    # Education
    "education", "academic background", "qualifications", "academic qualifications",
    "educational background", "degrees", "academic history", "studies",
    # Skills
    "skills", "technical skills", "core competencies", "competencies",
    "key skills", "areas of expertise", "expertise", "technologies",
    "tools", "programming languages", "soft skills", "hard skills",
    "languages", "technical expertise",
    # Projects
    "projects", "personal projects", "academic projects",
    "key projects", "notable projects", "portfolio", "works",
    # Certifications
    "certifications", "certificates", "licenses", "accreditations",
    "professional development", "courses", "training", "credentials",
    # Achievements
    "achievements", "accomplishments", "awards", "honors",
    "recognition", "publications", "research", "patents",
    # Other
    "references", "volunteer", "volunteering", "extracurricular",
    "interests", "hobbies", "activities", "leadership", "clubs",
    "contact", "personal information", "personal details", "links",
]


def is_section_header(line: str) -> bool:
    """
    Detects if a line is a CV section header using 3 rules:
    1. Matches a known keyword (flexible — handles colons, extra words, mixed case)
    2. Short ALL CAPS line  e.g. "WORK EXPERIENCE"
    3. Short line ending with colon  e.g. "Projects:"
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False

    lower = stripped.lower().rstrip(":")   # normalize: remove trailing colon, lowercase

    # Rule 1 — known keyword: exact or starts-with match
    for header in CV_SECTION_HEADERS:
        if lower == header:
            return True
        if lower.startswith(header) and len(stripped) < 45:
            return True

    # Rule 2 — short ALL CAPS line (min 3 chars to avoid false positives)
    if stripped.isupper() and 3 < len(stripped) < 45:
        return True

    # Rule 3 — short line ending with colon
    if stripped.endswith(":") and len(stripped) < 45:
        return True

    return False


def normalize_header(line: str) -> str:
    """Turn a raw header line into a clean readable label."""
    return line.strip().rstrip(":").strip().title()


# ── Smart Section-Based Chunker ───────────────────────────────────────────────

def chunk_text(text: str) -> list[dict]:
    """
    PRIMARY strategy: split CV at detected section boundaries.
    Each chunk = one CV section (Education, Skills, Experience, etc.)

    Returns list of dicts: [{"text": "...", "section": "Education"}, ...]

    Falls back to fixed-size chunking if fewer than 2 sections are detected.
    Large sections are further split into sub-chunks with (part N) labels.
    """
    lines = text.split("\n")

    sections: list[dict] = []
    current_section = "General"
    current_lines:  list[str] = []

    for line in lines:
        if is_section_header(line):
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({"text": content, "section": current_section})
            current_section = normalize_header(line)
            current_lines   = []
        else:
            current_lines.append(line)

    # Capture last section
    content = "\n".join(current_lines).strip()
    if content:
        sections.append({"text": content, "section": current_section})

    # ── Fallback if no sections detected ─────────────────────────────────────
    if len(sections) < 2:
        print("   ⚠️  No sections detected — using fixed-size fallback")
        return _fixed_size_chunks(text)

    print(f"   📂 Detected sections: {[s['section'] for s in sections]}")

    # ── Split oversized sections ──────────────────────────────────────────────
    final: list[dict] = []
    for sec in sections:
        if len(sec["text"]) <= CHUNK_SIZE:
            final.append(sec)
        else:
            sub_chunks = _fixed_size_chunks(sec["text"])
            for i, sub in enumerate(sub_chunks):
                final.append({
                    "text":    sub["text"],
                    "section": f"{sec['section']} (part {i + 1})"
                })

    return final


def _fixed_size_chunks(text: str) -> list[dict]:
    """Fixed-size fallback chunker. Returns same dict format for consistency."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            bp = text.rfind("\n", start, end)
            if bp > start:
                end = bp
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "section": "general"})
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

        chunk_dicts  = chunk_text(text)                          # list of {text, section}
        chunk_texts  = [c["text"] for c in chunk_dicts]           # plain text for embedding
        embeddings   = embed_texts(chunk_texts)

        print(f"   ✂️  {len(chunk_dicts)} chunks | 🔢 embeddings done")

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text":         cd["text"],
                    "section":      cd["section"],                 # ← section label
                    "candidate":    candidate_name,
                    "filename":     filename,
                    "chunk_index":  i,
                    "total_chunks": len(chunk_dicts),
                }
            )
            for i, (cd, emb) in enumerate(zip(chunk_dicts, embeddings))
        ]

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        total_added += len(points)
        print(f"   💾 {len(points)} vectors stored\n")

    print(f"✅ Done! {total_added} new chunks added.")
    print(f"📊 Total vectors in DB: {qdrant.count(COLLECTION_NAME).count}")


def reindex_all():
    """
    Delete ALL vectors from Qdrant and re-index every CV in /data from scratch.
    Use this when you upgrade the chunking strategy to avoid stale/mixed data.
    """
    qdrant = get_qdrant()

    # Step 1 — wipe the collection
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Deleted collection '{COLLECTION_NAME}'")

    # Step 2 — recreate it fresh
    from qdrant_client.models import Distance, VectorParams
    from clients import VECTOR_DIM
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )
    print(f"✅ Fresh collection created")

    # Step 3 — re-index everything
    print()
    ingest_cvs()


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
        elif cmd == "reindex":
            reindex_all()
        else:
            print("Usage:\n  python ingest.py\n  python ingest.py list\n  python ingest.py delete <name>\n  python ingest.py reindex")
    else:
        ingest_cvs()
