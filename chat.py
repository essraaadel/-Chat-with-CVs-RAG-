"""
chat.py
-------
RAG chat engine: Qdrant retrieval + Gemini 2.5 Flash answers + source chunks shown.

Usage:
  python chat.py                        → interactive terminal chat
  python chat.py "who knows Python?"    → single question
"""

import os
from clients import get_qdrant, get_gemini, embed_query, COLLECTION_NAME

TOP_K = 5
SCORE_THRESHOLD = 0.3   # discard chunks below this cosine similarity

SYSTEM_PROMPT = """You are an expert HR assistant helping a recruiter evaluate candidates.

Rules:
- Answer ONLY from the CV excerpts provided — never invent information
- Always mention the candidate's name when referencing their data
- Be concise but complete; use structure when comparing multiple candidates
- If relevant info is missing from the excerpts, say so clearly
- End with a short recommendation when the question involves selection or ranking"""


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, qdrant, top_k: int = TOP_K) -> list[dict]:
    """Embed question and fetch top-K chunks from Qdrant."""
    vector = embed_query(question)

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=SCORE_THRESHOLD
    )

    return [
        {
            "text":        h.payload["text"],
            "candidate":   h.payload["candidate"],
            "filename":    h.payload["filename"],
            "chunk_index": h.payload["chunk_index"],
            "score":       round(h.score, 3),
        }
        for h in hits
    ]


def build_context(chunks: list[dict]) -> str:
    """Group chunks by candidate into labeled sections."""
    by_candidate: dict[str, list] = {}
    for c in chunks:
        by_candidate.setdefault(c["candidate"], []).append(c)

    parts = []
    for candidate, items in by_candidate.items():
        parts.append(f"{'='*50}\nCANDIDATE: {candidate}\n{'='*50}")
        for item in items:
            parts.append(item["text"])
            parts.append("")
    return "\n".join(parts)


# ── Generation ────────────────────────────────────────────────────────────────

def ask_gemini(question: str, context: str, gemini_model, history: list[dict] = None) -> str:
    """Send context + question to Gemini 2.5 Flash and return the answer."""

    # Build full prompt
    history_text = ""
    if history:
        for turn in history[-4:]:   # last 2 exchanges
            role = "Recruiter" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n\n"

    prompt = f"""{SYSTEM_PROMPT}

{"Previous conversation:" + chr(10) + history_text if history_text else ""}
CV Excerpts:
{context}

---

Recruiter question: {question}"""

    response = gemini_model.generate_content(prompt)
    return response.text


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def rag(question: str, qdrant, gemini_model, history: list[dict] = None) -> dict:
    """Retrieve → build context → generate. Returns {answer, chunks}."""
    chunks = retrieve(question, qdrant)

    if not chunks:
        return {
            "answer": "⚠️ No relevant CV content found. Try rephrasing or index more CVs.",
            "chunks": []
        }

    context = build_context(chunks)
    answer  = ask_gemini(question, context, gemini_model, history)

    return {"answer": answer, "chunks": chunks}


# ── Source Display ────────────────────────────────────────────────────────────

def print_sources(chunks: list[dict]):
    """Print a formatted box for each source chunk."""
    if not chunks:
        return

    print(f"\n{'─'*62}")
    print(f"📎 SOURCE CHUNKS  ({len(chunks)} retrieved, ranked by relevance)\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"  ┌─ [{i}] candidate: {chunk['candidate']}  "
              f"chunk: {chunk['chunk_index']}  score: {chunk['score']}")
        for line in chunk["text"].split("\n"):
            line = line.strip()
            if not line:
                continue
            # Simple word-wrap at 68 chars
            while len(line) > 68:
                print(f"  │  {line[:68]}")
                line = line[68:]
            print(f"  │  {line}")
        print(f"  └{'─'*60}\n")


# ── Terminal Chat ─────────────────────────────────────────────────────────────

def interactive_chat():
    qdrant       = get_qdrant()
    gemini_model = get_gemini()

    count = qdrant.count(COLLECTION_NAME).count
    if count == 0:
        print("❌ No CVs indexed. Run `python ingest.py` first.")
        return

    print(f"\n🤖 HR CV Assistant  |  Gemini 2.5 Flash  |  {count} chunks in Qdrant")
    print("Type 'quit' to exit\n" + "─"*60)

    history     = []
    last_chunks = []

    while True:
        try:
            question = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        result      = rag(question, qdrant, gemini_model, history)
        last_chunks = result["chunks"]

        print(f"\n🤖 Gemini:\n{result['answer']}")
        print_sources(last_chunks)

        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant",  "content": result["answer"]})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        qdrant   = get_qdrant()
        gemini   = get_gemini()
        if qdrant.count(COLLECTION_NAME).count == 0:
            print("❌ No CVs indexed. Run `python ingest.py` first.")
        else:
            result = rag(question, qdrant, gemini)
            print(f"\n🤖 {result['answer']}\n")
            print_sources(result["chunks"])
    else:
        interactive_chat()
