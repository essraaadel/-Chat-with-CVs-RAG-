
# 🧑‍💼 HR CV Assistant — RAG-Powered Candidate Chat

> Upload your candidate CVs and chat with them using AI. Ask natural questions, get grounded answers, and see exactly which CV excerpts backed each response.

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-dc244c?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-LLM-4285f4?style=flat-square&logo=google&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-ffd21e?style=flat-square&logo=huggingface&logoColor=black)

---

## ✨ What It Does

You're an HR manager with a pile of CVs. Instead of reading each one manually, you drop them into this tool and **just ask questions**:

- *"Who has the most experience in machine learning?"*
- *"Which candidates know both Python and SQL?"*
- *"Compare the top 3 candidates for a senior backend role"*
- *"Who has a Computer Science degree and leadership experience?"*

The assistant answers based **only on your actual CVs** — no hallucinations — and always shows you the **exact text chunks** it used to form the answer.

---

## 🏗️ Architecture

```
Your CVs (PDF/DOCX/TXT)
        │
        ▼
  [ ingest.py ]
  Extract text → Chunk → Embed (HuggingFace local) → Store in Qdrant
        │
        ▼
  [ chat.py / app.py ]
  Your Question → Embed → Search Qdrant → Top-K chunks
                                               │
                                               ▼
                                     Gemini 2.5 Flash
                                               │
                                               ▼
                                    Answer + Source Chunks
```

### Tech Stack

| Layer | Tool | Why |
|---|---|---|
| **Embeddings** | `BAAI/bge-small-en-v1.5` (HuggingFace) | Free, local, no API key, 384-dim |
| **Vector DB** | Qdrant (local or cloud) | Production-grade, HTTP mode (no grpcio) |
| **LLM** | Gemini 2.5 Flash | Fast, free tier, excellent reasoning |
| **CV Parsing** | PyMuPDF + python-docx | PDF and Word support |
| **UI** | Streamlit | Simple web interface |

---

## 📁 Project Structure

```
hr-cv-rag/
├── data/               ← Drop your CV files here (PDF, DOCX, TXT)
├── qdrant_db/          ← Auto-created local vector database
│
├── clients.py          ← Shared setup: Qdrant, Gemini, HuggingFace
├── ingest.py           ← Parse CVs → chunk → embed → store in Qdrant
├── chat.py             ← RAG pipeline + terminal chat interface
├── app.py              ← Streamlit web UI
│
├── .env.example        ← Environment variable template
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hr-cv-rag.git
cd hr-cv-rag
```

### 2. Set up Python environment

> ⚠️ **Python 3.11 or 3.12 is strongly recommended.** Python 3.13+ may have dependency issues with some packages.

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The HuggingFace embedding model (`BAAI/bge-small-en-v1.5`, ~130MB) will download automatically on first run.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
# Required — get free key at: https://aistudio.google.com/apikey
GEMINI_API_KEY=your-gemini-api-key-here

# Optional — leave empty to use local Qdrant (recommended for starting out)
QDRANT_URL=
QDRANT_API_KEY=
```

### 5. Add CVs and index them

Drop your CV files (PDF, DOCX, or TXT) into the `/data` folder, then run:

```bash
python ingest.py
```

You'll see output like:
```
📄 john_doe.pdf
   ✅ 4,821 characters
   ✂️  11 chunks | 🔢 embeddings done
   💾 11 vectors stored

✅ Done! 11 new chunks added.
📊 Total vectors in DB: 11
```

### 6. Start chatting

**Web UI (recommended):**
```bash
streamlit run app.py
```

**Terminal:**
```bash
python chat.py
```

---

## 💬 Usage Examples

### Web UI
Upload CVs directly via drag & drop in the sidebar — no need to use the `/data` folder manually. The source chunks panel on the right updates after every answer.

### Terminal

```bash
# Interactive chat session
python chat.py

# Single question
python chat.py "who has the most Python experience?"
```

### Managing candidates

```bash
python ingest.py list              # list all indexed candidates
python ingest.py delete john_doe   # remove a specific candidate
python ingest.py                   # re-index (skips already indexed files)
```

---

## ⚙️ Configuration

All settings live at the top of `clients.py`:

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace model for embeddings |
| `VECTOR_DIM` | `384` | Must match the embedding model |
| `GEMINI_MODEL` | `gemini-2.5-flash-preview-04-17` | Swap to `gemini-1.5-pro` for heavier tasks |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |

In `chat.py`:

| Setting | Default | Description |
|---|---|---|
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `SCORE_THRESHOLD` | `0.3` | Minimum cosine similarity to include a chunk |

---

## ☁️ Qdrant Cloud (Optional)

By default the app uses a **local on-disk Qdrant database** stored in `./qdrant_db`. This is perfect for local use.

For a hosted/shared setup, you can use [Qdrant Cloud](https://cloud.qdrant.io) which offers a **free 1GB cluster**:

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster
3. Copy your cluster URL and API key into `.env`:

```env
QDRANT_URL=https://xxxx.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

---

## 🔑 API Keys Summary

| Key | Required | Where to get it | Cost |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier available |
| `QDRANT_API_KEY` | ❌ No (only for cloud) | [cloud.qdrant.io](https://cloud.qdrant.io) | Free 1GB cluster |

HuggingFace embeddings run **locally** — no API key needed.

---

## 🛡️ How It Stays Grounded

The system prompt explicitly instructs the LLM to answer **only from the provided CV excerpts**. If a candidate's CV doesn't mention a skill, the assistant will say so rather than guess. Every answer in the UI is accompanied by the exact source chunks with their similarity scores so you can verify the basis of every claim.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT
