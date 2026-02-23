# RAG Pipeline Classic

A Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, Pinecone, and OpenAI. Ingest PDF/text documents, store embeddings in Pinecone, and ask questions with cited, context-grounded answers.

## Architecture

```
PDF / TXT
    |
 Ingestion  ──>  Chunking (512 chars, 64 overlap)
    |
 Embedding  ──>  Pinecone (multilingual-e5-large)
    |
  Query  ──>  Semantic Search (top-k)
    |
 Reranker  ──>  Pinecone Rerank (bge-reranker-v2-m3)
    |
Generation  ──>  OpenAI GPT-4o-mini (cited answer)
```

## Pipeline Steps

1. **Ingestion** - Extract text from PDF or TXT files
2. **Chunking** - Split text into 512-character chunks with 64-character overlap, tracking page numbers
3. **Embedding** - Upsert chunks into Pinecone using integrated `multilingual-e5-large` embeddings
4. **Retrieval** - Semantic search over Pinecone to find the top-k relevant chunks
5. **Reranking** - Rerank results with `bge-reranker-v2-m3` for better precision
6. **Generation** - Generate a cited answer using OpenAI `gpt-4o-mini`

## Tech Stack

| Component     | Technology               |
|---------------|--------------------------|
| API Framework | FastAPI + Uvicorn        |
| Vector DB     | Pinecone (Serverless)    |
| Embeddings    | multilingual-e5-large    |
| Reranker      | bge-reranker-v2-m3       |
| LLM           | OpenAI gpt-4o-mini       |
| PDF Parsing   | PyPDF                    |

## Prerequisites

- Python 3.9+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Pinecone API key](https://app.pinecone.io/)

## Setup

```bash
# Clone the repo
git clone https://github.com/Ragavendira1/rag-pipeline-classic.git
cd rag-pipeline-classic

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or with uv:
uv sync

# Create .env file
cp .env.example .env
# Edit .env and add your API keys
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

## Running the Server

```bash
.venv/bin/uvicorn apps.api:app --reload --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs

## API Endpoints

### Health Check

```
GET /health
```

Response:
```json
{"status": "ok"}
```

### Ingest Document

```
POST /ingest
```

Request:
```json
{
  "file_path": "docs/Apple_Q24.pdf"
}
```

Response:
```json
{
  "file": "docs/Apple_Q24.pdf",
  "chunk": 18,
  "message": "Ingestion successful"
}
```

## Project Structure

```
rag-pipeline-classic/
├── apps/
│   ├── __init__.py
│   ├── api.py           # FastAPI endpoints
│   ├── config.py         # Environment variables and settings
│   ├── ingestion.py      # PDF/TXT parsing and chunking
│   ├── embedding.py      # Pinecone index creation and upsert
│   ├── retrival.py       # Semantic search over Pinecone
│   ├── reranker.py       # Reranking with bge-reranker-v2-m3
│   └── generation.py     # Answer generation with OpenAI
├── docs/                 # Sample documents for ingestion
├── pyproject.toml
├── .env                  # API keys (not committed)
└── README.md
```

## Configuration

All settings are in `apps/config.py`:

| Setting              | Default                  | Description                        |
|----------------------|--------------------------|------------------------------------|
| `CHUNK_SIZE`         | 512                      | Characters per chunk               |
| `CHUNK_OVERLAP`      | 64                       | Overlap between chunks             |
| `TOP_K`              | 10                       | Chunks retrieved per query         |
| `RERANK_TOP_N`       | 5                        | Chunks returned after reranking    |
| `OPENAI_MODEL`       | gpt-4o-mini              | LLM for answer generation          |
| `MAX_TOKENS`         | 1024                     | Max tokens in generated response   |
| `TEMPERATURE`        | 0.2                      | Sampling temperature               |

## License

MIT
