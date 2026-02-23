from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from .ingestion import ingest_document
from .embedding import upsert_chunks
from .retrieval import search
from .reranker import rerank
from .generation import generate_answer

app = FastAPI(
    title="RAG Pipeline API",
    description="API for the RAG pipeline with ingestion, embedding, retrieval, reranking, and generation.",
    version="1.0.0"
)


# --------------- Request / Response Models ---------------

class IngestRequest(BaseModel):
    file_path: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"file_path": "docs/Apple_Q24.pdf"}
            ]
        }
    }


class IngestResponse(BaseModel):
    file: str
    chunk: int
    message: str


class SourceChunk(BaseModel):
    id: str
    score: float
    source: str
    pages: str
    chunk_text: str
    citation: str = ""


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_reranker: Optional[bool] = True


class SearchResponse(BaseModel):
    query: str
    chunks: List[SourceChunk]
    pipeline: str


class ChatRequest(BaseModel):
    question: str
    use_reranker: bool = True
    top_k: int = 10
    top_n: int = 5
    debug: bool = False


class ChatResponse(BaseModel):
    answer: str
    source_chunks: List[SourceChunk]
    retrieved: Optional[List[SourceChunk]] = None
    reranked: Optional[List[SourceChunk]] = None


# --------------- Helper ---------------

def _hits_to_source_chunks(hits: list) -> List[SourceChunk]:
    """Convert raw hit dicts from retrieval/reranker into SourceChunk models."""
    chunks = []
    for h in hits:
        chunks.append(SourceChunk(
            id=h.get("id", ""),
            score=h.get("score", 0.0),
            source=h.get("source", ""),
            pages=h.get("pages", ""),
            chunk_text=h.get("chunk_text", ""),
            citation=f"{h.get('source', '')}, p.{h.get('pages', '')}",
        ))
    return chunks


# --------------- Endpoints ---------------

@app.get("/health")
def health_check():
    """Simple health check endpoint to verify that the API is running."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(request: IngestRequest):
    """Ingest a document: extract text, chunk it, embed, and upsert into Pinecone."""
    try:
        result = ingest_document(request.file_path)
        upserted = upsert_chunks(result)
        return IngestResponse(
            file=request.file_path,
            chunk=upserted,
            message="Ingestion successful",
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    """Retrieve relevant chunks for a query using semantic search, optionally with reranking."""
    try:
        if request.use_reranker:
            hits = rerank(request.query, top_k=request.top_k)
            pipeline = "retrieval + reranker"
        else:
            hits = search(request.query, top_k=request.top_k)
            pipeline = "retrieval only"

        return SearchResponse(
            query=request.query,
            chunks=_hits_to_source_chunks(hits),
            pipeline=pipeline,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """End-to-end RAG: retrieve chunks, optionally rerank, and generate a cited answer."""
    try:
        retrieved_hits = search(request.question, top_k=request.top_k)

        if request.use_reranker:
            reranked_hits = rerank(request.question, top_k=request.top_k, top_n=request.top_n)
            context_hits = reranked_hits
        else:
            reranked_hits = None
            context_hits = retrieved_hits

        answer = generate_answer(request.question, context_hits)

        response = ChatResponse(
            answer=answer,
            source_chunks=_hits_to_source_chunks(context_hits),
        )

        if request.debug:
            response.retrieved = _hits_to_source_chunks(retrieved_hits)
            if reranked_hits is not None:
                response.reranked = _hits_to_source_chunks(reranked_hits)

        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
