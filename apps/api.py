from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from .ingestion import ingest_document
from .embedding import upsert_chunks
from .retrival import search
from .reranker import rerank
from .generation import generate_answer

# Write FASTAPI code
app = FastAPI(
    title="RAG Pipeline API",
    description="API for the RAG pipeline with ingestion, embedding, retrieval, reranking, and generation.",
    version="1.0.0"
)

# Define request and response models
class IngestRequest(BaseModel):
    file_path: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"file_path": "docs/Apple_Q24.pdf"}
            ]
        }
    }

# Ingestion response
class IngestResponse(BaseModel):
    file: str
    chunk: int
    message: str
# Chat request

class ChatRequest(BaseModel):
    quetion: str
    use_reranker: bool=True
    debug: bool=False
class Source_chunk(BaseModel):
    id: str
    score: float
    source: str
    pages: str
    chunk_text: str
    citiation: str

class ChatResponse(BaseModel):
    answer: str
    source_chunks: List[Source_chunk]
    retrived: Optional[List[Source_chunk]] = None
    reranked: Optional[List[Source_chunk]] = None

class gnerate_request(BaseModel):
    question: str
    source_chunks: List[Source_chunk]
    top_k: int = 10
    top_n: int = 5
    use_reranker: bool = True

class generate_response(BaseModel):
    question: str
    answer: str
    sources: List[Source_chunk]
    pipeline: str

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_reranker: Optional[bool] = True

# add endpoints 
@app.get("/health")
def health_check():
    """
    Simple health check endpoint to verify that the API is running.
    """
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(request: IngestRequest):
    """
    Endpoint to ingest a document from a specified file path.
    """
    try:
        result = ingest_document(request.file_path)
        upserted=upsert_chunks(result)
    
        return IngestResponse(file=request.file_path, chunk=upserted, message="Ingestion successful")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

