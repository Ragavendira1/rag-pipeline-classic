import os 
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Chunking Settings
CHUNK_SIZE: int = 512 # Number of characters per chunk
CHUNK_OVERLAP: int = 64 # Number of characters to overlap between chunks
# Pinecone Settings
PINECONE_INDEX_NAME: str = "rag-pipeline-classic"
PINECONE_NAMESPACE: str = "documents"
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"
PINECONE_EMBED_MODEL: str = "multilingual-e5-large"
PINECONE_RERANK_MODEL: str = "bge-reranker-v2-m3"

# Retrieval Settings
TOP_K: int = 10 # Number of top relevant chunks to retrieve
RERANK_TOP_N: int = 5 # Number of top chunks to rerank and return

# Generation Settings
OPENAI_MODEL: str = "gpt-4o-mini" # OpenAI model for generation
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_TOKENS: int = 1024 # Maximum tokens for generated response
TEMPERATURE: float = 0.2 # Sampling temperature for generation      