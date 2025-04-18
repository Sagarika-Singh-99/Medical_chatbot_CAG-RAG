pip install chromadb sentence-transformers

import chromadb
from sentence_transformers import SentenceTransformer

# Set up ChromaDB client with updated configuration
client = chromadb.PersistentClient(path="./cag_chroma_db")

# Create or load collection
collection = client.get_or_create_collection(
    name="symptom_checker_memory",
    embedding_function=None  # You'll set this up separately
)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight and fast




