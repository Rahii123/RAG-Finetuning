import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# ==========================================
# CONFIG
# ==========================================

PERSIST_DIRECTORY = os.path.abspath("vector_store")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

CHUNK_FOLDER = r"D:\Rag+finetuning\data\chunks"

# ==========================================
# CHROMA SETUP
# ==========================================

client = chromadb.HttpClient(host='localhost', port=8000)

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="clinical_guidelines",
    embedding_function=embedding_function
)

print("Collection ready.")

# ==========================================
# LOAD CHUNKS
# ==========================================

total_chunks = 0

for filename in os.listdir(CHUNK_FOLDER):
    if filename.endswith(".json"):
        with open(os.path.join(CHUNK_FOLDER, filename), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        total_chunks += len(ids)

print(f"Total chunks added: {total_chunks}")
print("Chroma DB created successfully.")