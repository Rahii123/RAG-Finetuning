"""
chroma_store.py  –  Build/Rebuild ChromaDB Vector Store
=========================================================
Upgrades over v1:
  ✅ Uses upgraded embedding model: BAAI/bge-base-en-v1.5  (768 dims)
  ✅ --reset flag to wipe old collection (REQUIRED when changing embedding models)
  ✅ Upserts instead of add (idempotent re-runs)
  ✅ Validates chunk metadata before ingestion
  ✅ Handles ChromaDB metadata type coercion (all values must be str/int/float/bool)
  ✅ Progress reporting per file

Usage:
    # First time OR after changing embedding model → wipe + rebuild:
    python src/chroma_store.py --reset

    # Safe re-run (upsert only, keep existing collection):
    python src/chroma_store.py
"""

import os
import sys
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ==========================================
# CONFIG
# ==========================================

CHUNK_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chunks"))

# 🔹 Upgraded from all-MiniLM-L6-v2 → BAAI/bge-base-en-v1.5
# IMPORTANT: If you change the embedding model you MUST wipe and rebuild the collection.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

BATCH_SIZE = 64   # ChromaDB performs best with batched upserts

# ==========================================
# CHROMA SETUP
# ==========================================

client = chromadb.HttpClient(host="localhost", port=8000)

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# ── Handle --reset: wipe old collection so dimension schema is fresh ──
RESET = "--reset" in sys.argv

if RESET:
    try:
        client.delete_collection(name="clinical_guidelines")
        print("🗑️  Old collection deleted (dimension schema reset).")
    except Exception:
        print("ℹ️  No existing collection to delete — creating fresh.")

collection = client.get_or_create_collection(
    name="clinical_guidelines",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},   # cosine similarity space
)

print(f"✅ Collection ready  |  Existing docs: {collection.count()}")
print(f"📦 Embedding model  : {EMBEDDING_MODEL}  (768 dims)")
print(f"🔄 Reset mode       : {'YES — full rebuild' if RESET else 'NO — upsert only'}\n")

# ==========================================
# METADATA SANITIZER
# ==========================================

def sanitize_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str, int, float, bool values.
    Converts anything else to string; forces year to int.
    """
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, float, bool)):
            clean[k] = v
        elif isinstance(v, int):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)

    # Ensure year is always int
    try:
        clean["year"] = int(clean.get("year", 0))
    except (ValueError, TypeError):
        clean["year"] = 0

    return clean

# ==========================================
# LOAD CHUNKS
# ==========================================

total_chunks = 0
total_files  = 0

for filename in sorted(os.listdir(CHUNK_FOLDER)):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(CHUNK_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print(f"⚠️  Skipped (empty): {filename}")
        continue

    print(f"📄 Indexing: {filename}  ({len(chunks)} chunks)")

    # Batch upsert
    for start in range(0, len(chunks), BATCH_SIZE):
        batch  = chunks[start : start + BATCH_SIZE]
        ids    = [c["chunk_id"] for c in batch]
        docs   = [c["text"] for c in batch]
        metas  = [sanitize_metadata(c["metadata"]) for c in batch]

        try:
            collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
            )
        except Exception as e:
            print(f"   ⚠️  Batch failed ({start}-{start+len(batch)}): {e}")
            continue

    total_chunks += len(chunks)
    total_files  += 1

# ==========================================
# SUMMARY
# ==========================================

print(f"\n🎯 Done!")
print(f"   Files processed : {total_files}")
print(f"   Chunks indexed  : {total_chunks}")
print(f"   Collection total: {collection.count()}")
print("✅ ChromaDB vector store is ready.")