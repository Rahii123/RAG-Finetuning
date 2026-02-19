import chromadb
from chromadb.config import Settings

PERSIST_DIRECTORY = "vector_store"

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
    )
)

collection = client.get_collection(name="clinical_guidelines")

query = input("Enter your question: ")

results = collection.query(
    query_texts=[query],
    n_results=5
)

print("\nTop Results:\n")

for i, doc in enumerate(results["documents"][0]):
    print(f"Result {i+1}")
    print(doc)
    print("-" * 80)
