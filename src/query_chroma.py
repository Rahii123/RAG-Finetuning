import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ==========================================
# CHROMA SETUP
# ==========================================

client = chromadb.HttpClient(host='localhost', port=8000)

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_collection(
    name="clinical_guidelines",
    embedding_function=embedding_function
)

# ==========================================
# QUERY
# ==========================================

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