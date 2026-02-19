import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from dotenv import load_dotenv
from groq import Groq
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================
# CONNECT TO CHROMA SERVER
# ==========================================
client = chromadb.HttpClient(host="localhost", port=8000)
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection(
    name="clinical_guidelines",
    embedding_function=embedding_function
)

# ==========================================
# CONNECT TO GROQ
# ==========================================
groq_client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# UTILITIES
# ==========================================
def deduplicate_chunks(chunks):
    """
    Remove duplicate sentences across chunks and merge similar items.
    """
    seen = set()
    deduped = []
    for chunk in chunks:
        sentences = chunk.split("\n")
        new_sentences = []
        for s in sentences:
            s_clean = s.strip().lower()
            if s_clean and s_clean not in seen:
                seen.add(s_clean)
                new_sentences.append(s.strip())
        if new_sentences:
            deduped.append(" ".join(new_sentences))
    return deduped

def filter_by_metadata(documents, metadatas, disease_type=None, year=None):
    """
    Filter retrieved documents based on metadata.
    Allows partial matching and case-insensitive matching.
    """
    filtered_docs = []
    filtered_meta = []
    for doc, meta in zip(documents, metadatas):
        doc_disease = meta.get("disease_type", "").lower()
        doc_year = str(meta.get("year", "")).lower()

        if disease_type and disease_type.lower() not in doc_disease:
            continue
        if year and str(year).lower() not in doc_year:
            continue

        filtered_docs.append(doc)
        filtered_meta.append(meta)
    return filtered_docs, filtered_meta


def retrieve_and_rerank(query, k=8, disease_type=None, year=None):
    """
    Retrieve top-k chunks from Chroma, rerank using embeddings, 
    and filter by metadata.
    """
    query_embedding = embedding_function([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Apply metadata filtering
    documents, metadatas = filter_by_metadata(documents, metadatas, disease_type, year)

    # Compute embeddings locally for reranking
    if not documents:
        return [], []

    doc_embeddings = embedding_function(documents)
    sims = cosine_similarity([query_embedding], doc_embeddings)[0]
    ranked_indices = np.argsort(sims)[::-1]

    ranked_documents = [documents[i] for i in ranked_indices]
    ranked_metadatas = [metadatas[i] for i in ranked_indices]

    # Deduplicate chunks
    deduped_documents = deduplicate_chunks(ranked_documents)
    deduped_metadatas = ranked_metadatas[:len(deduped_documents)]

    # Debug: print top retrieved chunks with smart document ID
    print("\n===== TOP RETRIEVED CHUNKS (DEBUG) =====\n")
    for i, doc in enumerate(deduped_documents):
        meta = deduped_metadatas[i]
        # Choose best available unique identifier
        doc_id = meta.get("document_id") or meta.get("chunk_id") or f"doc_{i+1}"
        print(f"[{i+1}] {doc_id} | Year: {meta.get('year', 'Unknown')} | Disease: {meta.get('disease_type', 'Unknown')}")
        print(doc[:500] + "...\n" + "-"*80)
    print("\n=========================================\n")

    return deduped_documents, deduped_metadatas

def build_prompt(context_chunks, metadatas, question):
    """
    Build grounded prompt with citations.
    """
    context_lines = []
    for i, chunk in enumerate(context_chunks):
        context_lines.append(f"[Source {i+1}] {chunk}")
    context_text = "\n\n".join(context_lines)

    prompt = f"""
You are a professional clinical medical assistant.

Use ONLY the information from the provided context below.
If the answer is not in the context, say: "The information is not available in the provided guidelines."

Context:
---------------------
{context_text}
---------------------

Question:
{question}

Provide a clear, structured, professional medical answer with citations.
Separate answer into 'Symptoms' and 'Precautions'.
"""
    return prompt

def generate_answer(prompt):
    """
    Generate answer from Groq LLaMA 3
    """
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print("\nOptional: You can filter by disease type and year. Press Enter to skip.\n")
    disease_type = input("Filter by disease_type (e.g., diabetes): ").strip() or None
    year = input("Filter by year (e.g., 2024): ").strip() or None

    while True:
        question = input("\nEnter your clinical question (or type 'exit'): ").strip()
        if question.lower() == "exit":
            break

        print("\nRetrieving relevant guidelines, applying metadata filter, and reranking...")
        context_chunks, metadatas = retrieve_and_rerank(question, disease_type=disease_type, year=year)

        if not context_chunks:
            print("\nNo relevant documents found for the given metadata filters.\n")
            continue

        print("\nGenerating grounded answer using Groq LLaMA 3...\n")
        prompt = build_prompt(context_chunks, metadatas, question)
        answer = generate_answer(prompt)

        print("\n===== FINAL ANSWER =====\n")
        print(answer)
        print("\n===== SOURCES =====\n")
        for i, meta in enumerate(metadatas):
            doc_id = meta.get("document_id") or meta.get("chunk_id") or f"doc_{i+1}"
            print(f"[{i+1}] Document: {doc_id} | Year: {meta.get('year', 'Unknown')} | Disease: {meta.get('disease_type', 'Unknown')}")
        print("\n========================\n")
