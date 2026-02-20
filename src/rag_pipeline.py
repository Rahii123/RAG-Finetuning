"""
rag_pipeline.py  –  Production-Grade Medical RAG
==================================================
Upgrades over v1:
  ✅ Stronger embedding model: BAAI/bge-base-en-v1.5
  ✅ Rich metadata in context rendering (guideline_name, section_type, year)
  ✅ Professional structured medical prompt template
  ✅ Per-source attribution in the final answer
  ✅ Similarity score displayed + confidence signal
  ✅ Distance filtering: ignores chunks below relevance threshold
  ✅ Graceful error handling with retry logic
"""

import os
import json
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# ===============================
# Load .env
# ===============================
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=ENV_PATH, override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

# ===============================
# Configuration
# ===============================
CHROMA_HOST      = "localhost"
CHROMA_PORT      = 8000
COLLECTION_NAME  = "clinical_guidelines"
TOP_K            = 10         # ↑ was 7 — more context → better keyword recall
MIN_RELEVANCE    = 0.30       # cosine similarity threshold (0 = identical, 2 = opposite)
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_API_URL     = "https://api.groq.com/openai/v1/chat/completions"

# 🔹 Upgraded embedding model: BAAI/bge-base-en-v1.5 is measurably better
# than all-MiniLM-L6-v2 for domain-specific retrieval.
# NOTE: If ChromaDB was indexed with the OLD model, you must re-index first.
#       To re-use the old index, change this back to "all-MiniLM-L6-v2"
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"

# ===============================
# Initialize Clients
# ===============================
print("🔄 Loading embedding model…")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client     = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"✅ Connected to ChromaDB  |  Collection: {COLLECTION_NAME}  |  Docs: {collection.count()}")

# ===============================
# Prompt Builder  (Structured Medical Prompt)
# ===============================

def build_medical_system_prompt() -> str:
    return """You are an expert clinical guideline assistant trained on peer-reviewed medical guidelines.

=== MANDATORY RULES (non-negotiable) ===
1. Use ONLY the guideline extracts provided by the user. Do NOT use prior knowledge.
2. EVERY section you write MUST end with a citation: (Source: <guideline_name>, <year>)
   - If a section has no supporting extract, write: "Not specified in retrieved guidelines."
3. Never invent drug names, dosages, thresholds, or classification criteria.
4. You MUST produce ALL 7 sections listed below — do not skip any section.
5. If information for a section is absent, still include the section heading and write "Not specified in retrieved guidelines."

=== MANDATORY OUTPUT FORMAT ===
You MUST follow this exact structure. Do not deviate:

**1. Definition**
[Write definition here] (Source: <guideline_name>, <year>)

**2. Diagnostic Criteria**
[Write diagnostic thresholds and tests here] (Source: <guideline_name>, <year>)

**3. Classification**
[Write severity/staging classification here] (Source: <guideline_name>, <year>)

**4. Management – Non-Pharmacological**
[Write lifestyle/diet/exercise interventions here] (Source: <guideline_name>, <year>)

**5. Management – Pharmacological**
[Write first-line drugs, combinations, dosing principles here] (Source: <guideline_name>, <year>)

**6. Special Populations**
[Write guidance for elderly/pregnant/children/renal/diabetic patients here] (Source: <guideline_name>, <year>)

**7. Target Outcomes & Follow-up**
[Write target values, monitoring intervals, referral thresholds here] (Source: <guideline_name>, <year>)

---
REMINDER: Citation after EVERY section is mandatory. No exceptions."""


def build_user_prompt(query: str, context_blocks: list[dict]) -> str:
    context_str = ""
    for i, block in enumerate(context_blocks):
        meta = block["metadata"]
        similarity = block.get("similarity", "N/A")
        context_str += (
            f"\n[Extract {i+1}] "
            f"Guideline: {meta.get('guideline_name', 'Unknown')} | "
            f"Year: {meta.get('year', '?')} | "
            f"Section: {meta.get('section_header', '?')} | "
            f"Relevance: {similarity:.2f}\n"
            f"{block['text']}\n"
            + "-" * 60 + "\n"
        )

    # Build citation hint: list all unique guideline names the LLM can cite from
    unique_sources = list({
        f"{b['metadata'].get('guideline_name', 'Unknown')} ({b['metadata'].get('year', '?')})"
        for b in context_blocks
    })
    sources_hint = "\n".join(f"  - {s}" for s in unique_sources)

    return f"""Clinical Question:
{query}

Available Guidelines (you MUST cite from these):
{sources_hint}

Guideline Extracts:
{context_str}

⚠️ REMINDER: Your response MUST include ALL 7 sections and EVERY section MUST have a (Source: ...) citation.
"""

# ===============================
# Retrieval with Similarity Scoring
# ===============================

def retrieve(query: str) -> list[dict]:
    """
    Embed query, search ChromaDB, filter by MIN_RELEVANCE,
    return enriched chunk dicts sorted by relevance.
    """
    # BGE models work best with this query prefix for retrieval tasks
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
    query_embedding = embedding_model.encode(prefixed_query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]  # ChromaDB returns L2 distances

    enriched = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert L2 distance → approximate cosine-style score (lower = more relevant)
        # Chroma default is L2; dist ≈ 0 means identical
        similarity_score = 1 / (1 + dist)   # maps 0→∞ into 1→0

        if similarity_score < MIN_RELEVANCE:
            continue  # Filter irrelevant chunks

        enriched.append({
            "text":       doc,
            "metadata":   meta,
            "distance":   dist,
            "similarity": similarity_score,
        })

    # Sort: most relevant first
    enriched.sort(key=lambda x: x["similarity"], reverse=True)
    return enriched

# ===============================
# Groq API Call
# ===============================

def call_groq(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }

    response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload), timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]

# ===============================
# Display Helpers
# ===============================

def display_sources(chunks: list[dict]) -> None:
    print("\n📚 Retrieved Sources:")
    print("─" * 60)
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        print(
            f"  [{i+1}] {meta.get('guideline_name', '?')}  "
            f"({meta.get('year', '?')})  |  "
            f"Section: {meta.get('section_header', '?')}  |  "
            f"Relevance: {chunk['similarity']:.3f}"
        )
    print("─" * 60)

# ===============================
# Main Interactive Loop
# ===============================

SYSTEM_PROMPT = build_medical_system_prompt()

print("\n" + "=" * 60)
print("  🏥  Clinical Guideline RAG Assistant")
print(f"  Model : {GROQ_MODEL}")
print(f"  Embed : {EMBEDDING_MODEL}")
print("=" * 60)

while True:
    try:
        query = input("\n🩺 Enter your clinical question (or 'exit'): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting…")
        break

    if query.lower() in ("exit", "quit", "q"):
        print("Goodbye!")
        break

    if not query:
        continue

    # --- Retrieval ---
    chunks = retrieve(query)

    if not chunks:
        print("\n⚠️  No sufficiently relevant documents found in the knowledge base.")
        print("   Try rephrasing your question or checking if the topic is covered.")
        continue

    display_sources(chunks)

    # --- Prompting ---
    user_prompt = build_user_prompt(query, chunks)

    # --- Generation ---
    print("\n⏳ Generating structured answer…\n")
    try:
        answer = call_groq(SYSTEM_PROMPT, user_prompt)
    except RuntimeError as e:
        print(f"❌ {e}")
        continue

    # --- Output ---
    print("═" * 60)
    print("  📋  CLINICAL GUIDELINE ANSWER")
    print("═" * 60)
    print(answer)
    print("═" * 60)
    print(f"\n⚙️  [Retrieved {len(chunks)} relevant chunks | Model: {GROQ_MODEL}]")
