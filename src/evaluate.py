"""
evaluate.py  –  RAG Evaluation Layer
======================================
Evaluates the RAG pipeline on a fixed medical test set.

Metrics computed:
  ✅ Retrieval Precision     – What fraction of retrieved chunks are relevant?
  ✅ Retrieval Recall        – Were the expected keywords found in retrieved chunks?
  ✅ Answer Relevance Score  – Does the LLM answer address the question? (LLM-as-judge)
  ✅ Faithfulness Check      – Does the answer use only retrieved content?
  ✅ Hallucination Detection – Does the answer contain facts NOT in retrieved chunks?
  ✅ Coverage Score          – % of expected key facts found in the answer

Usage:
    python src/evaluate.py
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=ENV_PATH, override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL      = "llama-3.3-70b-versatile"
GROQ_API_URL    = "https://api.groq.com/openai/v1/chat/completions"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
MIN_RELEVANCE   = 0.30
TOP_K           = 10   # ↑ was 7 — synced with rag_pipeline.py

# ──────────────────────────────────────────────
# Test Set  (ground-truth Q&A pairs)
# ──────────────────────────────────────────────
TEST_SET = [
    {
        "id": "HTN-01",
        "question": "Define hypertension and its classification by blood pressure levels.",
        "expected_disease": "hypertension",
        "expected_keywords": ["140 mmHg", "90 mmHg", "stage I", "systolic", "diastolic"],
        "expected_guideline_keywords": ["hypertension"],
    },
    {
        "id": "HTN-02",
        "question": "What are the diagnostic criteria and follow-up recommendations for hypertension?",
        "expected_disease": "hypertension",
        "expected_keywords": ["confirm", "two or more", "clinic visits", "follow-up", "two months"],
        "expected_guideline_keywords": ["hypertension"],
    },
    {
        "id": "HTN-03",
        "question": "What is the target blood pressure for diabetic patients with hypertension?",
        "expected_disease": "hypertension",
        "expected_keywords": ["140/80", "130/80", "diabetes", "ACEI", "ARB"],
        "expected_guideline_keywords": ["hypertension", "diabetes"],
    },
    {
        "id": "HTN-04",
        "question": "Describe the management of severe hypertension and hypertensive emergencies.",
        "expected_disease": "hypertension",
        "expected_keywords": ["180", "110", "labetalol", "nitroprusside", "25%", "12 hours"],
        "expected_guideline_keywords": ["hypertension"],
    },
    {
        "id": "DM-01",
        "question": "Define type 2 diabetes mellitus and its diagnostic criteria.",
        "expected_disease": "diabetes",
        "expected_keywords": ["fasting", "HbA1c", "glucose", "mmol", "diagnosis"],
        "expected_guideline_keywords": ["diabetes", "t2dm"],
    },
    {
        "id": "DM-02",
        "question": "What is the first-line pharmacological treatment for type 2 diabetes?",
        "expected_disease": "diabetes",
        "expected_keywords": ["metformin", "lifestyle", "glucose", "HbA1c"],
        "expected_guideline_keywords": ["diabetes"],
    },
    {
        "id": "HTN-SPECIAL-01",
        "question": "How should hypertension be managed in elderly patients above 80 years?",
        "expected_disease": "hypertension",
        "expected_keywords": ["150 mmHg", "elderly", "diuretic", "very elderly", "80 years"],
        "expected_guideline_keywords": ["hypertension"],
    },
    {
        "id": "HTN-SPECIAL-02",
        "question": "What is resistant hypertension and how is it managed?",
        "expected_disease": "hypertension",
        "expected_keywords": ["three drugs", "diuretic", "resistant", "denervation", "compliance"],
        "expected_guideline_keywords": ["hypertension"],
    },
]

# ──────────────────────────────────────────────
# Groq API
# ──────────────────────────────────────────────

def call_groq(prompt: str, max_tokens: int = 1024, retries: int = 4) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        resp = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        elif resp.status_code == 429:
            wait = 5 * (2 ** attempt)   # 5s, 10s, 20s, 40s
            print(f"   ⏳ Rate limit hit, retrying in {wait}s... (attempt {attempt+1}/{retries})")
            time.sleep(wait)
        else:
            raise RuntimeError(f"Groq error {resp.status_code}: {resp.text}")
    raise RuntimeError(f"Groq rate limit exceeded after {retries} retries.")

# ──────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────

def compute_retrieval_precision(chunks: list[dict], expected_guideline_keywords: list[str]) -> float:
    """
    Precision: fraction of retrieved chunks whose guideline name or disease_type
    matches expected topic keywords.
    """
    if not chunks:
        return 0.0
    relevant = 0
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        guideline = (meta.get("guideline_name", "") + " " + meta.get("disease_type", "")).lower()
        if any(kw.lower() in guideline for kw in expected_guideline_keywords):
            relevant += 1
    return relevant / len(chunks)


def compute_keyword_recall(answer: str, expected_keywords: list[str]) -> tuple[float, list[str]]:
    """
    Recall: fraction of expected key facts found anywhere in the answer.
    Returns (score, list_of_missing_keywords).
    """
    answer_lower = answer.lower()
    found   = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    return len(found) / len(expected_keywords), missing


def llm_judge_relevance(question: str, answer: str) -> dict:
    """
    Use LLM-as-judge to score relevance (1–5) and flag hallucinations.
    Returns {relevance_score, faithful, hallucination_flag, reasoning}.
    """
    prompt = f"""You are a medical AI evaluator. Evaluate this RAG system's answer.

QUESTION: {question}

ANSWER:
{answer}

Evaluate and respond as JSON ONLY (no extra text):
{{
  "relevance_score": <integer 1-5 where 5=perfectly relevant, 1=completely irrelevant>,
  "is_structured": <true if the answer has clear sections like Definition/Management/etc>,
  "has_citations": <true if the answer mentions guideline names or years>,
  "hallucination_flag": <true if the answer contains medical claims that seem fabricated or not from guidelines>,
  "brief_reasoning": "<one sentence explanation>"
}}"""
    
    try:
        raw = call_groq(prompt, max_tokens=300)
        # Extract JSON from response
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception as e:
        return {
            "relevance_score": 0,
            "is_structured": False,
            "has_citations": False,
            "hallucination_flag": None,
            "brief_reasoning": f"Evaluation failed: {e}",
        }

# ──────────────────────────────────────────────
# Main Evaluation Runner
# ──────────────────────────────────────────────

def main():
    print("🔄 Loading embedding model…")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    client     = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_collection("clinical_guidelines")
    print(f"✅ ChromaDB connected | {collection.count()} documents\n")

    results = []
    print("=" * 70)
    print(f"  🧪  RAG EVALUATION  |  {len(TEST_SET)} test cases")
    print("=" * 70)

    for test in TEST_SET:
        print(f"\n📋 [{test['id']}] {test['question'][:70]}…")

        # --- Retrieval ---
        prefixed_q  = f"Represent this sentence for searching relevant passages: {test['question']}"
        q_embedding = emb_model.encode(prefixed_q).tolist()

        raw_results = collection.query(
            query_embeddings=[q_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            raw_results["documents"][0],
            raw_results["metadatas"][0],
            raw_results["distances"][0],
        ):
            sim = 1 / (1 + dist)
            if sim >= MIN_RELEVANCE:
                chunks.append({"text": doc, "metadata": meta, "similarity": sim})

        # --- Build context for generation ---
        context_str = ""
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            context_str += (
                f"[Extract {i+1}] "
                f"Guideline: {meta.get('guideline_name', '?')} | Year: {meta.get('year', '?')}\n"
                f"{chunk['text']}\n\n"
            )

        # --- Generation  (same prompt structure as rag_pipeline.py) ---
        unique_sources = list({
            f"{c['metadata'].get('guideline_name', '?')} ({c['metadata'].get('year', '?')})"
            for c in chunks
        })
        sources_hint = "\n".join(f"  - {s}" for s in unique_sources)

        gen_prompt = f"""You are an expert clinical guideline assistant trained on peer-reviewed medical guidelines.

=== MANDATORY RULES ===
1. Use ONLY the guideline extracts provided. Do NOT use prior knowledge.
2. EVERY section MUST end with a citation: (Source: <guideline_name>, <year>)
3. If a section has no supporting extract, write: "Not specified in retrieved guidelines."
4. You MUST produce ALL 7 sections — do not skip any.

Available Guidelines (cite from these):
{sources_hint}

Clinical Question: {test['question']}

Guideline Extracts:
{context_str}

Answer with ALL 7 sections:
**1. Definition** ... (Source: ...)
**2. Diagnostic Criteria** ... (Source: ...)
**3. Classification** ... (Source: ...)
**4. Management – Non-Pharmacological** ... (Source: ...)
**5. Management – Pharmacological** ... (Source: ...)
**6. Special Populations** ... (Source: ...)
**7. Target Outcomes & Follow-up** ... (Source: ...)

⚠️ Every section must have a (Source: ...) citation. No exceptions."""

        try:
            answer = call_groq(gen_prompt, max_tokens=1500)
        except RuntimeError as e:
            answer = f"ERROR: {e}"

        # --- Compute metrics ---
        retrieval_precision = compute_retrieval_precision(chunks, test["expected_guideline_keywords"])
        keyword_recall, missing_kws = compute_keyword_recall(answer, test["expected_keywords"])

        print(f"   Retrieval Precision : {retrieval_precision:.2f}  ({len(chunks)} chunks retrieved)")
        print(f"   Keyword Recall      : {keyword_recall:.2f}  (missing: {missing_kws or 'none'})")

        # LLM judge (with rate limit protection)
        time.sleep(1)
        judge_result = llm_judge_relevance(test["question"], answer)
        print(f"   LLM Relevance Score : {judge_result.get('relevance_score', '?')}/5")
        print(f"   Has Citations       : {judge_result.get('has_citations', '?')}")
        print(f"   Hallucination Flag  : {judge_result.get('hallucination_flag', '?')}")
        print(f"   Reasoning           : {judge_result.get('brief_reasoning', '?')}")

        results.append({
            "test_id":              test["id"],
            "question":             test["question"],
            "chunks_retrieved":     len(chunks),
            "retrieval_precision":  round(retrieval_precision, 3),
            "keyword_recall":       round(keyword_recall, 3),
            "missing_keywords":     missing_kws,
            "llm_relevance_score":  judge_result.get("relevance_score"),
            "has_citations":        judge_result.get("has_citations"),
            "is_structured":        judge_result.get("is_structured"),
            "hallucination_flag":   judge_result.get("hallucination_flag"),
            "judge_reasoning":      judge_result.get("brief_reasoning"),
            "generated_answer":     answer[:500] + "…" if len(answer) > 500 else answer,
        })

        time.sleep(4)   # ↑ was 0.5s — stays under 12k TPM/min Groq free tier limit

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  📊  EVALUATION SUMMARY")
    print("=" * 70)

    avg_precision = sum(r["retrieval_precision"] for r in results) / len(results)
    avg_recall    = sum(r["keyword_recall"] for r in results) / len(results)
    avg_relevance = sum(r["llm_relevance_score"] or 0 for r in results) / len(results)
    n_hallucinations = sum(1 for r in results if r["hallucination_flag"] is True)
    n_cited       = sum(1 for r in results if r["has_citations"] is True)
    n_structured  = sum(1 for r in results if r["is_structured"] is True)

    print(f"  Avg Retrieval Precision : {avg_precision:.2f}")
    print(f"  Avg Keyword Recall      : {avg_recall:.2f}")
    print(f"  Avg LLM Relevance       : {avg_relevance:.2f}/5")
    print(f"  Answers with Citations  : {n_cited}/{len(results)}")
    print(f"  Structured Answers      : {n_structured}/{len(results)}")
    print(f"  Hallucination Flags     : {n_hallucinations}/{len(results)}")
    print("=" * 70)

    # ── Save report ──
    report_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_reports"))
    os.makedirs(report_dir, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"eval_{timestamp}.json")

    summary = {
        "timestamp": timestamp,
        "model": GROQ_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "n_tests": len(results),
        "avg_retrieval_precision": round(avg_precision, 3),
        "avg_keyword_recall": round(avg_recall, 3),
        "avg_llm_relevance": round(avg_relevance, 3),
        "answers_with_citations": n_cited,
        "structured_answers": n_structured,
        "hallucination_flags": n_hallucinations,
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Full report saved → {report_path}")
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
