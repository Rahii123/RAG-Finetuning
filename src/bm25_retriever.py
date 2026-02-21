# Hybrid Search Utilities
# This module provides BM25 keyword search for hybrid retrieval.
# Uses rank_bm25 (pip install rank_bm25)

from rank_bm25 import BM25Okapi
import json
import os
import re

class BM25Retriever:
    def __init__(self, chunk_folder):
        self.chunk_folder = chunk_folder
        self.chunks = []
        self.corpus = []
        self.chunk_ids = []
        self._load_chunks()
        self.bm25 = BM25Okapi(self.corpus)

    def _load_chunks(self):
        for fname in os.listdir(self.chunk_folder):
            if not fname.endswith(".json"): continue
            with open(os.path.join(self.chunk_folder, fname), encoding="utf-8") as f:
                data = json.load(f)
                for chunk in data:
                    text = chunk["text"]
                    tokens = self._tokenize(text)
                    self.corpus.append(tokens)
                    self.chunks.append(chunk)
                    self.chunk_ids.append(chunk["chunk_id"])

    def _tokenize(self, text):
        # Simple whitespace + punctuation split
        return re.findall(r"\w+", text.lower())

    def search(self, query, top_k=5):
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        if not scores.any():
            return []
        max_score = max(scores)
        min_score = min(scores)
        # Avoid division by zero
        norm_scores = [(s - min_score) / (max_score - min_score) if max_score > min_score else 0.0 for s in scores]
        ranked = sorted(zip(norm_scores, self.chunks), key=lambda x: x[0], reverse=True)
        # Attach normalized similarity to each chunk
        results = []
        for sim, chunk in ranked[:top_k]:
            chunk = dict(chunk)  # copy to avoid mutating original
            chunk["similarity"] = sim
            results.append(chunk)
        return results
