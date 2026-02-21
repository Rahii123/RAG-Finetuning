"""
🔬 PRODUCTION RAG EVALUATION - Medical Domain
===========================================
Advanced metrics: RAGAS + Custom Medical Validators
"""
from ragas import evaluate
from ragas.metrics import (
    faithfulness, context_precision, context_recall, 
    answer_relevancy, answer_semantic_similarity
)
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any

class MedicalRAGEvaluator:
    def __init__(self):
        self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """FIX 5: Medical entity extraction for eval"""
        entities = {
            'bp_values': re.findall(r'\b\d{2,3}\s*(?:mmHg|BP)\b', text),
            'hba1c': re.findall(r'\b(?:HbA1c|A1C)\s*[<>]?\s*\d+(?:\.\d+)?%', text),
            'drugs': re.findall(r'\b(?:ACEI|ARB|CCB|beta-blocker|statin|metformin)\b', text, re.I),
            'targets': re.findall(r'\b[<>]?\s*[\d.]+(?:mmHg|mg/dL|mmol/L|%)', text)
        }
        return {k: list(set(v)) for k,v in entities.items()}
    
    def evaluate_medical_accuracy(self, answer: str, context_docs: List[Dict]) -> Dict:
        """FIX 6: Domain-specific medical eval"""
        answer_entities = self.extract_medical_entities(answer)
        context_entities = {}
        
        for doc in context_docs:
            text = doc.get('page_content', '')
            context_entities.update(self.extract_medical_entities(text))
        
        # Entity recall (did answer capture context entities?)
        scores = {}
        for entity_type in answer_entities:
            answer_set = set(answer_entities[entity_type])
            context_set = set(context_entities.get(entity_type, []))
            scores[f'{entity_type}_recall'] = len(answer_set & context_set) / len(context_set) if context_set else 1.0
            
        return scores
    
    def advanced_rag_eval(self, query: str, docs: List[Dict], answer: str, reference=None) -> Dict:
        """FIX 7: RAGAS + Custom Medical Metrics"""
        
        # Prepare RAGAS format
        ragas_docs = [{"page_content": d.get('page_content', ''), "metadata": d.get('metadata', {})} for d in docs]
        
        result = evaluate(
            dataset=[{
                "question": query,
                "contexts": [d['page_content'] for d in ragas_docs],
                "answer": answer,
                **({"ground_truth": reference} if reference else {})
            }],
            metrics=[
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy,
                answer_semantic_similarity
            ]
        )
        
        # Add medical-specific metrics
        medical_scores = self.evaluate_medical_accuracy(answer, docs)
        
        return {
            **result,
            **medical_scores,
            'total_docs_retrieved': len(docs),
            'medical_entity_coverage': np.mean(list(medical_scores.values()))
        }

# Usage in main()
evaluator = MedicalRAGEvaluator()
eval_results = evaluator.advanced_rag_eval(query, docs, answer)
print("🏥 Medical RAGAS Scores:", eval_results)