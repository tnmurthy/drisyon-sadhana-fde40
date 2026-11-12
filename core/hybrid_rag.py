"""
Drisyon Sadhana · Forward Deployed AI Engineer Starter Kit
Module 3: Reciprocal Rank Fusion (RRF) for Enterprise RAG
"""

from typing import List, Dict, Any


class HybridRAGRetriever:
    """Combines sparse keyword search (BM25) and dense vector search via RRF."""

    @staticmethod
    def compute_rrf_scores(
        dense_results: List[str],
        sparse_results: List[str],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Calculates RRF score:
        RRF_Score(doc) = Σ 1 / (k + rank)
        """
        scores: Dict[str, float] = {}

        # 1. Process dense vector rankings (1-indexed)
        for rank, doc_id in enumerate(dense_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # 2. Process sparse BM25 rankings (1-indexed)
        for rank, doc_id in enumerate(sparse_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # 3. Sort by highest combined score
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [{"doc_id": doc_id, "rrf_score": round(score, 6)} for doc_id, score in sorted_results]
