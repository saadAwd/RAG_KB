"""
Hybrid retriever (sparse + dense) over KB chunks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from .config import RAGConfig
from .dense_index import DenseKBIndex
from .sparse_index import SparseKBIndex


def _normalize_scores(pairs: List[tuple]) -> List[tuple]:
    """Min-max normalize scores to [0, 1]."""
    if not pairs:
        return []
    scores = np.array([s for _, s in pairs], dtype=float)
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-8:
        norm = np.full_like(scores, 0.5)
    else:
        norm = (scores - s_min) / (s_max - s_min)
    return [(pairs[i][0], float(norm[i])) for i in range(len(pairs))]


class HybridKBRetriever:
    """Combines sparse BM25 and dense semantic scores."""

    def __init__(
        self,
        sparse_index: SparseKBIndex,
        dense_index: DenseKBIndex,
        alpha: float = None,
        rerank_model: Optional[CrossEncoder] = None,
    ) -> None:
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.alpha = alpha if alpha is not None else RAGConfig.HYBRID_ALPHA
        self.rerank_model = rerank_model

    @classmethod
    def build(cls, alpha: float = None, use_cpu: bool = True) -> "HybridKBRetriever":
        """Build hybrid retriever."""
        if alpha is None:
            alpha = RAGConfig.HYBRID_ALPHA
        
        sparse = SparseKBIndex.build()
        print(f"[INFO] Initializing Bi-Encoder (MARBERTv2) on CPU...")
        dense = DenseKBIndex.build(device="cpu" if use_cpu else None)
        
        rerank_model = None
        if RAGConfig.USE_RERANKING and RAGConfig.RERANKING_MODEL:
            print(f"Loading re-ranking model: {RAGConfig.RERANKING_MODEL} on CPU...")
            rerank_model = CrossEncoder(RAGConfig.RERANKING_MODEL, device="cpu")
            
        return cls(sparse_index=sparse, dense_index=dense, alpha=alpha, rerank_model=rerank_model)

    def search(self, query: str, top_k: int = 10, rerank: bool = True) -> List[Dict]:
        """Return top_k chunks with combined scores."""
        rerank_top_k = RAGConfig.RERANKING_TOP_K if rerank else top_k
        retrieve_k = rerank_top_k if (rerank and self.rerank_model) else top_k
        
        sparse_pairs = self.sparse_index.search(query, top_k=retrieve_k * 2)
        dense_pairs = self.dense_index.search(query, top_k=retrieve_k * 2)

        sparse_norm = _normalize_scores(sparse_pairs)
        dense_norm = _normalize_scores(dense_pairs)

        # Create maps
        sparse_raw_map = {c["chunk_id"]: (c, s) for c, s in sparse_pairs}
        sparse_norm_map = {c["chunk_id"]: (c, s) for c, s in sparse_norm}
        dense_raw_map = {c["chunk_id"]: (c, s) for c, s in dense_pairs}
        dense_norm_map = {c["chunk_id"]: (c, s) for c, s in dense_norm}
        
        sparse_map = {}
        for cid in sparse_raw_map.keys():
            c_raw, s_raw = sparse_raw_map[cid]
            c_norm, s_norm = sparse_norm_map.get(cid, (c_raw, 0.0))
            sparse_map[cid] = (c_norm, s_norm, s_raw)
        
        dense_map = {}
        for cid in dense_raw_map.keys():
            c_raw, s_raw = dense_raw_map[cid]
            c_norm, s_norm = dense_norm_map.get(cid, (c_raw, 0.0))
            dense_map[cid] = (c_norm, s_norm, s_raw)

        all_ids = set(sparse_map.keys()) | set(dense_map.keys())
        initial_results: List[Dict] = []

        for cid in all_ids:
            sparse_data = sparse_map.get(cid, (None, 0.0, 0.0))
            dense_data = dense_map.get(cid, (None, 0.0, 0.0))
            
            c_sparse, s_sparse_norm, s_sparse_raw = sparse_data
            c_dense, s_dense_norm, s_dense_raw = dense_data
            
            chunk = c_sparse or c_dense
            if not chunk:
                continue
            
            score_hybrid = self.alpha * s_sparse_norm + (1.0 - self.alpha) * s_dense_norm
            score_raw_dense = s_dense_raw
            
            initial_results.append(
                {
                    "chunk": chunk,
                    "score_hybrid": float(score_hybrid),
                    "score_hybrid_original": float(score_hybrid),
                    "score_sparse": float(s_sparse_norm),
                    "score_dense": float(s_dense_norm),
                    "score_raw_dense": float(score_raw_dense),
                }
            )

        # Sort by hybrid score
        initial_results.sort(key=lambda r: r["score_hybrid"], reverse=True)
        
        # Deduplicate
        seen_texts = {}
        deduplicated = []
        for result in initial_results:
            chunk_text = result["chunk"].get("text", result["chunk"].get("clean_text", ""))
            text_key = " ".join(chunk_text.split())
            
            if text_key not in seen_texts:
                seen_texts[text_key] = result
                deduplicated.append(result)
            else:
                existing = seen_texts[text_key]
                if result["score_hybrid"] > existing["score_hybrid"]:
                    deduplicated.remove(existing)
                    seen_texts[text_key] = result
                    deduplicated.append(result)
        
        candidates = deduplicated[:retrieve_k]

        # Re-ranking
        if rerank and self.rerank_model and candidates:
            sentence_pairs = [[query, c["chunk"].get("text", c["chunk"].get("clean_text", ""))] for c in candidates]
            
            try:
                rerank_scores = self.rerank_model.predict(sentence_pairs)
                
                for i, score in enumerate(rerank_scores):
                    raw = float(score)  # CrossEncoder raw logit (unbounded)
                    # Sigmoid maps to (0,1); pipeline uses this for threshold
                    norm_score = 1.0 / (1.0 + np.exp(-raw))
                    candidates[i]["score_rerank"] = raw
                    candidates[i]["score_rerank_norm"] = norm_score

                candidates.sort(key=lambda x: x.get("score_rerank", -1e9), reverse=True)
            except Exception as e:
                print(f"[WARNING] Reranking failed: {e}")
                print(f"[INFO] Returning results without reranking")

        return candidates[:top_k]

