"""
Sparse BM25 index over KB chunks.

Uses `rank_bm25` with a simple Arabic-aware tokenizer.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Tuple

from rank_bm25 import BM25Okapi

from . import loader


KB_CHUNKS_FILENAME = "kb_chunks.jsonl"


ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]"
)


def normalize_arabic(text: str) -> str:
    """
    Very light Arabic normalization:
    - remove diacritics
    - unify different Alef forms
    - unify Yeh / dotless yeh
    """
    text = ARABIC_DIACRITICS_RE.sub("", text)
    # Normalize Alef variants
    text = re.sub("[\u0622\u0623\u0625]", "\u0627", text)  # آأإ -> ا
    # Normalize Yeh / dotless yeh / Alef maqsura
    text = re.sub("[\u0649\u064A\u06CC]", "\u064A", text)  # ى ي ی -> ي
    return text


def tokenize(text: str) -> List[str]:
    """
    Very simple tokenization: normalize then split on whitespace.
    """
    text = normalize_arabic(text)
    return text.split()


def _iter_kb_chunks(
    filename: str = KB_CHUNKS_FILENAME,
) -> Iterable[Tuple[Dict, List[str]]]:
    processed_dir = loader.PROCESSED_DIR
    path = os.path.join(processed_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"KB chunks file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            yield obj, tokenize(text)


class SparseKBIndex:
    """
    In-memory BM25 index over KB chunks.
    """

    def __init__(self, bm25: BM25Okapi, chunks: List[Dict]) -> None:
        self.bm25 = bm25
        self.chunks = chunks

    @classmethod
    def build(cls) -> "SparseKBIndex":
        chunks: List[Dict] = []
        tokenized_docs: List[List[str]] = []

        for obj, tokens in _iter_kb_chunks():
            chunks.append(obj)
            tokenized_docs.append(tokens)

        bm25 = BM25Okapi(tokenized_docs)
        return cls(bm25=bm25, chunks=chunks)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Return top_k chunks with BM25 scores for a given Arabic query.
        """
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        # Get indices of top_k scores
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]
    
    def get_stats(self) -> Dict:
        """Get statistics about the sparse index."""
        kb_families = set()
        doc_ids = set()
        for chunk in self.chunks:
            kb_families.add(chunk.get("kb_family", "unknown"))
            doc_ids.add(chunk.get("doc_id", "unknown"))
        
        return {
            "total_chunks": len(self.chunks),
            "kb_families": len(kb_families),
            "total_docs": len(doc_ids),
            "index_type": "BM25"
        }

