"""
LangChain retriever wrapper for the hybrid retriever.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr, Field

from .hybrid_retriever import HybridKBRetriever


class HybridKBRetrieverWrapper(BaseRetriever):
    """LangChain retriever wrapper for HybridKBRetriever."""
    
    _hybrid_retriever: HybridKBRetriever = PrivateAttr()
    return_metadata: bool = Field(default=True, description="Whether to include metadata in returned documents")
    
    def __init__(
        self,
        hybrid_retriever: HybridKBRetriever,
        return_metadata: bool = True,
        **kwargs
    ):
        super().__init__(return_metadata=return_metadata, **kwargs)
        self._hybrid_retriever = hybrid_retriever
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        chunks = self._hybrid_retriever.search(
            query=query,
            top_k=top_k,
            rerank=rerank
        )
        
        documents = []
        for chunk in chunks:
            text = chunk.get("text", chunk.get("clean_text", ""))
            
            metadata = {
                "chunk_id": chunk.get("chunk_id", ""),
                "kb_family": chunk.get("kb_family", ""),
                "doc_id": chunk.get("doc_id", ""),
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "score": chunk.get("score", 0.0),
                "sparse_score": chunk.get("sparse_score", 0.0),
                "dense_score": chunk.get("dense_score", 0.0),
                "hybrid_score": chunk.get("hybrid_score", 0.0),
            }
            
            if "rerank_score" in chunk:
                metadata["rerank_score"] = chunk.get("rerank_score")
            
            doc = Document(
                page_content=text,
                metadata=metadata if self.return_metadata else {}
            )
            documents.append(doc)
        
        return documents

