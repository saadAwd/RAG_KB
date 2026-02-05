"""
Configuration for KB Retriever.
"""

import os


class RAGConfig:
    """Configuration for RAG system."""
    
    # Hybrid Retrieval Configuration
    HYBRID_ALPHA: float = float(os.getenv("RAG_HYBRID_ALPHA", "0.4"))  # 40% sparse, 60% dense
    
    # Reranking Configuration
    USE_RERANKING: bool = os.getenv("RAG_USE_RERANKING", "true").lower() == "true"
    RERANKING_MODEL: str = os.getenv("RAG_RERANKING_MODEL", "BAAI/bge-reranker-base")
    RERANKING_TOP_K: int = int(os.getenv("RAG_RERANKING_TOP_K", "10"))
    
    # Reranker Threshold Configuration
    RERANKER_THRESHOLD: float = float(os.getenv("RAG_RERANKER_THRESHOLD", "0.1"))  # Threshold for using KB context
    
    # Generator Configuration
    GENERATOR_MODEL_PATH: str = os.getenv("RAG_GENERATOR_MODEL_PATH", "")  # Empty = use default
    GENERATOR_DEVICE: str = os.getenv("RAG_GENERATOR_DEVICE", "cpu")  # "cpu" or "cuda"
    GENERATOR_LOAD_IN_4BIT: bool = os.getenv("RAG_GENERATOR_4BIT", "false").lower() == "true"
    GENERATOR_MAX_NEW_TOKENS: int = int(os.getenv("RAG_GENERATOR_MAX_TOKENS", "512"))
    GENERATOR_TEMPERATURE: float = float(os.getenv("RAG_GENERATOR_TEMPERATURE", "0.7"))
    GENERATOR_TOP_P: float = float(os.getenv("RAG_GENERATOR_TOP_P", "0.9"))

