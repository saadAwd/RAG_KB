"""
RAG Pipeline: Complete Retrieval-Augmented Generation system.
Integrates retrieval, reranking, and generation with proper context handling.
"""

import gc
import logging
import math
from typing import Dict, List, Optional, Tuple

from .hybrid_retriever import HybridKBRetriever
from .generator import RAGGenerator, detect_device

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline: retrieval -> reranking -> generation."""
    
    def __init__(
        self,
        retriever: HybridKBRetriever,
        generator: RAGGenerator,
        rerank_threshold: float = 0.1,
        max_context_chunks: int = 3,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: HybridKBRetriever instance
            generator: RAGGenerator instance
            rerank_threshold: Minimum reranker score to use KB context (default: 0.1)
            max_context_chunks: Maximum number of chunks to use for generation (default: 3)
        """
        self.retriever = retriever
        self.generator = generator
        self.rerank_threshold = rerank_threshold
        self.max_context_chunks = max_context_chunks
        
        logger.info(f"[RAG] Pipeline initialized (rerank_threshold={rerank_threshold}, max_context_chunks={max_context_chunks})")
    
    def process(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.4,
        rerank: bool = True,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            alpha: Hybrid retrieval alpha (0=all dense, 1=all sparse)
            rerank: Whether to use reranking
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
        
        Returns:
            Dictionary with:
            - response: Generated response text
            - used_kb_context: Whether KB context was used
            - rerank_scores: List of reranker scores
            - num_chunks: Number of chunks used
            - retrieval_info: Information about retrieval
        """
        logger.info(f"[RAG] Processing query: {query[:100]}...")
        
        # Step 1: Retrieve chunks
        logger.info(f"[RAG] Retrieving chunks (top_k={top_k}, alpha={alpha}, rerank={rerank})...")
        chunks = self.retriever.search(
            query=query,
            top_k=top_k,
            rerank=rerank
        )
        
        if not chunks:
            logger.info("[RAG] No chunks retrieved, using general knowledge")
            response = self.generator.generate(
                query=query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            return {
                "response": response,
                "used_kb_context": False,
                "rerank_scores": [],
                "num_chunks": 0,
                "retrieval_info": "No chunks retrieved",
            }
        
        # Step 2: Check reranker scores
        rerank_scores = []
        rerank_scores_norm = []
        max_rerank_score = -1e9
        max_rerank_score_norm = -1.0
        
        for chunk_result in chunks:
            rerank_score = chunk_result.get("score_rerank")
            rerank_score_norm = chunk_result.get("score_rerank_norm")
            if rerank_score is not None:
                rerank_scores.append(rerank_score)
                max_rerank_score = max(max_rerank_score, rerank_score)
            if rerank_score_norm is not None:
                rerank_scores_norm.append(rerank_score_norm)
                max_rerank_score_norm = max(max_rerank_score_norm, rerank_score_norm)
        
        # Step 3: Decide whether to use KB context
        # rerank_threshold is in [0,1]; compared to score_rerank_norm = sigmoid(CrossEncoder raw)
        if rerank_scores_norm:
            use_kb_context = max_rerank_score_norm > self.rerank_threshold
            threshold_score = max_rerank_score_norm
        elif rerank_scores:
            # Fallback: compute sigmoid(raw) so threshold 0–1 is comparable; fill norm list for return
            rerank_scores_norm = [1.0 / (1.0 + math.exp(-float(s))) for s in rerank_scores]
            threshold_score = max(rerank_scores_norm)
            use_kb_context = threshold_score > self.rerank_threshold
        else:
            use_kb_context = False
            threshold_score = -1.0
        
        logger.info(f"[RAG] Max reranker score: {threshold_score:.4f}, threshold: {self.rerank_threshold}, using KB context: {use_kb_context}")
        
        # Step 4: Generate response
        if use_kb_context:
            # Use KB context - limit to top chunks
            context_chunks = chunks[:self.max_context_chunks]
            
            # Extract chunk dictionaries for generator
            chunk_dicts = []
            for chunk_result in context_chunks:
                chunk = chunk_result.get("chunk", chunk_result)
                chunk_dicts.append(chunk)
            
            logger.info(f"[RAG] Using KB context with {len(chunk_dicts)} chunks")
            
            # Free memory before generation
            gc.collect()
            
            response = self.generator.generate(
                query=query,
                chunks=chunk_dicts,  # Pass chunks directly - generator will summarize
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            return {
                "response": response,
                "used_kb_context": True,
                "rerank_scores": rerank_scores[:len(context_chunks)],
                "rerank_scores_norm": rerank_scores_norm[:len(context_chunks)],
                "num_chunks": len(chunk_dicts),
                "retrieval_info": f"Used {len(chunk_dicts)} chunks (max rerank score: {threshold_score:.4f})",
            }
        else:
            # Use general knowledge
            logger.info("[RAG] Reranker scores below threshold, using general knowledge")
            
            # Free memory before generation
            gc.collect()
            
            response = self.generator.generate(
                query=query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            return {
                "response": response,
                "used_kb_context": False,
                "rerank_scores": rerank_scores,
                "rerank_scores_norm": rerank_scores_norm,
                "num_chunks": 0,
                "retrieval_info": f"General knowledge (max rerank score: {threshold_score:.4f} <= {self.rerank_threshold})",
            }
    
    @classmethod
    def build(
        cls,
        retriever_alpha: float = 0.4,
        retriever_use_cpu: bool = None,
        generator_device: str = None,
        generator_load_in_4bit: bool = None,
        rerank_threshold: float = 0.1,
        max_context_chunks: int = 3,
    ) -> "RAGPipeline":
        """
        Build a RAG pipeline with auto-detected settings.
        
        Args:
            retriever_alpha: Hybrid retrieval alpha
            retriever_use_cpu: Whether to use CPU for retriever (auto-detect if None)
            generator_device: Device for generator (auto-detect if None)
            generator_load_in_4bit: Whether to use 4-bit quantization (auto-detect if None)
            rerank_threshold: Minimum reranker score to use KB context
            max_context_chunks: Maximum number of chunks to use for generation
        """
        # Auto-detect device if not specified
        if generator_device is None:
            generator_device = detect_device()
        
        if generator_load_in_4bit is None:
            generator_load_in_4bit = (generator_device == "cuda")
        
        if retriever_use_cpu is None:
            retriever_use_cpu = (generator_device == "cpu")  # Use CPU for retriever if generator is on CPU
        
        logger.info(f"[RAG] Building pipeline (retriever_use_cpu={retriever_use_cpu}, generator_device={generator_device}, generator_load_in_4bit={generator_load_in_4bit})")
        
        # Build retriever
        retriever = HybridKBRetriever.build(alpha=retriever_alpha, use_cpu=retriever_use_cpu)
        
        # Build generator
        generator = RAGGenerator.build(
            device=generator_device,
            load_in_4bit=generator_load_in_4bit
        )
        generator.initialize()
        
        return cls(
            retriever=retriever,
            generator=generator,
            rerank_threshold=rerank_threshold,
            max_context_chunks=max_context_chunks,
        )
