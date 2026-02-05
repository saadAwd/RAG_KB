"""
Dense embedding index over KB chunks.

Uses `sentence-transformers` with MARBERTv2 for Arabic.
Uses vector database (ChromaDB) if available.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from tqdm import tqdm

from . import loader
from .sparse_index import KB_CHUNKS_FILENAME, normalize_arabic


MODEL_NAME = "UBC-NLP/MARBERTv2"
USE_MARBERTV2 = True


def _create_marbertv2_model():
    """Create a SentenceTransformer model from MARBERTv2."""
    word_embedding_model = Transformer("UBC-NLP/MARBERTv2", max_seq_length=512)
    pooling_model = Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


def _load_kb_chunks(filename: str = KB_CHUNKS_FILENAME) -> List[Dict]:
    processed_dir = loader.PROCESSED_DIR
    path = os.path.join(processed_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"KB chunks file not found: {path}")

    chunks: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


class DenseKBIndex:
    """
    Dense index for KB chunks using sentence-transformers.
    
    Uses vector database (ChromaDB) if available.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        chunks: List[Dict] = None,
        embeddings: np.ndarray = None,
        vector_db=None,
    ) -> None:
        self.model = model
        self.chunks = chunks
        self.embeddings = embeddings
        self.vector_db = vector_db

    @classmethod
    def build(cls, use_vector_db: bool = True, device: str = None) -> "DenseKBIndex":
        """Build dense index, preferring vector database if available."""
        # Load model
        if USE_MARBERTV2:
            print(f"Loading MARBERTv2 with sentence-transformers wrapper...")
            model = _create_marbertv2_model()
        else:
            model = SentenceTransformer(MODEL_NAME)
        
        # Check if GPU is available
        try:
            import torch
            if torch and torch.cuda.is_available():
                device = "cuda"
                model = model.to(device)
                print(f"[INFO] Using GPU ({torch.cuda.get_device_name(0)}) for embedding model")
            else:
                device = "cpu"
                model = model.to(device)
                print(f"[INFO] Using CPU for embedding model (GPU not available)")
        except ImportError:
            device = "cpu"
            model = model.to(device)
            print(f"[INFO] Using CPU for embedding model")
        
        # Try to use vector database first
        if use_vector_db:
            try:
                from .vector_db import VectorDB
                import chromadb.errors
                print("Loading vector database...")
                vector_db = VectorDB.load(device=device if device else None)
                # Try to get stats, but don't fail if it's corrupted
                stats_failed = False
                try:
                    stats = vector_db.get_stats()
                    print(f"[OK] Loaded vector database with {stats['total_chunks']} chunks")
                except Exception as stats_error:
                    print(f"⚠ Warning: Could not get vector database stats: {stats_error}")
                    print(f"[OK] Loaded vector database (stats unavailable)")
                    stats_failed = True
                
                # If stats failed, the database might be corrupted - load chunks/embeddings as backup
                # But make it optional - only load if vector_db search actually fails
                # This avoids the long encoding time if vector_db still works for searches
                if stats_failed:
                    print("⚠ Vector database stats unavailable (may be corrupted)")
                    print("⚠ Will attempt to use vector_db for searches; will load backup only if search fails")
                    # Don't load backup now - load lazily if search fails
                    return cls(model=model, vector_db=vector_db)
                else:
                    return cls(model=model, vector_db=vector_db)
            except (FileNotFoundError, ImportError) as e:
                if isinstance(e, ImportError):
                    print(f"⚠ ChromaDB not available, falling back to in-memory index")
                else:
                    print(f"⚠ Vector database not found, falling back to in-memory index")
            except Exception as e:
                # Catch any other ChromaDB errors (corruption, etc.)
                error_msg = str(e)
                print(f"⚠ Error loading vector database: {error_msg}")
                if "file is not a database" in error_msg or "code: 26" in error_msg:
                    print(f"⚠ Vector database file appears corrupted.")
                    print(f"⚠ To fix: Run 'python rebuild_vector_db.py --device cpu --force'")
                    print(f"⚠ Falling back to in-memory index (will rebuild embeddings)")
                else:
                    print(f"⚠ Falling back to in-memory index")
        
        # Fallback to in-memory computation
        print(f"Loading model: {MODEL_NAME}")
        chunks = _load_kb_chunks()
        texts = [normalize_arabic(c.get("text", "")) for c in chunks]

        print(f"Encoding {len(texts)} chunks...")
        print(f"Using batch size 256 for faster CPU encoding (this will take 15-20 minutes)...")
        
        # Encode in batches with progress tracking
        batch_size = 256
        embeddings_list = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            print(f"[Progress] Encoding batch {batch_num}/{total_batches} ({batch_num * 100 // total_batches}%) - {len(batch_texts)} chunks...")
            
            batch_embeddings = model.encode(
                batch_texts,
                batch_size=32,  # Internal batch size
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings_list.append(batch_embeddings)
        
        # Concatenate all batches
        embeddings = np.concatenate(embeddings_list, axis=0)
        print(f"[OK] Encoding completed! Total embeddings: {len(embeddings)}")

        return cls(model=model, chunks=chunks, embeddings=embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Return top_k chunks with cosine similarity scores."""
        # Use vector database if available
        if self.vector_db is not None:
            try:
                return self.vector_db.search(query, top_k=top_k)
            except (RuntimeError, Exception) as e:
                # If vector database search fails, load backup if not already loaded
                if self.chunks is None or self.embeddings is None:
                    print(f"⚠ Vector database search failed: {type(e).__name__}")
                    print("⚠ Loading chunks/embeddings as fallback (this may take a few minutes)...")
                    chunks = _load_kb_chunks()
                    texts = [normalize_arabic(c.get("text", "")) for c in chunks]
                    print(f"Encoding {len(texts)} chunks...")
                    print(f"Using batch size 256 for faster CPU encoding (this will take 15-20 minutes)...")
                    self.chunks = chunks
                    
                    # Encode in batches with progress tracking
                    batch_size = 256
                    embeddings_list = []
                    total_batches = (len(texts) + batch_size - 1) // batch_size
                    
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]
                        batch_num = (i // batch_size) + 1
                        print(f"[Progress] Encoding batch {batch_num}/{total_batches} ({batch_num * 100 // total_batches}%) - {len(batch_texts)} chunks...")
                        
                        batch_embeddings = self.model.encode(
                            batch_texts,
                            batch_size=32,  # Internal batch size
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        )
                        embeddings_list.append(batch_embeddings)
                    
                    # Concatenate all batches
                    self.embeddings = np.concatenate(embeddings_list, axis=0)
                    print(f"[OK] Encoding completed! Total embeddings: {len(self.embeddings)}")
                    print("⚠ Fallback embeddings loaded, retrying search...")
                # Continue to in-memory computation below
        
        # Fallback to in-memory computation (or primary method if vector_db not used)
        if self.chunks is None or self.embeddings is None:
            raise RuntimeError("No search method available: vector_db failed and in-memory chunks not loaded")
        
        q_norm = normalize_arabic(query)
        q_vec = self.model.encode(
            [q_norm],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = np.dot(self.embeddings, q_vec)

        top_indices = np.argsort(-scores)[:top_k]
        return [
            (self.chunks[int(i)], float(scores[int(i)])) for i in top_indices
        ]



