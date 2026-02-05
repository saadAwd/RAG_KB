"""
Vector database for KB chunks using ChromaDB.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

from . import loader
from .sparse_index import KB_CHUNKS_FILENAME, normalize_arabic
from .dense_index import _create_marbertv2_model, MODEL_NAME, USE_MARBERTV2


COLLECTION_NAME = "kb_chunks"
VECTOR_DB_DIR = "vector_db"


def _get_vector_db_path() -> str:
    """Get the path to the vector database directory."""
    processed_dir = loader.PROCESSED_DIR
    return os.path.join(processed_dir, VECTOR_DB_DIR)


class VectorDB:
    """Vector database wrapper for ChromaDB."""

    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection: chromadb.Collection,
        model: SentenceTransformer,
    ) -> None:
        self.client = client
        self.collection = collection
        self.model = model

    @classmethod
    def load(cls, model_name: str = MODEL_NAME, device: str = None) -> "VectorDB":
        """Load existing vector database."""
        db_path = _get_vector_db_path()
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Vector database not found at {db_path}. "
                f"Vector database should be included in the deployment package."
            )

        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            distance_metric = (collection.metadata or {}).get("hnsw:space", "l2")
            if distance_metric != "cosine":
                print(f"[WARNING] Collection is using '{distance_metric}' distance, not 'cosine'")
            else:
                print(f"[INFO] Collection using cosine distance metric [OK]")
        except Exception:
            raise FileNotFoundError(
                f"Collection '{COLLECTION_NAME}' not found. "
                f"Vector database should be included in the deployment package."
            )
        
        # Load model
        print(f"[INFO] Loading embedding model: {model_name}")
        if USE_MARBERTV2:
            print("  Using MARBERTv2 with sentence-transformers wrapper...")
            model = _create_marbertv2_model()
        else:
            model = SentenceTransformer(model_name)
        
        # Set device
        if device:
            model = model.to(device)
            print(f"[INFO] Embedding model on {device}")
        elif torch and torch.cuda.is_available():
            device = "cuda"
            model = model.to(device)
            print(f"[INFO] Embedding model on GPU ({torch.cuda.get_device_name(0)})")
        else:
            device = "cpu"
            model = model.to(device)
            print(f"[INFO] Embedding model on CPU (GPU not available)")
        
        return cls(client=client, collection=collection, model=model)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Tuple[Dict, float]]:
        """Search for similar chunks."""
        try:
            # Encode query
            query_norm = normalize_arabic(query)
            query_embedding = self.model.encode(
                [query_norm],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].tolist()

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
            )

            # Format results
            chunks_with_scores: List[Tuple[Dict, float]] = []
            
            if results.get("ids") and len(results["ids"][0]) > 0:
                for i, chunk_id in enumerate(results["ids"][0]):
                    chunk = {
                        "chunk_id": chunk_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "parent_doc_id": results["metadatas"][0][i].get("parent_doc_id", ""),
                        "kb_family": results["metadatas"][0][i].get("kb_family", ""),
                        "content_type": results["metadatas"][0][i].get("content_type", ""),
                        "title": results["metadatas"][0][i].get("title", ""),
                        "url": results["metadatas"][0][i].get("url", ""),
                        "language": results["metadatas"][0][i].get("language", "ar"),
                        "chunk_index": int(results["metadatas"][0][i].get("chunk_index", 0)),
                    }
                    
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    score = max(0.0, 1.0 - distance)
                    
                    chunks_with_scores.append((chunk, score))

            return chunks_with_scores
        except Exception as e:
            # If vector database search fails (corrupted index, etc.), raise to trigger fallback
            raise RuntimeError(f"Vector database search failed: {type(e).__name__}: {e}")

    def get_stats(self) -> Dict:
        """Get statistics about the vector database."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": COLLECTION_NAME,
                "model_name": MODEL_NAME,
            }
        except Exception as e:
            # If counting fails (e.g., corrupted index), return default stats
            raise Exception(f"Failed to get vector database stats: {type(e).__name__}: {e}")



