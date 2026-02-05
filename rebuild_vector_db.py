#!/usr/bin/env python3
"""
Script to rebuild the vector database from kb_chunks.jsonl.
This ensures the database is built on CPU to avoid GPU/CPU compatibility issues.
"""

import os
import json
import shutil
from pathlib import Path

# Set CPU threading for optimal performance (use all 4 cores)
# MUST be set before importing torch
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['TORCH_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import torch and set threading after import
try:
    import torch
    if torch is not None:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        print(f"[INFO] PyTorch threads set to: {torch.get_num_threads()}")
except ImportError:
    pass

from kb_retriever import loader
from kb_retriever.sparse_index import KB_CHUNKS_FILENAME, normalize_arabic
from kb_retriever.dense_index import _create_marbertv2_model, MODEL_NAME, USE_MARBERTV2

COLLECTION_NAME = "kb_chunks"
VECTOR_DB_DIR = "vector_db"


def get_vector_db_path():
    """Get the path to the vector database directory."""
    processed_dir = loader.PROCESSED_DIR
    return os.path.join(processed_dir, VECTOR_DB_DIR)


def load_kb_chunks():
    """Load all KB chunks from jsonl file."""
    processed_dir = loader.PROCESSED_DIR
    chunks_path = os.path.join(processed_dir, KB_CHUNKS_FILENAME)
    
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"KB chunks file not found: {chunks_path}")
    
    chunks = []
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def rebuild_vector_db(force: bool = False, device: str = "cpu"):
    """Rebuild the vector database from scratch."""
    db_path = get_vector_db_path()
    
    # Check if database already exists
    if os.path.exists(db_path):
        if force:
            print(f"Removing existing vector database at {db_path}...")
            shutil.rmtree(db_path)
        else:
            response = input(f"Vector database already exists at {db_path}. Delete and rebuild? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            shutil.rmtree(db_path)
    
    # Create directory
    os.makedirs(db_path, exist_ok=True)
    print(f"Creating vector database at {db_path}...")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Create or get collection
    print(f"Creating collection '{COLLECTION_NAME}'...")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    # Load embedding model
    print(f"Loading embedding model: {MODEL_NAME}")
    if USE_MARBERTV2:
        print("  Using MARBERTv2 with sentence-transformers wrapper...")
        model = _create_marbertv2_model()
    else:
        model = SentenceTransformer(MODEL_NAME)
    
    # Force CPU
    if device == "cpu":
        model = model.to("cpu")
        print("[INFO] Using CPU for embeddings (as requested)")
    else:
        try:
            import torch
            if torch and torch.cuda.is_available():
                model = model.to("cuda")
                print(f"[INFO] Using GPU ({torch.cuda.get_device_name(0)}) for embeddings")
            else:
                model = model.to("cpu")
                print("[INFO] Using CPU for embeddings (GPU not available)")
        except ImportError:
            model = model.to("cpu")
            print("[INFO] Using CPU for embeddings")
    
    # Load chunks
    chunks = load_kb_chunks()
    
    # Prepare data for batch insertion
    print("Preparing chunks for embedding...")
    texts = []
    metadatas = []
    ids = []
    
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", f"chunk_{len(ids)}")
        text = chunk.get("text", chunk.get("clean_text", ""))
        normalized_text = normalize_arabic(text)
        
        texts.append(normalized_text)
        # Ensure all metadata values are strings (ChromaDB doesn't accept None)
        metadatas.append({
            "parent_doc_id": str(chunk.get("parent_doc_id") or ""),
            "kb_family": str(chunk.get("kb_family") or ""),
            "content_type": str(chunk.get("content_type") or ""),
            "title": str(chunk.get("title") or ""),
            "url": str(chunk.get("url") or ""),
            "language": str(chunk.get("language") or "ar"),
            "chunk_index": int(chunk.get("chunk_index") or 0),
        })
        ids.append(chunk_id)
    
    # Encode in batches (optimized for device)
    print(f"Encoding {len(texts)} chunks (this will take a while)...")
    
    # Auto-detect device and optimize batch sizes
    try:
        import torch
        is_gpu = torch.cuda.is_available() and str(model.device) != "cpu"
    except:
        is_gpu = False
    
    if is_gpu:
        # GPU: Use smaller batches to avoid OOM (especially if generator model is loaded)
        chunk_size = 500  # Process 500 chunks at a time (reduced from 2000)
        internal_batch = 128  # Smaller internal batch to avoid OOM (reduced from 512)
        print(f"🚀 GPU mode: Processing in batches of {chunk_size} with internal batch {internal_batch}...")
        print(f"   Expected time: 5-10 minutes (much faster than CPU)")
        print(f"   ⚠️  Note: If you get OOM errors, stop the service first: sudo systemctl stop kb-retriever")
    else:
        # CPU: Conservative batches for 16GB RAM
        chunk_size = 1000
        internal_batch = 256
        print(f"💻 CPU mode: Processing in {chunk_size} chunks with internal batch {internal_batch}...")
        try:
            import torch
            torch_threads = torch.get_num_threads()
        except:
            torch_threads = 'N/A'
        print(f"   Using all CPU cores (PyTorch threads: {torch_threads}) and ~6-8GB RAM")
        print(f"   Expected time: 30-60 minutes")
    
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size
    
    embeddings = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Encoding", total=total_chunks):
        chunk_texts = texts[i:i + chunk_size]
        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=internal_batch,  # Larger internal batch for better CPU/memory utilization
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings.extend(chunk_embeddings.tolist())
    
    # Insert into ChromaDB in batches
    print("Inserting embeddings into vector database...")
    insert_batch_size = 1000
    
    for i in tqdm(range(0, len(ids), insert_batch_size), desc="Inserting batches"):
        batch_ids = ids[i:i + insert_batch_size]
        batch_embeddings = embeddings[i:i + insert_batch_size]
        batch_metadatas = metadatas[i:i + insert_batch_size]
        batch_texts = texts[i:i + insert_batch_size]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_texts,
        )
    
    # Verify
    count = collection.count()
    print(f"\n[OK] Vector database rebuilt successfully!")
    print(f"  - Total chunks: {count}")
    print(f"  - Location: {db_path}")
    print(f"  - Collection: {COLLECTION_NAME}")
    print(f"  - Distance metric: cosine")
    print(f"  - Built on: {device}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rebuild vector database from kb_chunks.jsonl")
    parser.add_argument("--force", action="store_true", help="Force rebuild without confirmation")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                       help="Device to use for embeddings (default: cpu)")
    args = parser.parse_args()
    
    rebuild_vector_db(force=args.force, device=args.device)

