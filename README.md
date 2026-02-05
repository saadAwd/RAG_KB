# KB Retriever - Knowledge Base Retrieval System

Self-contained deployment package for the Arabic mental health knowledge base retrieval system using the fine-tuned **ALLaM-7B** model with RAG.

## 📋 Overview

This package provides a complete, working knowledge base retrieval system with:
- **Hybrid Retrieval**: Combines sparse (BM25) and dense (MARBERTv2) retrieval
- **Reranking**: Optional Cross-Encoder re-scoring for improved relevance
- **RAG + Fine-tuned ALLaM-7B**: Generator and evaluation framework (Config 5 = same system for experiments)
- **Gradio Interface**: Interactive web UI for testing and visualization
- **LangChain Integration**: Compatible with LangChain frameworks

---

## 🧪 Running the Full Experiment (for colleagues)

Use this workflow to reproduce and extend the evaluation with the **same system as Config 5** (RAG system prompt + default parameters).

### Prerequisites

- **Python 3.8+**
- **GPU** recommended (faster); CPU works with `requirements.txt`. For GPU use `requirements_gpu.txt`.
- **Test data**: `test.jsonl` (included), or TSV files in this folder for `--tsv-test`: `Subtask3_input_test (1).tsv`, `Subtask3_output_test (2).tsv`.

### Step 1: Install dependencies

```bash
cd RAG_KB
pip install -r requirements.txt
# Or for GPU:
# pip install -r requirements_gpu.txt
```

### Step 2: Prepare the vector database

RAG needs a built vector DB from `data/processed/kb_chunks.jsonl`. Either:

- **Option A – Build locally** (requires `kb_chunks.jsonl` in `data/processed/`):

  ```bash
  python rebuild_vector_db.py --device cpu
  # Or with GPU:
  # python rebuild_vector_db.py --device cuda
  ```

- **Option B –** If you have a pre-built `vector_db` bundle, place it at `data/processed/vector_db/`.

### Step 3: Run the model comparison (BERTScore)

This runs all 4 configurations (Base Zero-shot, Base Few-shot, Fine-tuned Zero-shot, Fine-tuned RAG) and writes results to `comparison_results/model_comparison_results.json`.

```bash
python model_comparison_bertscore.py
```

- **Using TSV test/reference files** (in this folder):

  ```bash
  python model_comparison_bertscore.py --tsv-test
  ```

- **Only Config 5 (RAG)** to re-run RAG with different parameters:

  ```bash
  python model_comparison_bertscore.py --only-config5 --rerank-threshold 0.6
  ```

- **RAG parameters** (defaults = same system as Config 5): `--rag-top-k 3`, `--rag-alpha 0.8`, `--rerank-threshold 0.6`.

### Step 4: Run RAG diagnostics (optional)

Analyse when RAG helps vs hurts (KB usage, BERTScore by usage, per-question delta) **without** re-running generation:

```bash
python run_rag_diagnostics.py
# Or specify results file:
python run_rag_diagnostics.py --file comparison_results/model_comparison_results.json
```

### Reproducibility

- **Same system as Config 5**: Use the RAG system prompt in `kb_retriever/generator.py` (`_build_rag_prompt`) and default RAG settings (`top_k=3`, `alpha=0.8`, `rerank_threshold=0.6`) so your results are comparable.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Gradio Interface

```bash
python app.py
```

The interface will be available at `http://127.0.0.1:7860`

### 3. For External Access (e.g., Oracle Cloud)

```bash
python app.py --server-name 0.0.0.0 --server-port 7860
```

## 📁 Package Structure

```
kb_retriever_deployment/
├── kb_retriever/          # Main package
│   ├── __init__.py
│   ├── config.py          # Configuration
│   ├── loader.py          # Data loading utilities
│   ├── sparse_index.py    # BM25 sparse retrieval
│   ├── dense_index.py     # MARBERTv2 dense retrieval
│   ├── vector_db.py       # ChromaDB vector database
│   ├── hybrid_retriever.py # Hybrid retrieval logic
│   ├── langchain_retriever.py # LangChain wrapper
│   └── gradio_retriever_test.py # Gradio interface
├── data/
│   └── processed/
│       ├── kb_chunks.jsonl    # Knowledge base chunks
│       └── vector_db/          # ChromaDB vector database
├── app.py                 # Main entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

Configuration is managed in `kb_retriever/config.py`:

- **HYBRID_ALPHA**: Weight for sparse vs dense retrieval (default: 0.4 = 40% sparse, 60% dense)
- **USE_RERANKING**: Enable/disable reranking (default: True)
- **RERANKING_MODEL**: Model for reranking (default: "BAAI/bge-reranker-base")
- **RERANKING_TOP_K**: Number of candidates to rerank (default: 10)

You can override these via environment variables:
```bash
export RAG_HYBRID_ALPHA=0.5
export RAG_USE_RERANKING=true
python app.py
```

## 📊 Features

### Gradio Interface

The Gradio interface provides:
- **Search Tab**: Query the knowledge base with configurable parameters
- **Comparison Tab**: Compare sparse, dense, and hybrid retrieval methods
- **Statistics Tab**: View knowledge base statistics

### Retrieval Methods

1. **Sparse (BM25)**: Keyword-based retrieval using BM25 algorithm
2. **Dense (MARBERTv2)**: Semantic retrieval using Arabic BERT embeddings
3. **Hybrid**: Weighted combination of sparse and dense scores
4. **Reranking**: Cross-Encoder re-scoring of top candidates

## 🐳 Deployment

### Local Development

```bash
python app.py
```

### Oracle Cloud / Remote Server

1. Upload the entire `kb_retriever_deployment` folder to your server
2. SSH into the server
3. Install dependencies: `pip install -r requirements.txt`
4. Run with external access: `python app.py --server-name 0.0.0.0`
5. Access via `http://YOUR_SERVER_IP:7860`

### Using systemd (Linux)

Create `/etc/systemd/system/kb-retriever.service`:

```ini
[Unit]
Description=KB Retriever Gradio App
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/kb_retriever_deployment
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python app.py --server-name 0.0.0.0 --server-port 7860
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable kb-retriever
sudo systemctl start kb-retriever
```

## 📝 Usage Examples

### Python API

```python
from kb_retriever.hybrid_retriever import HybridKBRetriever

# Initialize retriever
retriever = HybridKBRetriever.build(alpha=0.4)

# Search
results = retriever.search("ما هو الاكتئاب؟", top_k=5, rerank=True)

# Access results
for result in results:
    chunk = result["chunk"]
    print(f"Title: {chunk['title']}")
    print(f"Text: {chunk['text'][:200]}...")
    print(f"Hybrid Score: {result['score_hybrid']:.4f}")
```

### LangChain Integration

```python
from kb_retriever.langchain_retriever import HybridKBRetrieverWrapper

# Create LangChain retriever
retriever = HybridKBRetrieverWrapper.build(alpha=0.4)

# Use with LangChain chains
from langchain.chains import RetrievalQA

# ... your chain setup
```

## 🔍 Knowledge Base Contents

The knowledge base includes:
- **Articles**: Mental health articles
- **Books**: Mental health books
- **QA Pairs**: Question-answer pairs from Shifaa

All content is in Arabic and has been chunked and indexed for retrieval.

## 📦 What's Included

- ✅ All Python code for retrieval system and **evaluation framework**
- ✅ Pre-built chunks (`data/processed/kb_chunks.jsonl`)
- ✅ Vector DB build script (`rebuild_vector_db.py`); pre-built `vector_db/` can be built or provided separately
- ✅ **Model comparison** (`model_comparison_bertscore.py`), **RAG diagnostics** (`run_rag_diagnostics.py`)
- ✅ Fine-tuned model adapter (`Model/Allam7B-Physiology-RAG-finetuned-final`)
- ✅ Gradio interface (`app.py`)
- ✅ LangChain integration
- ✅ Configuration in `kb_retriever/config.py`; full experiment steps in this README
- ✅ Dependencies list (`requirements.txt`, `requirements_gpu.txt`)

## ⚠️ Requirements

- Python 3.8+
- ~2GB RAM minimum (4GB+ recommended)
- GPU optional (CPU works fine, but slower)
- Internet connection for first run (to download models)

## 🐛 Troubleshooting

### Vector Database Not Found

If you see "Vector database not found", ensure:
1. The `data/processed/vector_db/` directory exists
2. It contains the ChromaDB collection files

### Model Download Issues

On first run, models will be downloaded from Hugging Face:
- MARBERTv2: ~500MB
- BGE-Reranker: ~400MB

Ensure you have internet connection and sufficient disk space.

### Memory Issues

If you encounter memory errors:
- Reduce batch sizes in `dense_index.py`
- Use CPU instead of GPU
- Reduce `top_k` in searches

## 📄 License

[Add your license information here]

## 👥 Credits

Knowledge base retrieval system for Arabic mental health content.

