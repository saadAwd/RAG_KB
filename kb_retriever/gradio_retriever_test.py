"""
Gradio interface for testing and visualizing the hybrid retriever.
Optimized with pre-initialization and request queuing.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import gradio as gr

from .hybrid_retriever import HybridKBRetriever
from .langchain_retriever import HybridKBRetrieverWrapper
from .rag_pipeline import RAGPipeline

# Add parent directory to path for manual_evaluation import
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from manual_evaluation import ManualEvaluator
    MANUAL_EVAL_AVAILABLE = True
except ImportError:
    ManualEvaluator = None
    MANUAL_EVAL_AVAILABLE = False
    print("Warning: manual_evaluation module not found. Manual evaluation tab will not be available.")


class RequestQueue:
    """Thread-safe queue manager for handling multiple requests sequentially."""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.processing = False
        self.current_request_id = None
        self.request_positions = {}
        self.lock = threading.Lock()
        self.worker_thread = None
        self.results = {}
        self.status_updates = {}
    
    def add_request(self, request_id: str, query: str, top_k: int, alpha: float, rerank: bool, tester):
        """Add a request to the queue."""
        with self.lock:
            position = self.queue.qsize() + (1 if self.processing else 0)
            self.request_positions[request_id] = position
            if position > 0:
                self.status_updates[request_id] = f"⏳ في قائمة الانتظار: {position} طلب أمامك"
            else:
                self.status_updates[request_id] = "🔄 جاري المعالجة..."
        
        self.queue.put({
            'request_id': request_id,
            'query': query,
            'top_k': top_k,
            'alpha': alpha,
            'rerank': rerank,
            'tester': tester
        })
        
        if not self.processing and (self.worker_thread is None or not self.worker_thread.is_alive()):
            print("[DEBUG] Starting queue worker thread")
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()
            print("[DEBUG] Queue worker thread started successfully")
    
    def _process_queue(self):
        """Process requests from the queue sequentially."""
        print("[DEBUG] Queue worker thread started")
        while True:
            try:
                request = self.queue.get(timeout=1)
                if request is None:
                    break
                
                request_id = request['request_id']
                print(f"[DEBUG] Processing request: {request_id}, query: {request['query'][:50]}...")
                
                with self.lock:
                    self.processing = True
                    self.current_request_id = request_id
                    for rid in self.request_positions:
                        if rid != request_id:
                            self.request_positions[rid] = max(0, self.request_positions[rid] - 1)
                            if self.request_positions[rid] > 0:
                                self.status_updates[rid] = f"⏳ في قائمة الانتظار: {self.request_positions[rid]} طلب أمامك"
                            else:
                                self.status_updates[rid] = "🔄 جاري المعالجة..."
                
                self.status_updates[request_id] = "🔄 جاري المعالجة..."
                print(f"[DEBUG] Starting search for request: {request_id}")
                
                try:
                    html, json_data = request['tester'].search(
                        query=request['query'],
                        top_k=request['top_k'],
                        alpha=request['alpha'],
                        rerank=request['rerank']
                    )
                    print(f"[DEBUG] Search completed for request: {request_id}")
                    self.results[request_id] = (html, json_data, None)
                except Exception as e:
                    error_msg = f"Error during retrieval: {str(e)}"
                    print(f"[ERROR] Search failed for request {request_id}: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    self.results[request_id] = (error_msg, "{}", None)
                
                with self.lock:
                    self.request_positions.pop(request_id, None)
                    self.status_updates.pop(request_id, None)
                    if self.queue.empty():
                        self.processing = False
                        self.current_request_id = None
                    else:
                        for rid in list(self.request_positions.keys()):
                            if self.request_positions[rid] > 0:
                                self.request_positions[rid] -= 1
                                if self.request_positions[rid] > 0:
                                    self.status_updates[rid] = f"⏳ في قائمة الانتظار: {self.request_positions[rid]} طلب أمامك"
                                else:
                                    self.status_updates[rid] = "🔄 جاري المعالجة..."
                
                self.queue.task_done()
                
            except queue.Empty:
                with self.lock:
                    if self.queue.empty():
                        self.processing = False
                        self.current_request_id = None
                        break
    
    def get_status(self, request_id: str) -> str:
        """Get current status for a request."""
        with self.lock:
            return self.status_updates.get(request_id, "⏳ في قائمة الانتظار...")
    
    def get_result(self, request_id: str, timeout: float = 300.0) -> Optional[Tuple[str, str]]:
        """Get result for a request, waiting if necessary."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.results:
                result = self.results.pop(request_id)
                return result[0], result[1]
            time.sleep(0.5)
        return None, "{}"


# Global queue instance
request_queue = RequestQueue()


class RetrieverTester:
    """Wrapper class to manage retriever state and provide testing interface."""
    
    def __init__(self):
        self.retriever: HybridKBRetriever = None
        self.retriever_wrapper: HybridKBRetrieverWrapper = None
        self.rag_pipeline: RAGPipeline = None
        self._initialized = False
        self._rag_initialized = False
    
    def initialize(self, alpha: float = 0.4, use_cpu: bool = True):
        """Initialize the retriever."""
        if not self._initialized or self.retriever is None:
            print(f"[INFO] Initializing HybridKBRetriever with alpha={alpha}...")
            self.retriever = HybridKBRetriever.build(alpha=alpha, use_cpu=use_cpu)
            self.retriever_wrapper = HybridKBRetrieverWrapper(
                hybrid_retriever=self.retriever,
                return_metadata=True
            )
            self._initialized = True
            print("[OK] Retriever initialized")
        else:
            if self.retriever.alpha != alpha:
                print(f"[INFO] Updating alpha from {self.retriever.alpha} to {alpha}")
                self.retriever.alpha = alpha
    
    def initialize_rag(self, alpha: float = 0.8, rerank_threshold: float = 0.6):
        """Initialize the RAG pipeline (retriever + generator)."""
        if not self._rag_initialized or self.rag_pipeline is None:
            print(f"[INFO] Initializing RAG Pipeline (this may take a few minutes)...")
            print("[INFO] Loading generator model...")
            self.rag_pipeline = RAGPipeline.build(
                retriever_alpha=alpha,
                rerank_threshold=rerank_threshold,
                max_context_chunks=3,
            )
            self._rag_initialized = True
            print("[OK] RAG Pipeline initialized")
        else:
            if self.rag_pipeline.retriever.alpha != alpha:
                print(f"[INFO] Updating alpha from {self.rag_pipeline.retriever.alpha} to {alpha}")
                self.rag_pipeline.retriever.alpha = alpha
            if self.rag_pipeline.rerank_threshold != rerank_threshold:
                print(f"[INFO] Updating rerank threshold from {self.rag_pipeline.rerank_threshold} to {rerank_threshold}")
                self.rag_pipeline.rerank_threshold = rerank_threshold
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.4,
        show_scores: bool = True,
        rerank: bool = True
    ) -> Tuple[str, str]:
        """Search the knowledge base and return formatted results."""
        if not query or not query.strip():
            return "Please enter a query.", "{}"
        
        if self._initialized and self.retriever.alpha != alpha:
            self.retriever.alpha = alpha
        
        try:
            chunks = self.retriever.search(
                query=query.strip(),
                top_k=top_k,
                rerank=rerank
            )
        except Exception as e:
            error_msg = f"Error during retrieval: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg, "{}"
        
        if not chunks:
            return "No results found.", "{}"
        
        html_parts = []
        html_parts.append(f"<h3 style='color: #ffffff;'>Query: <em style='color: #66b3ff; direction: rtl; text-align: right; display: inline-block;'>{query}</em></h3>")
        html_parts.append(f"<p style='color: #e0e0e0;'><strong style='color: #ffffff;'>Found {len(chunks)} results</strong></p>")
        html_parts.append("<hr style='border-color: #555;'>")
        
        for i, result_item in enumerate(chunks, 1):
            chunk = result_item.get("chunk", result_item)
            
            text = chunk.get("text", chunk.get("clean_text", ""))
            title = chunk.get("title", "No title")
            kb_family = chunk.get("kb_family", "unknown")
            chunk_id = chunk.get("chunk_id", "unknown")
            url = chunk.get("url", "")
            
            hybrid_score_original = result_item.get("score_hybrid_original", result_item.get("score_hybrid", result_item.get("hybrid_score", 0.0)))
            hybrid_score = result_item.get("score_hybrid", result_item.get("hybrid_score", 0.0))
            sparse_score = result_item.get("score_sparse", result_item.get("sparse_score", 0.0))
            dense_score = result_item.get("score_raw_dense", result_item.get("score_dense", result_item.get("dense_score", 0.0)))
            rerank_score = result_item.get("score_rerank", result_item.get("rerank_score"))
            
            dense_retrieved = dense_score > 1e-6
            sparse_retrieved = sparse_score > 1e-6 or (dense_score <= 1e-6)
            
            html_parts.append(f"<div style='margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #2b2b2b; color: #e0e0e0;'>")
            html_parts.append(f"<h4 style='color: #ffffff; margin-top: 0;'>Result #{i}</h4>")
            
            html_parts.append("<div style='background: #3a3a3a; padding: 10px; margin-bottom: 10px; border-radius: 3px; color: #e0e0e0;'>")
            html_parts.append(f"<strong style='color: #ffffff;'>Scores:</strong><br>")
            if rerank_score is not None:
                html_parts.append(f"  • Hybrid (original): <span style='color: #66b3ff; font-weight: bold;'>{hybrid_score_original:.4f}</span> <span style='color: #888; font-size: 0.9em;'>(before rerank)</span><br>")
            else:
                html_parts.append(f"  • Hybrid: <span style='color: #66b3ff; font-weight: bold;'>{hybrid_score:.4f}</span><br>")
            
            sparse_display = f"{sparse_score:.4f}" if sparse_retrieved else f"{sparse_score:.4f} <span style='color: #888; font-size: 0.9em;'>(not retrieved)</span>"
            html_parts.append(f"  • Sparse (BM25): <span style='color: #ff9966; font-weight: bold;'>{sparse_display}</span><br>")
            
            dense_display = f"{dense_score:.4f}" if dense_retrieved else f"{dense_score:.4f} <span style='color: #888; font-size: 0.9em;'>(not retrieved)</span>"
            html_parts.append(f"  • Dense (MARBERTv2): <span style='color: #66ff66; font-weight: bold;'>{dense_display}</span><br>")
            
            rerank_norm = result_item.get("score_rerank_norm")
            raw = result_item.get("score_rerank", rerank_score)
            if rerank_score is not None or rerank_norm is not None:
                # Threshold uses 0–1 (sigmoid of raw). Show 0–1 as primary; raw for reference.
                if rerank_norm is not None:
                    raw_str = f" (raw {raw:.4f})" if raw is not None else ""
                    html_parts.append(f"  • Rerank [0–1]: <span style='color: #ff66ff; font-weight: bold;'>{rerank_norm:.4f}</span><span style='color: #888; font-size: 0.9em;'>{raw_str}</span><br>")
                else:
                    html_parts.append(f"  • Rerank: <span style='color: #ff66ff; font-weight: bold;'>{rerank_score:.4f}</span> <span style='color: #888;'>(raw; threshold uses sigmoid→0–1)</span><br>")
            html_parts.append("</div>")
            
            html_parts.append("<div style='margin-bottom: 10px; color: #e0e0e0;'>")
            html_parts.append(f"<strong style='color: #ffffff;'>Source:</strong> <span style='color: #b0b0b0;'>{kb_family}</span><br>")
            html_parts.append(f"<strong style='color: #ffffff;'>Title:</strong> <span style='color: #b0b0b0; direction: rtl; text-align: right; display: inline-block;'>{title}</span><br>")
            html_parts.append(f"<strong style='color: #ffffff;'>Chunk ID:</strong> <span style='color: #b0b0b0; font-family: monospace;'>{chunk_id}</span><br>")
            if url:
                html_parts.append(f"<strong style='color: #ffffff;'>URL:</strong> <a href='{url}' target='_blank' style='color: #66b3ff;'>{url}</a><br>")
            html_parts.append("</div>")
            
            chunk_length = len(text)
            if chunk_length > 2000:
                preview_text = text[:2000]
                chunk_id_safe = chunk_id.replace(" ", "_").replace(".", "_")
                html_parts.append(f"<div style='background: #1e1e1e; padding: 10px; border-left: 3px solid #0066cc; color: #e0e0e0;'>")
                html_parts.append(f"<strong style='color: #ffffff;'>Content:</strong> <span style='color: #888; font-size: 0.9em;'>({chunk_length} chars)</span><br>")
                html_parts.append(f"<p id='preview_{chunk_id_safe}' style='white-space: pre-wrap; color: #d0d0d0; line-height: 1.6; direction: rtl; text-align: right; font-size: 1.05em;'>{preview_text}...</p>")
                html_parts.append(f"<p id='full_{chunk_id_safe}' style='white-space: pre-wrap; color: #d0d0d0; line-height: 1.6; direction: rtl; text-align: right; font-size: 1.05em; display: none;'>{text}</p>")
                html_parts.append(f"<button onclick=\"document.getElementById('preview_{chunk_id_safe}').style.display='none'; document.getElementById('full_{chunk_id_safe}').style.display='block'; this.style.display='none';\" style='background: #0066cc; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;'>Show Full Text</button>")
                html_parts.append("</div>")
            else:
                html_parts.append(f"<div style='background: #1e1e1e; padding: 10px; border-left: 3px solid #0066cc; color: #e0e0e0;'>")
                html_parts.append(f"<strong style='color: #ffffff;'>Content:</strong> <span style='color: #888; font-size: 0.9em;'>({chunk_length} chars)</span><br>")
                html_parts.append(f"<p style='white-space: pre-wrap; color: #d0d0d0; line-height: 1.6; direction: rtl; text-align: right; font-size: 1.05em;'>{text}</p>")
                html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        html_result = "\n".join(html_parts)
        
        json_result = json.dumps({
            "query": query,
            "num_results": len(chunks),
            "alpha": alpha,
            "top_k": top_k,
            "results": [
                {
                    "rank": i,
                    "chunk_id": result_item.get("chunk", result_item).get("chunk_id"),
                    "title": result_item.get("chunk", result_item).get("title"),
                    "kb_family": result_item.get("chunk", result_item).get("kb_family"),
                    "text": result_item.get("chunk", result_item).get("text", result_item.get("chunk", result_item).get("clean_text", ""))[:200] + "...",
                    "scores": {
                        "hybrid": result_item.get("score_hybrid", result_item.get("hybrid_score", 0.0)),
                        "sparse": result_item.get("score_sparse", result_item.get("sparse_score", 0.0)),
                        "dense": result_item.get("score_raw_dense", result_item.get("score_dense", result_item.get("dense_score", 0.0))),
                        "rerank": result_item.get("score_rerank", result_item.get("rerank_score"))
                    }
                }
                for i, result_item in enumerate(chunks, 1)
            ]
        }, indent=2, ensure_ascii=False)
        
        return html_result, json_result
    
    def compare_retrieval_methods(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """Compare sparse-only, dense-only, and hybrid retrieval."""
        if not query or not query.strip():
            return "Please enter a query."
        
        sparse_pairs = self.retriever.sparse_index.search(query, top_k=top_k * 2)
        dense_pairs = self.retriever.dense_index.search(query, top_k=top_k * 2)
        
        html_parts = []
        html_parts.append(f"<h3 style='color: #ffffff;'>Comparison for Query: <em style='color: #66b3ff; direction: rtl; text-align: right; display: inline-block;'>{query}</em></h3>")
        
        html_parts.append("<h4 style='color: #ff9966;'>🔍 Sparse Retrieval (BM25) - Top Results:</h4>")
        for i, (chunk, score) in enumerate(sparse_pairs[:top_k], 1):
            text = chunk.get("text", chunk.get("clean_text", ""))[:200]
            html_parts.append(f"<p style='color: #e0e0e0;'><strong style='color: #ffffff;'>{i}.</strong> [Score: <span style='color: #ff9966;'>{score:.4f}</span>] <span style='color: #d0d0d0; direction: rtl; text-align: right; display: inline-block;'>{text}...</span></p>")
        
        html_parts.append("<h4 style='color: #66ff66;'>🧠 Dense Retrieval (MARBERTv2) - Top Results:</h4>")
        for i, (chunk, score) in enumerate(dense_pairs[:top_k], 1):
            text = chunk.get("text", chunk.get("clean_text", ""))[:200]
            html_parts.append(f"<p style='color: #e0e0e0;'><strong style='color: #ffffff;'>{i}.</strong> [Score: <span style='color: #66ff66;'>{score:.4f}</span>] <span style='color: #d0d0d0; direction: rtl; text-align: right; display: inline-block;'>{text}...</span></p>")
        
        html_parts.append("<h4 style='color: #66b3ff;'>⚡ Hybrid Retrieval (Combined) - Top Results:</h4>")
        hybrid_results = self.retriever.search(query, top_k=top_k, rerank=False)
        for i, result_item in enumerate(hybrid_results[:top_k], 1):
            chunk = result_item.get("chunk", result_item)
            text = chunk.get("text", chunk.get("clean_text", ""))[:200]
            score = result_item.get("score_hybrid", result_item.get("hybrid_score", 0.0))
            html_parts.append(f"<p style='color: #e0e0e0;'><strong style='color: #ffffff;'>{i}.</strong> [Score: <span style='color: #66b3ff;'>{score:.4f}</span>] <span style='color: #d0d0d0; direction: rtl; text-align: right; display: inline-block;'>{text}...</span></p>")
        
        return "\n".join(html_parts)


def create_gradio_interface():
    """Create and launch the Gradio interface with pre-initialization."""
    print("[INFO] Pre-initializing retriever...")
    print("[INFO] This may take 2-5 minutes on first run...")
    tester = RetrieverTester()
    tester.initialize(alpha=0.4, use_cpu=True)
    print("[OK] Retriever pre-initialized successfully!")
    
    with gr.Blocks(title="KB Retriever Tester", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍اختبار قاعدة البيانات المعرفية للصحة النفسية
        
        الغرض هو اختبار وعرض استرجاع البيانات من قاعدة البيانات المعرفية للصحة النفسية.
        - **Sparse Retrieval**: BM25 keyword matching
        - **Dense Retrieval**: MARBERTv2 semantic embeddings
        - **Hybrid Retrieval**: Combined sparse + dense with configurable alpha
        - **Reranking**: Optional Cross-Encoder re-scoring
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Query (Arabic)",
                    placeholder="مثال: ما هو الاكتئاب؟",
                    lines=2
                )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Top K Results"
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.4,
                        step=0.1,
                        label="Alpha (0=all dense, 1=all sparse)"
                    )
                
                with gr.Row():
                    rerank_checkbox = gr.Checkbox(
                        value=True,
                        label="Use Reranking"
                    )
                    search_btn = gr.Button("🔍 Search", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Retrieval Configuration")
        
        queue_status = gr.Markdown("### 📍 حالة الطلب\n\nفي انتظار طلب...", visible=True)
        
        with gr.Tabs():
            with gr.Tab("📋 Results"):
                results_html = gr.HTML(label="Retrieval Results")
                results_json = gr.JSON(label="Results (JSON)", visible=False)
            
            with gr.Tab("🤖 RAG Generation"):
                gr.Markdown("""
                ### 🤖 RAG (Retrieval-Augmented Generation)
                
                This tab uses the complete RAG pipeline:
                1. **Retrieval**: Hybrid search (BM25 + MARBERTv2)
                2. **Reranking**: Cross-Encoder scoring
                3. **Decision**: If reranker score > 0.1, use KB context; otherwise, use general knowledge
                4. **Generation**: Generate response using fine-tuned Allam7B model
                
                The generator will:
                - Summarize and clean retrieved chunks
                - Remove personal information and names
                - Provide concise 300-500 character answers
                - Include safety guardrails for critical topics
                """)
                
                rag_query_input = gr.Textbox(
                    label="Query (Arabic)",
                    placeholder="مثال: ما هو الاكتئاب؟",
                    lines=2
                )
                
                with gr.Row():
                    rag_top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top K Results"
                    )
                    rag_alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="Alpha (0=all dense, 1=all sparse)"
                    )
                    rerank_threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        label="Rerank threshold (0–1, vs sigmoid of reranker; use KB when max > this)"
                    )
                
                with gr.Row():
                    rag_rerank_checkbox = gr.Checkbox(
                        value=True,
                        label="Use Reranking"
                    )
                    rag_generate_btn = gr.Button("🤖 Generate Response", variant="primary")
                
                rag_response_output = gr.Textbox(
                    label="Generated Response",
                    lines=10,
                    interactive=False
                )
                
                rag_info_output = gr.Markdown("### ℹ️ Generation Info\n\nWaiting for query...")
            
            with gr.Tab("🔬 Comparison"):
                comparison_html = gr.HTML(label="Method Comparison")
                compare_btn = gr.Button("Compare Methods", variant="secondary")
            
            with gr.Tab("📊 Statistics"):
                stats_md = gr.Markdown("### KB Statistics\n\nClick 'Get Stats' to load statistics.")
                stats_btn = gr.Button("Get Stats")
            
            # Add Manual Evaluation tab if available
            if MANUAL_EVAL_AVAILABLE:
                try:
                    # Try to find retrieval_test_results.json in parent directory
                    json_path = Path(__file__).parent.parent / "retrieval_test_results.json"
                    if not json_path.exists():
                        # Try current directory
                        json_path = Path("retrieval_test_results.json")
                    
                    evaluator = ManualEvaluator(json_path=str(json_path))
                    
                    with gr.Tab("📝 Manual Evaluation"):
                        gr.Markdown("""
                        # 📝 Manual Evaluation of Retrieval Results
                        
                        Evaluate the relevance of retrieved answers to questions.
                        - **5**: Retrieved answer is directly relevant and provides a solution
                        - **4**: Retrieved answer is mostly relevant with minor gaps
                        - **3**: Retrieved answer is somewhat relevant but incomplete
                        - **2**: Retrieved answer has limited relevance
                        - **1**: Retrieved chunks are not relevant
                        
                        Your evaluations will be saved automatically when you click Save, Next, or Previous.
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                item_display = gr.HTML(label="Question & Answers")
                                status_text = gr.Textbox(
                                    label="Status",
                                    value=f"📄 Item 1 of {evaluator.get_total_count()}",
                                    interactive=False
                                )
                                rating_slider = gr.Slider(
                                    minimum=1,
                                    maximum=5,
                                    step=1,
                                    value=None,
                                    label="Relevance Rating (1-5)",
                                    info="5 = Directly relevant with solution, 1 = Not relevant"
                                )
                                comment_box = gr.Textbox(
                                    label="Comments (Optional)",
                                    placeholder="Add any comments about this evaluation...",
                                    lines=3
                                )
                                with gr.Row():
                                    first_btn = gr.Button("⏮️ First", variant="secondary")
                                    prev_btn = gr.Button("◀️ Previous", variant="secondary")
                                    next_btn = gr.Button("Next ▶️", variant="secondary")
                                    last_btn = gr.Button("Last ⏭️", variant="secondary")
                                with gr.Row():
                                    save_btn = gr.Button("💾 Save Evaluation", variant="primary")
                                with gr.Row():
                                    jump_input = gr.Number(
                                        label="Jump to Index (0-based)",
                                        value=0,
                                        minimum=0,
                                        maximum=max(0, evaluator.get_total_count() - 1),
                                        step=1
                                    )
                                    jump_btn = gr.Button("🔢 Jump", variant="secondary")
                            
                            with gr.Column(scale=1):
                                stats_md_eval = gr.Markdown("### 📊 Statistics\n\nLoading...")
                                fast_access_html = gr.HTML(label="Fast Access")
                        
                        current_index_state = gr.State(value=0)
                        total_count_state = gr.State(value=evaluator.get_total_count())
                        
                        def load_item(index: int = None):
                            if index is not None:
                                evaluator.current_index = index
                            item, status, idx, total, evaluated = evaluator.navigate("")
                            display_html = evaluator.format_item_display(item)
                            current_rating = item.get('manual_rating') if item else None
                            current_comment = item.get('manual_comment', '') if item else ''
                            evaluated_count = evaluator.get_evaluated_count()
                            percentage = (evaluated_count/total*100) if total > 0 else 0
                            stats_text = f"""
                            ### 📊 Statistics
                            
                            - **Total Items**: {total}
                            - **Evaluated**: {evaluated_count} ({percentage:.1f}%)
                            - **Remaining**: {total - evaluated_count}
                            - **Current Position**: {idx + 1} / {total}
                            """
                            fast_access = evaluator.create_fast_access_buttons(evaluated, total)
                            return (display_html, status, current_rating, current_comment, stats_text, fast_access, idx, total)
                        
                        def save_and_next(rating, comment):
                            if rating is not None:
                                evaluator.update_evaluation(int(rating), comment)
                                evaluator.save_data()
                            item, status, idx, total, evaluated = evaluator.navigate("next")
                            display_html = evaluator.format_item_display(item)
                            current_rating = item.get('manual_rating') if item else None
                            current_comment = item.get('manual_comment', '') if item else ''
                            evaluated_count = evaluator.get_evaluated_count()
                            percentage = (evaluated_count/total*100) if total > 0 else 0
                            stats_text = f"""
                            ### 📊 Statistics
                            
                            - **Total Items**: {total}
                            - **Evaluated**: {evaluated_count} ({percentage:.1f}%)
                            - **Remaining**: {total - evaluated_count}
                            - **Current Position**: {idx + 1} / {total}
                            """
                            fast_access = evaluator.create_fast_access_buttons(evaluated, total)
                            return (display_html, status, current_rating, current_comment, stats_text, fast_access, idx, total, "✅ Saved and moved to next!")
                        
                        def save_evaluation(rating, comment):
                            if rating is not None:
                                evaluator.update_evaluation(int(rating), comment)
                                return evaluator.save_data()
                            return "⚠️ Please select a rating first"
                        
                        def navigate_with_save(direction, rating, comment):
                            if rating is not None:
                                evaluator.update_evaluation(int(rating), comment)
                                evaluator.save_data()
                            item, status, idx, total, evaluated = evaluator.navigate(direction)
                            display_html = evaluator.format_item_display(item)
                            current_rating = item.get('manual_rating') if item else None
                            current_comment = item.get('manual_comment', '') if item else ''
                            evaluated_count = evaluator.get_evaluated_count()
                            percentage = (evaluated_count/total*100) if total > 0 else 0
                            stats_text = f"""
                            ### 📊 Statistics
                            
                            - **Total Items**: {total}
                            - **Evaluated**: {evaluated_count} ({percentage:.1f}%)
                            - **Remaining**: {total - evaluated_count}
                            - **Current Position**: {idx + 1} / {total}
                            """
                            fast_access = evaluator.create_fast_access_buttons(evaluated, total)
                            return (display_html, status, current_rating, current_comment, stats_text, fast_access, idx, total)
                        
                        def jump_to(jump_idx):
                            item, status, idx, total, evaluated = evaluator.jump_to_index(int(jump_idx))
                            display_html = evaluator.format_item_display(item)
                            current_rating = item.get('manual_rating') if item else None
                            current_comment = item.get('manual_comment', '') if item else ''
                            evaluated_count = evaluator.get_evaluated_count()
                            percentage = (evaluated_count/total*100) if total > 0 else 0
                            stats_text = f"""
                            ### 📊 Statistics
                            
                            - **Total Items**: {total}
                            - **Evaluated**: {evaluated_count} ({percentage:.1f}%)
                            - **Remaining**: {total - evaluated_count}
                            - **Current Position**: {idx + 1} / {total}
                            """
                            fast_access = evaluator.create_fast_access_buttons(evaluated, total)
                            return (display_html, status, current_rating, current_comment, stats_text, fast_access, idx, total)
                        
                        # Initial load
                        item_display.load(
                            fn=lambda: load_item(0),
                            inputs=[],
                            outputs=[item_display, status_text, rating_slider, comment_box, stats_md_eval, fast_access_html, current_index_state, total_count_state]
                        )
                        
                        save_btn.click(fn=save_evaluation, inputs=[rating_slider, comment_box], outputs=[status_text])
                        next_btn.click(fn=lambda r, c: save_and_next(r, c), inputs=[rating_slider, comment_box], outputs=[item_display, status_text, rating_slider, comment_box, stats_md_eval, fast_access_html, current_index_state, total_count_state])
                        prev_btn.click(fn=lambda r, c: navigate_with_save("previous", r, c), inputs=[rating_slider, comment_box], outputs=[item_display, status_text, rating_slider, comment_box, stats_md_eval, fast_access_html, current_index_state, total_count_state])
                        first_btn.click(fn=lambda r, c: navigate_with_save("first", r, c), inputs=[rating_slider, comment_box], outputs=[item_display, status_text, rating_slider, comment_box, stats_md_eval, fast_access_html, current_index_state, total_count_state])
                        last_btn.click(fn=lambda r, c: navigate_with_save("last", r, c), inputs=[rating_slider, comment_box], outputs=[item_display, status_text, rating_slider, comment_box, stats_md_eval, fast_access_html, current_index_state, total_count_state])
                        jump_btn.click(fn=jump_to, inputs=[jump_input], outputs=[item_display, status_text, rating_slider, comment_box, stats_md_eval, fast_access_html, current_index_state, total_count_state])
                except Exception as e:
                    import traceback
                    with gr.Tab("📝 Manual Evaluation"):
                        gr.Markdown(f"### ⚠️ Error loading manual evaluation\n\nError: {str(e)}\n\n```\n{traceback.format_exc()}\n```")
        
        def search_handler_with_queue(query, top_k, alpha, rerank):
            """Handler that uses the queue system with status updates."""
            try:
                if not query or not query.strip():
                    # For non-generator case, return is fine
                    return "Please enter a query.", "{}", "### 📍 حالة الطلب\n\nيرجى إدخال استعلام."
                
                request_id = str(uuid.uuid4())
                print(f"[DEBUG] New search request: {request_id}, query: {query[:50]}...")
                
                request_queue.add_request(
                    request_id=request_id,
                    query=query,
                    top_k=int(top_k),
                    alpha=float(alpha),
                    rerank=rerank,
                    tester=tester
                )
                
                print(f"[DEBUG] Request added to queue: {request_id}")
                
                start_time = time.time()
                max_wait = 300
                check_count = 0
                
                while time.time() - start_time < max_wait:
                    check_count += 1
                    status = request_queue.get_status(request_id)
                    status_html = f"### 📍 حالة الطلب\n\n{status}"
                    
                    if request_id in request_queue.results:
                        html, json_data, _ = request_queue.results.pop(request_id)
                        print(f"[DEBUG] Request completed: {request_id}, yielding result")
                        # Yield the final result (Gradio requires yield, not return, for generators)
                        yield html, json_data, "### 📍 حالة الطلب\n\n✅ تمت المعالجة بنجاح!"
                        return  # Exit the generator
                    
                    if check_count % 10 == 0:  # Log every 5 seconds
                        print(f"[DEBUG] Waiting for request {request_id}, status: {status}")
                    
                    yield "", "{}", status_html
                    time.sleep(0.5)
                
                print(f"[DEBUG] Request timeout: {request_id}")
                yield "⏱️ انتهت مهلة الانتظار. يرجى المحاولة مرة أخرى.", "{}", "### 📍 حالة الطلب\n\n⏱️ انتهت مهلة الانتظار"
            except Exception as e:
                error_msg = f"Error in search handler: {str(e)}"
                print(f"[ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
                return f"❌ خطأ: {error_msg}", "{}", "### 📍 حالة الطلب\n\n❌ حدث خطأ"
        
        def compare_handler(query, top_k):
            return tester.compare_retrieval_methods(query, top_k=int(top_k))
        
        def stats_handler():
            try:
                sparse_stats = tester.retriever.sparse_index.get_stats()
                
                vector_db_stats = {}
                try:
                    if hasattr(tester.retriever.dense_index, 'vector_db') and tester.retriever.dense_index.vector_db:
                        vector_db_stats = tester.retriever.dense_index.vector_db.get_stats()
                except:
                    pass
                
                stats_text = f"""
                ### Knowledge Base Statistics
                
                - **Total Chunks**: {sparse_stats.get('total_chunks', 'N/A')}
                - **KB Families**: {sparse_stats.get('kb_families', 'N/A')}
                - **Documents**: {sparse_stats.get('total_docs', 'N/A')}
                
                ### Index Information
                - **BM25 Index**: Ready ({sparse_stats.get('index_type', 'BM25')})
                - **Vector DB**: {'Ready' if vector_db_stats else 'Not loaded'} (ChromaDB)
                - **Embedding Model**: MARBERTv2
                - **Retriever Alpha**: {tester.retriever.alpha} ({(1-tester.retriever.alpha)*100:.0f}% dense, {tester.retriever.alpha*100:.0f}% sparse)
                """
                return stats_text
            except Exception as e:
                return f"Error loading stats: {str(e)}"
        
        # Use queue handler with generator support
        search_btn.click(
            fn=search_handler_with_queue,
            inputs=[query_input, top_k_slider, alpha_slider, rerank_checkbox],
            outputs=[results_html, results_json, queue_status],
            show_progress="full"  # Show progress bar for generator functions
        )
        
        compare_btn.click(
            fn=compare_handler,
            inputs=[query_input, top_k_slider],
            outputs=[comparison_html]
        )
        
        stats_btn.click(
            fn=stats_handler,
            outputs=[stats_md]
        )
        
        # RAG Generation handler
        def rag_generate_handler(query, top_k, alpha, rerank_threshold, rerank):
            """Handle RAG generation request."""
            try:
                if not query or not query.strip():
                    return "يرجى إدخال استعلام.", "### ℹ️ Generation Info\n\nيرجى إدخال استعلام."
                
                # Initialize RAG pipeline if needed
                tester.initialize_rag(alpha=alpha, rerank_threshold=rerank_threshold)
                
                # Process through RAG pipeline
                result = tester.rag_pipeline.process(
                    query=query.strip(),
                    top_k=int(top_k),
                    alpha=float(alpha),
                    rerank=rerank,
                    max_new_tokens=150,
                    temperature=0.7,
                )
                
                # Format info
                info_text = f"""
### ℹ️ Generation Info

- **Used KB Context**: {'✅ Yes' if result['used_kb_context'] else '❌ No (General Knowledge)'}
- **Number of Chunks**: {result['num_chunks']}
- **Rerank [0–1]** (for threshold): {', '.join([f'{s:.4f}' for s in (result.get('rerank_scores_norm') or result.get('rerank_scores') or [])[:3]]) or 'N/A'}
- **Retrieval Info**: {result['retrieval_info']}
"""
                
                return result['response'], info_text
            except Exception as e:
                error_msg = f"Error during RAG generation: {str(e)}"
                print(f"[ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
                return f"❌ خطأ: {error_msg}", f"### ℹ️ Generation Info\n\n❌ حدث خطأ: {error_msg}"
        
        rag_generate_btn.click(
            fn=rag_generate_handler,
            inputs=[rag_query_input, rag_top_k_slider, rag_alpha_slider, rerank_threshold_slider, rag_rerank_checkbox],
            outputs=[rag_response_output, rag_info_output],
            show_progress="full"
        )
        
        # RAG Generation handler
        def rag_generate_handler(query, top_k, alpha, rerank_threshold, rerank):
            """Handle RAG generation request."""
            try:
                if not query or not query.strip():
                    return "يرجى إدخال استعلام.", "### ℹ️ Generation Info\n\nيرجى إدخال استعلام."
                
                # Initialize RAG pipeline if needed
                tester.initialize_rag(alpha=alpha, rerank_threshold=rerank_threshold)
                
                # Process through RAG pipeline
                result = tester.rag_pipeline.process(
                    query=query.strip(),
                    top_k=int(top_k),
                    alpha=float(alpha),
                    rerank=rerank,
                    max_new_tokens=150,
                    temperature=0.7,
                )
                
                # Format info (rerank_scores_norm = sigmoid(raw); threshold is compared to these 0–1 values)
                norms = result.get('rerank_scores_norm') or result.get('rerank_scores') or []
                info_text = f"""
### ℹ️ Generation Info

- **Used KB Context**: {'✅ Yes' if result['used_kb_context'] else '❌ No (General Knowledge)'}
- **Number of Chunks**: {result['num_chunks']}
- **Rerank [0–1]** (threshold uses max of these): {', '.join([f'{s:.4f}' for s in norms[:3]]) or 'N/A'}
- **Retrieval Info**: {result['retrieval_info']}
"""
                
                return result['response'], info_text
            except Exception as e:
                error_msg = f"Error during RAG generation: {str(e)}"
                print(f"[ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
                return f"❌ خطأ: {error_msg}", f"### ℹ️ Generation Info\n\n❌ حدث خطأ: {error_msg}"
        
        rag_generate_btn.click(
            fn=rag_generate_handler,
            inputs=[rag_query_input, rag_top_k_slider, rag_alpha_slider, rerank_threshold_slider, rag_rerank_checkbox],
            outputs=[rag_response_output, rag_info_output],
            show_progress="full"
        )
        
        gr.Markdown("### 💡 Example Queries")
        examples = gr.Examples(
            examples=[
                ["ما هو الاكتئاب؟"],
                ["كيف أتعامل مع القلق؟"],
                ["أعراض اضطراب الهلع"],
                ["علاج الرهاب الاجتماعي"],
                ["مشاكل النوم والأرق"]
            ],
            inputs=query_input
        )
    
    return demo


def main():
    """Main entry point to launch the Gradio interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio interface for KB retriever testing")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    args = parser.parse_args()
    
    demo = create_gradio_interface()
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port
    )


if __name__ == "__main__":
    main()
