"""
Generator module for RAG system using Allam7B-Physiology-RAG-finetuned-final model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Auto-detect best available device (GPU > CPU)."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"[DEVICE] GPU detected: {device_name} ({memory_gb:.1f} GB)")
        return "cuda"
    else:
        logger.info("[DEVICE] No GPU detected, using CPU")
        return "cpu"


class RAGGenerator:
    """Generator for RAG system using fine-tuned Allam7B model."""
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str = "humain-ai/ALLaM-7B-Instruct-preview",
        device: str = "cpu",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the RAG generator.
        
        Args:
            model_path: Path to the fine-tuned adapter model directory
            base_model_name: Name of the base model on HuggingFace
            device: Device to run on ('cpu', 'cuda', etc.)
            load_in_4bit: Whether to load in 4-bit quantization
            load_in_8bit: Whether to load in 8-bit quantization
            torch_dtype: Optional torch dtype (e.g., torch.float16)
        """
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype or torch.float32
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
        logger.info(f"[GENERATOR] Initializing with model_path={model_path}, device={device}")
    
    def initialize(self):
        """Load the model and tokenizer."""
        if self._initialized:
            logger.info("[GENERATOR] Already initialized, skipping...")
            return
        
        try:
            # Load tokenizer with fallback to base model and slow tokenizer
            logger.info(f"[GENERATOR] Loading tokenizer from {self.model_path}...")
            try:
                # Try fast tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True,
                    use_fast=True
                )
                logger.info("[GENERATOR] ✅ Tokenizer loaded from model path (fast)")
            except Exception as e:
                logger.warning(f"[GENERATOR] ⚠️ Fast tokenizer failed: {e}")
                try:
                    # Try slow tokenizer from model path
                    logger.info("[GENERATOR] Trying slow tokenizer from model path...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(self.model_path),
                        trust_remote_code=True,
                        use_fast=False
                    )
                    logger.info("[GENERATOR] ✅ Tokenizer loaded from model path (slow)")
                except Exception as e2:
                    logger.warning(f"[GENERATOR] ⚠️ Model path tokenizer failed: {e2}")
                    logger.info(f"[GENERATOR] Falling back to base model tokenizer: {self.base_model_name}")
                    try:
                        # Try fast tokenizer from base model
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.base_model_name,
                            trust_remote_code=True,
                            use_fast=True
                        )
                        logger.info("[GENERATOR] ✅ Tokenizer loaded from base model (fast)")
                    except Exception as e3:
                        logger.warning(f"[GENERATOR] ⚠️ Base model fast tokenizer failed: {e3}")
                        # Final fallback: slow tokenizer from base model
                        logger.info("[GENERATOR] Using slow tokenizer from base model...")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.base_model_name,
                            trust_remote_code=True,
                            use_fast=False
                        )
                        logger.info("[GENERATOR] ✅ Tokenizer loaded from base model (slow)")
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"[GENERATOR] Loading base model: {self.base_model_name}...")
            
            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            # Configure quantization based on device
            if self.device == "cuda":
                # GPU: bitsandbytes 4-bit is FAST and memory-efficient
                if self.load_in_4bit:
                    try:
                        from transformers import BitsAndBytesConfig
                        # Test if bitsandbytes is properly installed
                        import bitsandbytes as bnb
                        model_kwargs["low_cpu_mem_usage"] = True
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        logger.info("[GENERATOR] ✅ Using 4-bit quantization on GPU (fast & efficient)")
                    except (ImportError, AttributeError, ModuleNotFoundError, RuntimeError) as e:
                        logger.warning(f"[GENERATOR] ⚠️ bitsandbytes not available: {e}")
                        logger.warning("[GENERATOR] Falling back to float16 (still fast on GPU)")
                        model_kwargs["torch_dtype"] = torch.float16
                        model_kwargs["low_cpu_mem_usage"] = True
                        self.load_in_4bit = False  # Disable 4-bit for this session
                elif self.load_in_8bit:
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["low_cpu_mem_usage"] = True
                    logger.info("[GENERATOR] Using 8-bit quantization on GPU")
                else:
                    # Full precision on GPU (faster but uses more memory)
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["low_cpu_mem_usage"] = True
                    logger.info("[GENERATOR] Using float16 on GPU (full precision)")
            else:
                # CPU: bitsandbytes is EXTREMELY slow (0.1 tokens/sec)
                # Must use regular dtypes instead
                if self.load_in_4bit or self.load_in_8bit:
                    logger.warning("[GENERATOR] ⚠️ bitsandbytes disabled on CPU (extremely slow)")
                    logger.warning("[GENERATOR] Using float16 instead for CPU performance")
                # Use float16 for CPU - much faster than bitsandbytes 4-bit
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["low_cpu_mem_usage"] = True
                logger.info("[GENERATOR] Using float16 on CPU")
            
            # Set device_map only for CUDA
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            else:
                # For CPU, don't use device_map (loads directly to CPU)
                model_kwargs["device_map"] = None
            
            # Load base model
            logger.info("[GENERATOR] Loading base model (this may take a few minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" or model_kwargs.get("device_map") is None:
                self.model = self.model.to(self.device)
            
            logger.info(f"[GENERATOR] Loading LoRA adapter from {self.model_path}...")
            # Prepare adapter loading kwargs
            adapter_kwargs = {}
            if self.device == "cuda":
                adapter_kwargs["device_map"] = "auto"
            # For CPU, don't pass device_map - let it use the model's current device
            
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.model_path),
                **adapter_kwargs
            )
            
            # Ensure model is on correct device after loading adapter
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Merge adapter if needed (for faster inference)
            # self.model = self.model.merge_and_unload()
            
            self.model.eval()
            self._initialized = True
            
            logger.info("[GENERATOR] ✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"[GENERATOR] ❌ Error loading model: {e}")
            raise
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        chunks: Optional[List[Dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        return_full_text: bool = False,
    ) -> str:
        """
        Generate response to query, optionally with context from KB.
        
        Args:
            query: User query
            context: Optional pre-formatted context string (if provided, chunks are ignored)
            chunks: Optional list of retrieved chunks (will be summarized if provided)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            return_full_text: Whether to return full prompt + response
        
        Returns:
            Generated response text
        """
        if not self._initialized:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
        
        # Summarize chunks if provided (preferred over raw context)
        if chunks and not context:
            logger.info(f"[GENERATOR] Summarizing {len(chunks)} retrieved chunks...")
            context = self.summarize_context(chunks)
            logger.info(f"[GENERATOR] Created clean summary ({len(context)} chars)")
        
        # Build prompt
        if context:
            # RAG prompt with context
            prompt = self._build_rag_prompt(query, context)
            logger.info(f"[GENERATOR] Using RAG mode with context ({len(context)} chars)")
        else:
            # General knowledge mode
            prompt = self._build_general_prompt(query)
            logger.info("[GENERATOR] Using general knowledge mode (no context)")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Reasonable limit
        ).to(self.device)
        
        # Optimize generation settings based on device
        import time
        start_time = time.time()
        input_length = inputs['input_ids'].shape[1]
        
        # Limit max_new_tokens for shorter responses (300-500 chars ≈ 75-125 tokens)
        # Use conservative limits to ensure concise answers
        target_max_tokens = min(max_new_tokens, 150)  # Cap at 150 tokens for concise answers
        
        if self.device == "cuda":
            # GPU-optimized settings: faster, can handle more tokens
            expected_time = "10-30 seconds"
            max_tokens = target_max_tokens
            top_k_value = top_k  # No reduction on GPU
            logger.info(f"[GENERATOR] 🚀 GPU mode: Starting generation (max_new_tokens={max_tokens}, input_tokens={input_length})...")
            logger.info(f"[GENERATOR] Expected time: {expected_time}")
        else:
            # CPU-optimized settings: conservative to prevent hanging
            expected_time = "2-5 minutes"
            max_tokens = min(target_max_tokens, 100)  # Cap at 100 for CPU
            top_k_value = min(top_k, 40)  # Reduce top_k for faster CPU generation
            logger.info(f"[GENERATOR] 💻 CPU mode: Starting generation (max_new_tokens={max_tokens}, input_tokens={input_length})...")
            logger.info(f"[GENERATOR] Expected time: {expected_time}")
        
        # Log device info
        # Get actual model device (may be different if offloaded by accelerate)
        model_device = next(self.model.parameters()).device
        logger.info(f"[GENERATOR] Model device: {model_device}")
        logger.info(f"[GENERATOR] Input device: {inputs['input_ids'].device}")
        
        # Move inputs to match model device (important for accelerate offloading)
        if inputs['input_ids'].device != model_device:
            logger.info(f"[GENERATOR] Moving inputs from {inputs['input_ids'].device} to {model_device}")
            inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        try:
            with torch.inference_mode():  # Memory efficient
                logger.info(f"[GENERATOR] Calling model.generate()...")
                
                # Build generation kwargs
                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k_value,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": 1.1,
                    "use_cache": True,  # KV cache for efficiency
                }
                
                # GPU-specific optimizations
                if self.device == "cuda":
                    generation_kwargs["num_beams"] = 1  # Greedy decoding (fast on GPU)
                    # Note: Flash Attention is configured at model load time, not during generation
                    # The model will use it automatically if available
                else:
                    generation_kwargs["num_beams"] = 1  # Greedy decoding (faster on CPU)
                
                outputs = self.model.generate(**generation_kwargs)
                
                logger.info(f"[GENERATOR] model.generate() completed, output shape: {outputs.shape}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[GENERATOR] ❌ Generation failed after {elapsed:.1f}s: {e}")
            import traceback
            logger.error(f"[GENERATOR] Traceback: {traceback.format_exc()}")
            raise
        
        generation_time = time.time() - start_time
        logger.info(f"[GENERATOR] ✅ Generation completed in {generation_time:.1f}s ({generation_time/60:.1f} minutes)")
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        if return_full_text:
            cleaned_text = generated_text
        else:
            # Extract only the generated part (after the prompt)
            prompt_length = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            response = generated_text[prompt_length:].strip()
            cleaned_text = response
        
        # Post-process: remove names, dates, artifact tokens, credentials
        cleaned_text = self._clean_response(cleaned_text)
        
        # Enforce length (target 200-400 chars; hard cap 400)
        if len(cleaned_text) > 400:
            sentences = cleaned_text.split('.')
            truncated = ""
            for s in sentences:
                if len(truncated + s + '.') <= 400:
                    truncated += s + '.'
                else:
                    break
            cleaned_text = truncated.strip() if truncated else cleaned_text[:400].strip()
        
        # Ensure recommendation if missing and there is room
        if 'مختص' not in cleaned_text and 'طبيب' not in cleaned_text and 'استشارة' not in cleaned_text:
            if len(cleaned_text) < 350:
                cleaned_text += " يُنصح بمراجعة طبيب نفسي مختص."
        
        return cleaned_text
    
    def summarize_context(self, chunks: List[Dict]) -> str:
        """
        Summarize and clean retrieved chunks into a single coherent context.
        Removes names, greetings, closing phrases, and merges into clean summary.
        """
        import re
        
        # Extract text from chunks
        texts = []
        for chunk in chunks:
            text = chunk.get("text", chunk.get("clean_text", ""))
            if text:
                texts.append(text)
        
        if not texts:
            return ""
        
        # Clean each text - using same patterns as test script
        cleaned_texts = []
        for text in texts:
            # Remove common greetings (same as test script)
            text = re.sub(r'بسم الله[^.]*\.', '', text)
            text = re.sub(r'الأخت الفاضلة/[^،\n]+', '', text)
            text = re.sub(r'الأخ الفاضل/[^،\n]+', '', text)
            text = re.sub(r'حفظها الله|حفظه الله', '', text)
            text = re.sub(r'السلام عليكم[^.]*\.', '', text)
            
            # Remove closing phrases (same as test script)
            text = re.sub(r'انتهت إجابة[^.]*\.', '', text)
            text = re.sub(r'تليها إجابة[^.]*\.', '', text)
            text = re.sub(r'وتضيف[^.]*:', '', text)
            text = re.sub(r'بارك الله فيك[^.]*\.', '', text)
            text = re.sub(r'جزاك الله خيرًا[^.]*\.', '', text)
            text = re.sub(r'وبالله التوفيق[^.]*\.', '', text)
            text = re.sub(r'وفقك الله[^.]*\.', '', text)
            text = re.sub(r'وفقكم الله[^.]*\.', '', text)
            text = re.sub(r'والله الموفق[^.]*\.', '', text)
            text = re.sub(r'شكراً لسؤالك[^.]*\.', '', text)
            text = re.sub(r'شكرا لسؤالك[^.]*\.', '', text)
            text = re.sub(r'شكراً لك[^.]*\.', '', text)
            text = re.sub(r'شكرا لك[^.]*\.', '', text)
            
            # Remove specific names (same as test script)
            text = re.sub(r'الدكتور[^،\n]+(?:استشاري|مستشار)[^،\n]*', '', text)
            text = re.sub(r'المستشار[^،\n]+', '', text)
            text = re.sub(r'المستشارة[^،\n]+', '', text)
            
            # Remove doctor/person names (single or multiple words)
            text = re.sub(r'د\.\s+[أ-ي]+(?:\s+[أ-ي]+)*', '', text)
            text = re.sub(r'دكتورة\s+[أ-ي]+(?:\s+[أ-ي]+)*', '', text)
            text = re.sub(r'الدكتور\s+[أ-ي]+(?:\s+[أ-ي]+)*', '', text)
            
            # Remove tags and artifact tokens
            text = re.sub(r'\[/INST\]', '', text)
            text = re.sub(r'INSTAINSTANT_ANSWER', '', text)
            text = re.sub(r'INSTANT_ANSWER', '', text)
            text = re.sub(r'INST[A-Za-z0-9_]*', '', text)
            text = re.sub(r'[A-Z]{2,}_[A-Za-z0-9_]+', '', text)
            
            # Remove dates and metadata stamps
            text = re.sub(r'تم التحديث بتاريخ\s*[\d٠-٩/]+', '', text)
            text = re.sub(r'طبيبة\s+عامة\s*\.?', '', text)
            text = re.sub(r'طبيب\s+عام\s*\.?', '', text)
            
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Remove very short fragments (likely artifacts)
            if len(text.strip()) > 50:  # Keep only substantial text
                cleaned_texts.append(text.strip())
        
        # Merge into single coherent summary
        # Remove duplicates and filter for medical content only
        medical_sentences = []
        seen = set()
        
        for text in cleaned_texts:
            # Split into sentences
            sentences = re.split(r'[.!?]\s+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                # Simple deduplication (normalize)
                normalized = re.sub(r'[^\w\s]', '', sentence.lower())
                if normalized in seen or len(normalized) < 10:
                    continue
                
                # Only keep sentences with medical/psychological content
                medical_keywords = [
                    'اكتئاب', 'قلق', 'نفسي', 'عاطفي', 'نوم', 'أرق', 'أعراض', 
                    'علاج', 'طبيب', 'مختص', 'صحة', 'صحي', 'مرض', 'حالة',
                    'depression', 'anxiety', 'mental', 'health', 'symptom', 'treatment'
                ]
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in medical_keywords):
                    medical_sentences.append(sentence)
                    seen.add(normalized)
        
        if not medical_sentences:
            return ""
        
        # Remove contradictory information (physical vs mental health)
        # If we have mental health recommendations, prioritize them and remove conflicting physical mentions
        has_mental_health = any('نفسي' in s or 'عاطفي' in s or 'اكتئاب' in s or 'قلق' in s 
                                for s in medical_sentences)
        
        if has_mental_health:
            # Filter out sentences about physical conditions that contradict mental health focus
            filtered_sentences = []
            for sentence in medical_sentences:
                # Skip sentences that mention physical conditions when we're focusing on mental health
                physical_keywords = ['عضوي', 'جسدي', 'فحص مخبري', 'تحليل', 'طبيب عام']
                if any(keyword in sentence.lower() for keyword in physical_keywords):
                    # But keep general health advice (like sleep hygiene)
                    if 'نوم' in sentence.lower() or 'صحة' in sentence.lower():
                        filtered_sentences.append(sentence)
                else:
                    filtered_sentences.append(sentence)
            medical_sentences = filtered_sentences if filtered_sentences else medical_sentences
        
        # Create a coherent paragraph by joining sentences
        # Ensure it flows naturally as a single paragraph
        summary = ' '.join(medical_sentences)
        
        # Clean up spacing to ensure it's a single flowing paragraph
        summary = re.sub(r'\s+', ' ', summary)
        summary = re.sub(r'\n+', ' ', summary)
        
        # Limit length to prevent excessively long prompts
        if len(summary) > 1200:
            # Truncate at sentence boundary
            sentences = summary.split('.')
            summary = ""
            for sentence in sentences:
                if len(summary + sentence + '.') <= 1200:
                    summary += sentence + '.'
                else:
                    break
            if not summary:
                summary = summary[:1200] + "..."
        
        return summary.strip()
    
    def _clean_response(self, text: str) -> str:
        """Remove common patterns copied from knowledge base: names, dates, artifact tokens, credentials."""
        import re
        
        # --- Artifact tokens (placeholders, internal tags) ---
        text = re.sub(r'INST[A-Za-z0-9_]*', '', text)
        text = re.sub(r'[A-Z]{2,}[_][A-Za-z0-9_]+', '', text)  # ALL_CAPS_WITH_UNDERSCORES
        
        # --- Dates and update stamps (leaked from KB metadata) ---
        text = re.sub(r'تم التحديث بتاريخ\s*[\d٠-٩/]+', '', text)
        text = re.sub(r'تاريخ\s*[\d٠-٩/]+\s*\.?', '', text)
        
        # --- Credentials / role labels (طبيبة عامة، استشاري، إلخ) ---
        text = re.sub(r'طبيبة\s+عامة\s*\.?', '', text)
        text = re.sub(r'طبيب\s+عام\s*\.?', '', text)
        text = re.sub(r'استشاري[ة]?\s*(?:نفسي[ة]?|طب نفس[ية]?)?\s*\.?', '', text)
        
        # --- Person/doctor names: د. إيناس، دكتورة X، الدكتور X Y ---
        text = re.sub(r'د\.\s+[أ-ي]+\s*(?:[أ-ي]+\s*)*', '', text)
        text = re.sub(r'دكتورة\s+[أ-ي]+\s*(?:[أ-ي]+\s*)*', '', text)
        text = re.sub(r'الدكتور\s+[أ-ي]+\s*(?:[أ-ي]+\s*)*', '', text)
        text = re.sub(r'الدكتور[^،.\n]+(?:استشاري|مستشار)[^،.\n]*', '', text)
        text = re.sub(r'المستشار[ة]?[^،.\n]+', '', text)
        
        # --- Common greetings with names ---
        text = re.sub(r'بسم الله الرحمن الرحيم\s*\n*', '', text)
        text = re.sub(r'السلام عليكم[^.]*\.', '', text)
        text = re.sub(r'الأخت الفاضلة/[^،\n]+', '', text)
        text = re.sub(r'الأخ الفاضل/[^،\n]+', '', text)
        text = re.sub(r'الابنة الفاضلة/[^،\n]+', '', text)
        text = re.sub(r'حفظها الله|حفظه الله', '', text)
        
        # --- Closing phrases and thank you messages ---
        text = re.sub(r'انتهت إجابة[^.]*\.', '', text)
        text = re.sub(r'تليها إجابة[^.]*\.', '', text)
        text = re.sub(r'وتضيف[^.]*:', '', text)
        text = re.sub(r'بارك الله فيك[^.]*\.', '', text)
        text = re.sub(r'جزاك الله خيرًا[^.]*\.', '', text)
        text = re.sub(r'وبالله التوفيق[^.]*\.', '', text)
        text = re.sub(r'وفقك الله[^.]*\.', '', text)
        text = re.sub(r'وفقكم الله[^.]*\.', '', text)
        text = re.sub(r'والله الموفق[^.]*\.', '', text)
        text = re.sub(r'شكراً لسؤالك[^.]*\.', '', text)
        text = re.sub(r'شكرا لسؤالك[^.]*\.', '', text)
        text = re.sub(r'شكراً لك[^.]*\.', '', text)
        text = re.sub(r'شكرا لك[^.]*\.', '', text)
        
        # --- Greetings ---
        text = re.sub(r'مرحباً[،,]?\s*', '', text)
        
        # --- Tags ---
        text = re.sub(r'\[/INST\]', '', text)
        text = re.sub(r'INSTAINSTANT_ANSWER', '', text)
        text = re.sub(r'INSTANT_ANSWER', '', text)
        
        # --- Leading stray period or fragment ---
        text = re.sub(r'^\s*[.,،]\s*', '', text)
        
        # --- Cleanup ---
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[،,]\s*$', '', text)
        text = text.strip()
        
        # --- Trim obviously incomplete tail (e.g. "إن ما حدث" with no ending) ---
        if len(text) > 60 and text[-1] not in '.?!。':
            last_period = text.rfind('.')
            if last_period > 0:
                tail = text[last_period + 1:].strip()
                if 0 < len(tail) < 25:  # short fragment, likely incomplete
                    text = text[:last_period + 1].strip()
        
        return text
    
    def _build_rag_prompt(self, query: str, context: str) -> str:
        """Build RAG prompt with context."""
        
        # Check for critical safety topics
        critical_keywords = ['انتحار', 'انتحاري', 'suicide', 'قتل نفس', 'إنهاء الحياة', 'أريد الموت', 'أفضل الموت', 'مملة', 'لا فائدة', 'لا معنى', 'لا أمل']
        is_critical = any(keyword in query.lower() for keyword in critical_keywords)
        
        system_message = """أنت خبير فسيولوجي ونفسي متخصص. قدم إجابات داعمة ومطمئنة باللغة العربية فقط.

تعليمات صارمة:
1. ابدأ مباشرة بالإجابة - لا تحيات ولا أسماء (لا "السلام عليكم"، "بسم الله"، أسماء أطباء أو أشخاص).
2. استخدم فقط المعلومات الطبية من السياق - لا معلومات إضافية. لا تنسخ من النص: تواريخ، توقيعات، "تم التحديث بتاريخ"، "طبيبة عامة"، أو رموز (مثل INST_).
3. لا تذكر أسماء أدوية ولا جرعات ولا تفاصيل دوائية - اكتفِ دائماً بتوصية مراجعة الطبيب أو الصيدلي.
4. قدم رسالة مطمئنة وداعمة ثم توصية واضحة لاستشارة المختصين.
5. تجنب التناقضات - لا حالات عضوية إذا الحالة نفسية والعكس.
6. الطول: 200–400 حرف فقط (مختصر كإجابة أخصائي، 2–4 جمل). لا تطيل.
7. أنهِ الجملة حتى النهاية ثم توقف - لا تترك جملة ناقصة.
8. ركز على المعلومات الطبية الأساسية فقط - لا أسماء، لا تواريخ، لا أدوية بأسمائها.

هيكل الإجابة:
1. رسالة مطمئنة: ابدأ بتطمين السائل أن ما يمر به مفهوم ويمكن التعامل معه
2. شرح مختصر: اشرح الحالة بشكل مبسط ومطمئن (2-3 جمل)
3. توصية داعمة: شجع على طلب المساعدة المهنية بطريقة داعمة وغير مخيفة

للمواضيع الحرجة (انتحار، إيذاء النفس):
- ابدأ فوراً بتأكيد أن المساعدة متاحة وأن هذه الأفكار يمكن التعامل معها
- قدم رسالة أمل قوية
- شجع على التواصل الفوري مع مختص نفسي أو خط مساعدة
- ركز على الأمل وإمكانية التحسن"""
        
        # Safety message for critical topics
        safety_message = ""
        if is_critical:
            safety_message = """
⚠️ موضوع حرج - ابدأ فوراً:
- أكد أن المساعدة متاحة وهذه الأفكار يمكن التعامل معها
- شجع على التواصل الفوري مع مختص نفسي أو خط مساعدة
- ركز على الأمل وإمكانية التحسن
- لا تقلل من خطورة الوضع
"""
        
        user_message = f"""{safety_message}

المعلومات الطبية المرجعية (معلومات موحدة ومنسقة - استخدمها فقط):

{context}

السؤال: {query}

أجب على السؤال بهذا الهيكل:
1. رسالة مطمئنة وداعمة (ما يمر به السائل مفهوم ويمكن التعامل معه)
2. شرح مختصر للحالة بناءً على المعلومات المقدمة (2–3 جمل)
3. توصية لاستشارة المختصين

⚠️ مهم:
- الطول: 200–400 حرف (مختصر). لا تطيل. أنهِ الفكرة ثم توقف.
- لا تحيات، لا أسماء أشخاص أو أطباء، لا تواريخ، لا "تم التحديث"، لا أدوية بأسمائها أو جرعات.
- ابدأ مباشرة بالإجابة. استخدم المعلومات المقدمة فقط."""
        
        # Use chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def _build_general_prompt(self, query: str) -> str:
        """Build prompt for general knowledge (no context)."""
        system_message = """أنت خبير فسيولوجي ونفسي متخصص ومحترف في مجال الفسيولوجيا الطبية والصحة الإنسانية.

تعليمات أساسية:
1. الرد باللغة العربية فقط - لا تستخدم أي لغات أخرى
2. تقديم إجابات شاملة وعلمية دقيقة بناءً على المعرفة النفسية والفسيولوجية المتخصصة
3. عدم استخدام أي رموز غامضة أو علامات غير واضحة
4. استخدام لغة احترافية وواضحة وسهلة الفهم

معايير الإجابة:
- الشمولية: تغطية جميع جوانب السؤال بشكل دقيق
- الدقة العلمية: الالتزام بالمعرفة الفسيولوجية الحديثة
- الوضوح: استخدام لغة بسيطة وواضحة
- الاحترافية: الحفاظ على نبرة احترافية طبية

إجراءات السلامة والحماية:
1. تحديد المخاطر الصحية المحتملة في السؤال
2. عند اكتشاف أي مخاطر صحية محتملة:
   - حذر واضح من المخاطر
   - نصيحة قوية بطلب الدعم الطبي الرسمي من متخصصين معتمدين
   - عدم تقديم تشخيص نهائي أو علاج مباشر
3. التأكيد على أهمية استشارة الأطباء المتخصصين

معايير الجودة:
- الطول: إجابات شاملة وتفصيلية (200-400 كلمة)
- البناء: منظمة وسهلة المتابعة
- الموثوقية: معلومات دقيقة وموثوقة
- الفائدة: إجابات عملية وقابلة للتطبيق

ممنوع تماماً:
- استخدام رموز أو علامات غامضة
- تقديم تشخيصات نهائية بدون تحذير
- الخروج عن اللغة العربية

مرحباً! أنا هنا لتقديم معلومات نفسية وفسيولوجية دقيقة وموثوقة. كيف يمكنني مساعدتك؟"""
        
        user_message = f"السؤال: {query}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    @classmethod
    def build(
        cls,
        model_path: Optional[str] = None,
        device: str = None,  # Auto-detect if None
        load_in_4bit: bool = None,  # Auto-detect based on device
        load_in_8bit: bool = False,
    ) -> "RAGGenerator":
        """
        Build a RAGGenerator instance with auto-detection.
        
        Args:
            model_path: Path to model directory. Auto-detects if None.
            device: Device to use ('cpu' or 'cuda'). Auto-detects if None.
            load_in_4bit: Use 4-bit quantization. Auto-enabled for GPU, disabled for CPU.
            load_in_8bit: Use 8-bit quantization
        """
        # Auto-detect device if not specified
        if device is None:
            device = detect_device()
        
        # Auto-configure quantization based on device
        if load_in_4bit is None:
            if device == "cuda":
                # GPU: Try 4-bit first, but fall back to float16 if bitsandbytes unavailable
                # Float16 is still very fast on GPU and uses ~14GB (fits in 16GB V100)
                try:
                    import bitsandbytes as bnb
                    load_in_4bit = True
                    logger.info("[GENERATOR] Auto-enabled 4-bit quantization for GPU")
                except (ImportError, RuntimeError):
                    load_in_4bit = False
                    logger.info("[GENERATOR] bitsandbytes unavailable, using float16 on GPU (still fast)")
            else:
                # CPU: bitsandbytes is extremely slow, use float16 instead
                load_in_4bit = False
                logger.info("[GENERATOR] Auto-disabled 4-bit quantization for CPU (using float16)")
        
        if model_path is None:
            # Default path relative to project root
            import os
            package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(package_root, "Model", "Allam7B-Physiology-RAG-finetuned-final")
        
        return cls(
            model_path=model_path,
            device=device,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
