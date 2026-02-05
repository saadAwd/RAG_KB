#!/usr/bin/env python3
"""
Comprehensive Model Comparison Script using BERTScore

Compares 5 model configurations:
1. Base ALLAM7B with zero-shot
2. Base ALLAM7B with few-shot
3. Fine-tuned ALLAM7B with zero-shot
4. Fine-tuned ALLAM7B with few-shot
5. Fine-tuned ALLAM7B with RAG architecture

Uses BERTScore for evaluation as per the notebook methodology.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from kb_retriever.generator import RAGGenerator, detect_device
from kb_retriever.rag_pipeline import RAGPipeline
from kb_retriever.hybrid_retriever import HybridKBRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Install bert_score if needed
try:
    from bert_score import score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  bert_score not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bert_score"])
    from bert_score import score
    BERTSCORE_AVAILABLE = True


# System prompt from the notebook
SYSTEM_PROMPT = """أنت خبير فسيولوجي متخصص ومحترف في مجال الفسيولوجيا الطبية والصحة الإنسانية.

تعليمات أساسية:
1. الرد باللغة العربية فقط - لا تستخدم أي لغات أخرى
2. تقديم إجابات شاملة وعلمية دقيقة بناءً على المعرفةالنفسيةوالفسيولوجية المتخصصة
3. استخدام المعلومات المرجعية كخلفية معرفية فقط لفهم الموضوع بشكل أعمق
4. عدم نسخ أو الإشارة إلى التفاصيل الشخصية من السياق
5. تعميم الإجابات لتكون قابلة للتطبيق على حالات مختلفة
6. عدم استخدام أي رموز غامضة أو علامات غير واضحة
7. استخدام لغة احترافية وواضحة وسهلة الفهم

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
- استخدام عبارات مثل "بناءً على النص" أو "حسب المعلومات المعطاة"
- نسخ الأسماء أو المعلومات الشخصية من السياق
- استخدام رموز أو علامات غامضة
- تقديم تشخيصات نهائية بدون تحذير
- الخروج عن اللغة العربية

مرحباً! أنا هنا لتقديم معلومات  نفسيةو فسيولوجية دقيقة وموثوقة. كيف يمكنني مساعدتك؟"""


def load_test_data(test_path: str) -> List[Dict]:
    """Load test questions from test.jsonl or TSV file."""
    test_data = []
    
    # Check if it's a TSV file
    if test_path.endswith('.tsv'):
        with open(test_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                if line.strip():
                    test_data.append({
                        'id': idx,
                        'question': line.strip(),
                    })
    else:
        # JSONL format
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Use 'text' field as question
                    test_data.append({
                        'id': item.get('id', 'unknown'),
                        'question': item.get('text', ''),
                    })
    return test_data


def load_reference_answers(reference_path: str) -> Dict[int, str]:
    """Load reference answers from TSV file."""
    reference_answers = {}
    with open(reference_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if line.strip():
                reference_answers[idx] = line.strip()
    return reference_answers


def get_hardcoded_fewshot_examples() -> List[Dict]:
    """Return hardcoded few-shot examples for physiology/mental health domain."""
    return [
        {
            'question': 'أعاني من قلق مستمر وصعوبة في النوم، ما هي النصائح التي يمكن أن تساعدني؟',
            'answer': 'يمكنك متابعة جلسات علاج نفسي مع معالج نفسي متخصص، كما يمكنك ممارسة تقنيات الاسترخاء والتأمل قبل النوم. من المهم أيضاً الحفاظ على روتين نوم منتظم وتجنب المنبهات قبل النوم. إذا استمرت المشكلة، يُنصح بمراجعة طبيب نفسي لتقييم حالتك ووصف العلاج المناسب.'
        },
        {
            'question': 'أشعر بالاكتئاب منذ فترة طويلة ولا أجد متعة في الأنشطة التي كنت أستمتع بها سابقاً، ما الحل؟',
            'answer': 'يجب التواصل مع طبيب نفسي من أجل تقييم وتشخيص الحالة والعلاج، كما يجب المتابعة مع معالج نفسي متخصص لعلاج الأسباب الكامنة التي تؤدي إلى الاكتئاب. من خلال العلاج النفسي يمكنك التحدث عن كل الأزمات والصدمات التي مررت بها وكل الأمور التي تشعر بها حالياً وتمر بها ويساعدك على تخطي كل ذلك والشعور بالتحسن التدريجي في حياتك الحالية.'
        },
        {
            'question': 'أعاني من نوبات هلع مفاجئة مع تسارع في نبضات القلب وصعوبة في التنفس، ما الذي يجب أن أفعله؟',
            'answer': 'نوبات الهلع يمكن أن تكون مخيفة ولكنها قابلة للعلاج. يجب مراجعة طبيب نفسي لتقييم حالتك وتحديد ما إذا كانت هذه نوبات هلع أم حالة طبية أخرى. يمكن للعلاج النفسي أن يساعدك في فهم أسباب هذه النوبات وتعلم تقنيات للتعامل معها. في بعض الحالات، قد يوصي الطبيب بعلاج دوائي أيضاً.'
        }
    ]


def load_train_data_for_fewshot(train_path: str, retriever: Optional[HybridKBRetriever] = None, num_examples: int = 3) -> List[Dict]:
    """Load few-shot examples from train.jsonl and retrieve answers from KB."""
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                train_data.append({
                    'id': item.get('id', 'unknown'),
                    'question': item.get('text', ''),
                })
    
    # Get random sample
    if len(train_data) > num_examples:
        selected = random.sample(train_data, num_examples)
    else:
        selected = train_data
    
    # Retrieve answers from KB for few-shot examples
    if retriever:
        for item in selected:
            question = item['question']
            try:
                retrieved = retriever.search(question, top_k=1, rerank=True)
                if retrieved:
                    answer = retrieved[0].get('chunk', {}).get('text', '')[:300]  # Limit length
                    item['answer'] = answer
                else:
                    item['answer'] = ""  # No answer found
            except Exception as e:
                print(f"⚠️  Warning: Could not retrieve answer for few-shot example: {e}")
                item['answer'] = ""
    else:
        # No retriever, just use questions
        for item in selected:
            item['answer'] = ""
    
    return selected


def format_chat_template(question: str, answer: Optional[str] = None, few_shot_examples: Optional[List[Dict]] = None) -> str:
    """Format chat template for Allam-7B with optional few-shot examples."""
    if few_shot_examples and len(few_shot_examples) > 0:
        # Build few-shot prompt with question-answer examples
        examples_text = "\n\nأمثلة:\n"
        for i, ex in enumerate(few_shot_examples[:3], 1):  # Use up to 3 examples
            ex_question = ex.get('question', '')
            ex_answer = ex.get('answer', '')
            if ex_question:
                examples_text += f"\nمثال {i}:\nالسؤال: {ex_question}\n"
                if ex_answer:
                    examples_text += f"الإجابة: {ex_answer}\n"
        
        prompt = f"""[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{examples_text}

السؤال الحالي: {question} [/INST]"""
    else:
        # Zero-shot prompt
        prompt = f"""[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{question} [/INST]"""
    
    if answer:
        prompt += f" {answer}</s>"
    
    return prompt


def generate_with_base_model(
    generator: RAGGenerator,
    question: str,
    few_shot_examples: Optional[List[Dict]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7
) -> str:
    """Generate response using base model (no fine-tuning)."""
    # Build prompt
    prompt = format_chat_template(question, few_shot_examples=few_shot_examples)
    
    # Tokenize
    inputs = generator.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(generator.device)
    
    # Generate
    with torch.inference_mode():
        outputs = generator.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=generator.tokenizer.pad_token_id,
            eos_token_id=generator.tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response (after [/INST])
    if "[/INST]" in generated_text:
        response = generated_text.split("[/INST]")[-1].strip()
    else:
        response = generated_text[len(prompt):].strip()
    
    return response


def generate_with_finetuned_model(
    generator: RAGGenerator,
    question: str,
    few_shot_examples: Optional[List[Dict]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7
) -> str:
    """Generate response using fine-tuned model."""
    # Use the generator's built-in method
    if few_shot_examples:
        # For few-shot, we need to build a custom prompt
        prompt = format_chat_template(question, few_shot_examples=few_shot_examples)
        
        inputs = generator.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(generator.device)
        
        with torch.inference_mode():
            outputs = generator.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=generator.tokenizer.pad_token_id,
                eos_token_id=generator.tokenizer.eos_token_id,
            )
        
        generated_text = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "[/INST]" in generated_text:
            response = generated_text.split("[/INST]")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        return response
    else:
        # Zero-shot: use generator's general knowledge method
        return generator.generate(
            query=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


def generate_with_rag(
    rag_pipeline: RAGPipeline,
    question: str,
    top_k: int = 5,
    alpha: float = 0.8,
    rerank_threshold: float = 0.6,
    max_new_tokens: int = 256,
    temperature: float = 0.7
) -> Dict:
    """Generate response using RAG pipeline with specified settings.

    rerank_threshold is in [0,1] and is compared to sigmoid(CrossEncoder raw) = score_rerank_norm.
    If max(score_rerank_norm) <= threshold, KB is discarded and the model answers without context.
    """
    rag_pipeline.retriever.alpha = alpha
    rag_pipeline.rerank_threshold = rerank_threshold

    result = rag_pipeline.process(
        query=question,
        top_k=top_k,
        alpha=alpha,
        rerank=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # For threshold: pipeline uses score_rerank_norm (sigmoid of raw). Store both for logging.
    norms = result.get('rerank_scores_norm') or []
    raws = result.get('rerank_scores') or []
    result['max_rerank_score_norm'] = max(norms, default=-1.0)  # value compared to threshold
    result['max_rerank_score'] = max(raws, default=-1.0)        # raw CrossEncoder logit (unbounded)
    result['rerank_discarded'] = not result.get('used_kb_context', False)

    return result


def _is_valid_response(r: Dict) -> bool:
    """Same filter as BERTScore loop: exclude ERROR only (for index alignment with f1_scores)."""
    return not str(r.get("response", "")).startswith("ERROR")


def run_rag_diagnostics(
    results: Dict[str, List[Dict]],
    bertscore_results: Dict[str, Dict],
) -> None:
    """
    Analyse RAG vs Fine-tuned (no RAG) to understand when RAG helps or hurts.
    Prints: KB usage, BERTScore by used_kb_context, per-question delta, and recommendations.
    """
    c3 = "config_3_finetuned_zeroshot"
    c5 = "config_5_finetuned_rag"
    if c3 not in results or c5 not in results or c3 not in bertscore_results or c5 not in bertscore_results:
        return
    if "f1_scores" not in bertscore_results[c3] or "f1_scores" not in bertscore_results[c5]:
        return

    # Use same filter as BERTScore: exclude ERROR (and SKIPPED if any)
    valid_3 = [r for r in results[c3] if _is_valid_response(r)]
    valid_5 = [r for r in results[c5] if _is_valid_response(r)]
    if not valid_3 or not valid_5:
        return

    f1_3 = bertscore_results[c3]["f1_scores"]
    f1_5 = bertscore_results[c5]["f1_scores"]
    if len(f1_3) != len(valid_3) or len(f1_5) != len(valid_5):
        return

    # Build (id -> f1) for config_3; (id -> (f1, used_kb)) for config_5
    by_id_3 = {r["id"]: f1_3[i] for i, r in enumerate(valid_3)}
    by_id_5 = {}
    for i, r in enumerate(valid_5):
        uid = r["id"]
        used = r.get("used_kb_context", False)
        by_id_5[uid] = (f1_5[i], used)

    common = [uid for uid in by_id_3 if uid in by_id_5]
    if not common:
        return

    # RAG: BERTScore by used_kb_context
    with_kb = [by_id_5[uid][0] for uid in common if by_id_5[uid][1]]
    without_kb = [by_id_5[uid][0] for uid in common if not by_id_5[uid][1]]

    # Per-question delta: RAG_F1 - FineTuned_F1
    deltas = []
    deltas_with_kb = []
    deltas_without_kb = []
    for uid in common:
        d = by_id_5[uid][0] - by_id_3[uid]
        deltas.append(d)
        if by_id_5[uid][1]:
            deltas_with_kb.append(d)
        else:
            deltas_without_kb.append(d)

    n = len(common)
    n_with = len(with_kb)
    n_without = len(without_kb)
    n_better = sum(1 for d in deltas if d > 0.001)
    n_worse = sum(1 for d in deltas if d < -0.001)
    n_same = n - n_better - n_worse

    # --- Print ---
    print("\n" + "=" * 80)
    print("RAG Diagnostics: When Does RAG Help or Hurt?")
    print("=" * 80)

    print("\n1) KB usage (Config 5 – Fine-tuned RAG)")
    print(f"   - With KB context:   {n_with:3d} / {n}  ({100*n_with/n:.1f}%)")
    if n_with:
        print(f"     Mean BERTScore F1: {sum(with_kb)/n_with:.4f}")
    print(f"   - Without KB (below rerank threshold): {n_without:3d} / {n}  ({100*n_without/n:.1f}%)")
    if n_without:
        print(f"     Mean BERTScore F1: {sum(without_kb)/n_without:.4f}")

    print("\n2) RAG vs Fine-tuned (no RAG) – per-question F1 delta")
    print(f"   - Mean delta (RAG − Fine-tuned): {sum(deltas)/n:+.4f}")
    print(f"   - RAG better (δ > 0.001):  {n_better:3d}  ({100*n_better/n:.1f}%)")
    print(f"   - RAG worse  (δ < −0.001): {n_worse:3d}  ({100*n_worse/n:.1f}%)")
    print(f"   - ~same     (|δ| ≤ 0.001): {n_same:3d}  ({100*n_same/n:.1f}%)")
    if deltas_with_kb:
        print(f"   - When RAG used KB:    mean δ = {sum(deltas_with_kb)/len(deltas_with_kb):+.4f}  (n={len(deltas_with_kb)})")
    if deltas_without_kb:
        print(f"   - When RAG skipped KB: mean δ = {sum(deltas_without_kb)/len(deltas_without_kb):+.4f}  (n={len(deltas_without_kb)})")

    print("\n3) Why RAG can have lower BERTScore than Fine-tuned (no RAG)")
    print("   - BERTScore measures lexical/semantic overlap with the reference.")
    print("   - Specialist references may use different wording than your KB.")
    print("   - RAG adds KB phrasing → less overlap with reference → lower BERTScore.")
    print("   - Retrieval can be off-topic for this test set → extra noise.")
    print("   - So: lower BERTScore does NOT always mean worse answers.")

    print("\n4) How to check if RAG is really better")
    print("   - Human evaluation: correctness, completeness, safety (gold standard).")
    print("   - If 'With KB' F1 > 'Without KB' F1: retrieval helps when used; consider")
    print("     raising rerank_threshold to use KB only when very confident.")
    print("   - If 'With KB' F1 < 'Without KB' F1: retrieval may be hurting; try")
    print("     higher rerank_threshold (e.g. 0.20–0.25) or check KB/retriever.")
    print("   - Ablation: run with --only-config5 and try different rerank_threshold,")
    print("     top_k, or alpha in the script to find better RAG settings.")
    print("=" * 80)


def calculate_bertscore(generated_answers: List[str], reference_answers: List[str]) -> Dict:
    """Calculate BERTScore metrics."""
    print("\n" + "="*70)
    print("Calculating BERTScore")
    print("="*70)
    print("Calculating BERTscore (this may take a few minutes)...")
    
    # Calculate BERTscore using Arabic model
    P, R, F1 = score(
        generated_answers,
        reference_answers,
        lang="ar",
        model_type="bert-base-multilingual-cased",
        verbose=True
    )
    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item(),
        'precision_scores': P.tolist(),
        'recall_scores': R.tolist(),
        'f1_scores': F1.tolist(),
    }


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Comparison with BERTScore')
    parser.add_argument('--skip-finetuned', action='store_true', 
                       help='Skip fine-tuned model configs (3, 4, 5) - only test base model')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Maximum number of questions to test (for quick testing)')
    parser.add_argument('--only-config5', action='store_true',
                       help='Only run config 5 (RAG) and update existing results')
    parser.add_argument('--tsv-test', action='store_true',
                       help='Use TSV test files (Subtask3_input_test and Subtask3_output_test)')
    parser.add_argument('--rerank-threshold', type=float, default=0.6,
                       help='RAG: use KB when max(sigmoid(reranker_raw)) > this; scale 0–1 (default 0.6).')
    parser.add_argument('--rag-top-k', type=int, default=3, help='RAG: number of chunks to retrieve (default 3)')
    parser.add_argument('--rag-alpha', type=float, default=0.8, help='RAG: hybrid retrieval alpha, 0=dense 1=sparse (default 0.8)')
    args = parser.parse_args()
    
    print("="*80)
    print("Model Comparison: 5 Configurations with BERTScore Evaluation")
    if args.only_config5:
        print("⚠️  MODE: Only running Config 5 (RAG) - will update existing results")
    if args.skip_finetuned:
        print("⚠️  MODE: Skipping fine-tuned configs (testing base model only)")
    if args.max_questions:
        print(f"⚠️  MODE: Testing only first {args.max_questions} questions")
    if args.rerank_threshold != 0.6 or args.rag_top_k != 3 or args.rag_alpha != 0.8:
        print(f"⚠️  RAG overrides: rerank_threshold={args.rerank_threshold}, rag_top_k={args.rag_top_k}, rag_alpha={args.rag_alpha}")
    print("="*80)
    
    # Paths
    if args.tsv_test:
        # Use TSV test files
        test_path = Path(__file__).parent / "Subtask3_input_test (1).tsv"
        reference_path = Path(__file__).parent / "Subtask3_output_test (2).tsv"
        train_path = None  # No few-shot for TSV test
        
        if not test_path.exists():
            print(f"❌ Error: Test input file not found: {test_path}")
            return
        if not reference_path.exists():
            print(f"❌ Error: Test output file not found: {reference_path}")
            return
        
        print(f"\n📊 Loading TSV test data...")
        print(f"   Input: {test_path}")
        print(f"   Reference: {reference_path}")
        test_data = load_test_data(str(test_path))
        reference_answers = load_reference_answers(str(reference_path))
        
        if args.max_questions:
            test_data = test_data[:args.max_questions]
            print(f"✓ Loaded {len(test_data)} test questions (limited to {args.max_questions})")
        else:
            print(f"✓ Loaded {len(test_data)} test questions")
        print(f"✓ Loaded {len(reference_answers)} reference answers")
    else:
        # Use JSONL format (original)
        test_path = Path(__file__).parent / "test.jsonl"
        train_path = Path(__file__).parent.parent / "train.jsonl"
        reference_path = None
        
        if not test_path.exists():
            print(f"❌ Error: Test file not found: {test_path}")
            return
        
        if not train_path.exists():
            print(f"⚠️  Warning: Train file not found: {train_path}")
            print("   Few-shot examples will not be available")
            train_path = None
        
        # Load test data
        print(f"\n📊 Loading test data from: {test_path}")
        test_data = load_test_data(str(test_path))
        if args.max_questions:
            test_data = test_data[:args.max_questions]
            print(f"✓ Loaded {len(test_data)} test questions (limited to {args.max_questions})")
        else:
            print(f"✓ Loaded {len(test_data)} test questions")
        
        reference_answers = None  # Will be loaded from KB later
    
    # Detect device first with diagnostics
    print("\n🔍 Checking GPU availability...")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB)")
    else:
        print("  ⚠️  CUDA not available")
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 13:
            print("  ⚠️  Python 3.13 detected - PyTorch CUDA builds not available yet")
            print("     The script will use CPU (will be slow: ~50-150s per question)")
            print("     For GPU support, use Python 3.11 or 3.12")
        else:
            print("     - PyTorch may not be installed with CUDA support")
            print("     - CUDA drivers may not be installed")
            print("     - Try: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    device = detect_device()
    print(f"\n🖥️  Selected device: {device}")
    
    if device == "cpu":
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 13:
            print("\n⚠️  Using CPU mode (Python 3.13 - no CUDA support yet)")
            print("   Expected time: ~10-15 hours for 70 questions")
            print("   For GPU support, switch to Python 3.11 or 3.12")
        else:
            print("\n⚠️  Warning: GPU not detected")
            print("   The script will use CPU (very slow).")
            print("   To use GPU, ensure PyTorch is installed with CUDA support")
    
    # Load existing results if --only-config5 flag is set
    existing_results = None
    if args.only_config5:
        results_file = Path(__file__).parent / "comparison_results" / "model_comparison_results.json"
        if results_file.exists():
            print(f"\n📂 Loading existing results from: {results_file}")
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_results = existing_data.get('results', {})
            print(f"✓ Loaded existing results for {len(existing_results)} configurations")
        else:
            print(f"⚠️  Warning: Existing results file not found: {results_file}")
            print("   Will create new results file")
    
    # Initialize retriever for few-shot examples and reference answers
    print("\n📊 Initializing retriever...")
    retriever = HybridKBRetriever.build(alpha=0.8, use_cpu=(device == "cpu"))
    print("✓ Retriever initialized")
    
    # Load few-shot examples
    few_shot_examples = None
    if args.tsv_test:
        # Use hardcoded few-shot examples for TSV test
        print("\n📊 Using hardcoded few-shot examples...")
        few_shot_examples = get_hardcoded_fewshot_examples()
        print(f"✓ Loaded {len(few_shot_examples)} hardcoded few-shot examples")
    elif train_path and train_path.exists():
        print(f"\n📊 Loading few-shot examples from: {train_path}")
        few_shot_examples = load_train_data_for_fewshot(str(train_path), retriever=retriever, num_examples=3)
        print(f"✓ Loaded {len(few_shot_examples)} few-shot examples")
        # Filter to only include examples with answers
        few_shot_examples = [ex for ex in few_shot_examples if ex.get('answer', '').strip()]
        print(f"✓ {len(few_shot_examples)} examples have answers from KB")
    else:
        print("\n⚠️  Train file not found, using hardcoded few-shot examples")
        few_shot_examples = get_hardcoded_fewshot_examples()
        print(f"✓ Loaded {len(few_shot_examples)} hardcoded few-shot examples")
    
    # Initialize models - Load on-demand to save GPU memory
    print("\n" + "="*80)
    print("Model Loading Strategy")
    print("="*80)
    print("⚠️  Loading models on-demand to fit in 8GB GPU memory")
    print("   Models will be loaded/unloaded as needed")
    print("="*80)
    
    # Store model loading functions instead of loading immediately
    base_model_name = "humain-ai/ALLaM-7B-Instruct-preview"
    base_tokenizer = None
    base_model = None
    base_generator = None
    
    finetuned_generator = None
    rag_pipeline = None
    
    def load_base_model():
        """Load base model on demand."""
        nonlocal base_model, base_tokenizer, base_generator
        
        if base_model is not None:
            return base_generator
        
        print("\n📦 Loading base model (humain-ai/ALLaM-7B-Instruct-preview)...")
        print("   Using 4-bit quantization (same as fine-tuned model for fair comparison)")
        
        # Clear GPU cache first
        if device == "cuda":
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Load tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        
        # Load base model with same quantization as fine-tuned model
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes as bnb
                # Use EXACT same quantization config as fine-tuned model
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"
                print("   ✓ Using 4-bit quantization (NF4, double quant)")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not use 4-bit quantization: {e}")
                print("   Falling back to float16")
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = None
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )
        
        if device != "cuda" or model_kwargs.get("device_map") is None:
            base_model = base_model.to(device)
        
        base_model.eval()
        
        # Create a simple wrapper for base model
        class BaseModelGenerator:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
        
        base_generator = BaseModelGenerator(base_model, base_tokenizer, device)
        print("✓ Base model loaded")
        
        # Verify device
        model_device = next(base_model.parameters()).device
        if model_device.type == "cuda":
            print(f"   ✓ Model on GPU: {model_device}")
        else:
            print(f"   ⚠️  Model on CPU: {model_device}")
        
        return base_generator
    
    def unload_base_model():
        """Unload base model to free GPU memory."""
        nonlocal base_model, base_generator
        if base_model is not None:
            print("\n🗑️  Unloading base model to free GPU memory...")
            del base_model
            base_model = None
            base_generator = None
            if device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            print("✓ Base model unloaded")
    
    def load_finetuned_model():
        """Load fine-tuned model on demand."""
        nonlocal finetuned_generator, rag_pipeline
        
        if finetuned_generator is not None:
            return finetuned_generator, rag_pipeline
        
        print("\n📦 Loading fine-tuned model...")
        print("   Using 4-bit quantization (same as base model for fair comparison)")
        
        # Clear GPU cache first
        if device == "cuda":
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        finetuned_generator = RAGGenerator.build(
            device=device,
            load_in_4bit=(device == "cuda"),  # Same quantization as base model
        )
        finetuned_generator.initialize()
        print("✓ Fine-tuned model loaded")
        
        # Verify model is on GPU
        model_device = next(finetuned_generator.model.parameters()).device
        if model_device.type == "cuda":
            print(f"   ✓ Model on GPU: {model_device}")
        else:
            print(f"   ⚠️  Model on CPU: {model_device} (may be slow)")
        
        # Initialize RAG Pipeline
        print("\n📦 Initializing RAG pipeline...")
        rag_generator = finetuned_generator
        
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            generator=rag_generator,
            rerank_threshold=args.rerank_threshold,
            max_context_chunks=3,
        )
        print(f"✓ RAG pipeline initialized (rerank_threshold={args.rerank_threshold})")
        
        return finetuned_generator, rag_pipeline
    
    def unload_finetuned_model():
        """Unload fine-tuned model to free GPU memory."""
        nonlocal finetuned_generator, rag_pipeline
        if finetuned_generator is not None:
            print("\n🗑️  Unloading fine-tuned model to free GPU memory...")
            del finetuned_generator
            del rag_pipeline
            finetuned_generator = None
            rag_pipeline = None
            if device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            print("✓ Fine-tuned model unloaded")
    
    # Generate responses for all configurations
    print("\n" + "="*80)
    print("Generating Responses")
    print("="*80)
    
    # Initialize results dictionary (config 4 removed)
    if args.only_config5 and existing_results:
        # Use existing results, but clear config 5
        results = existing_results.copy()
        results['config_5_finetuned_rag'] = []
        print("\n✓ Using existing results for configs 1-3")
        print("  Will regenerate config 5 only")
    else:
        results = {
            'config_1_base_zeroshot': [],
            'config_2_base_fewshot': [],
            'config_3_finetuned_zeroshot': [],
            'config_5_finetuned_rag': [],
        }
    
    # Pre-load reference answers (from TSV if using TSV test, otherwise from KB)
    if not args.tsv_test:
        print("\n📊 Pre-loading reference answers from KB...")
        reference_answers = {}
        for test_item in tqdm(test_data, desc="Loading references"):
            question = test_item['question']
            test_id = test_item['id']
            try:
                retrieved = retriever.search(question, top_k=1, rerank=True)
                if retrieved:
                    reference_answers[test_id] = retrieved[0].get('chunk', {}).get('text', '')[:500]
                else:
                    reference_answers[test_id] = "لا توجد إجابة متاحة في قاعدة المعرفة."
            except Exception as e:
                reference_answers[test_id] = "لا توجد إجابة متاحة في قاعدة المعرفة."
        print(f"✓ Loaded {len(reference_answers)} reference answers")
    else:
        # Reference answers already loaded from TSV
        print(f"✓ Using reference answers from TSV file ({len(reference_answers)} answers)")
    
    # ========================================================================
    # PHASE 1: Base Model Configurations (1 & 2)
    # ========================================================================
    if not args.only_config5:
        print("\n" + "="*80)
        print("PHASE 1: Base Model Configurations (1 & 2)")
        print("="*80)
        
        # Load base model once
        print("\n📦 Loading base model for configs 1 & 2...")
        base_generator = load_base_model()
        
        # Process all questions for configs 1 & 2
        print("\n📝 Processing all questions for Config 1 & 2...")
        for idx, test_item in enumerate(tqdm(test_data, desc="Config 1 & 2"), 1):
            question = test_item['question']
            test_id = test_item['id']
            reference_answer = reference_answers.get(test_id, "لا توجد إجابة متاحة في قاعدة المعرفة.")
            
            print(f"\n--- Question {idx}/{len(test_data)} (ID: {test_id}) ---")
            print(f"Question: {question[:100]}...")
            
            # Config 1: Base model zero-shot
            print("  → Config 1: Base zero-shot...")
            try:
                start_time = time.time()
                response_1 = generate_with_base_model(
                    base_generator,
                    question,
                    few_shot_examples=None,
                    max_new_tokens=256,
                    temperature=0.7
                )
                time_1 = time.time() - start_time
                results['config_1_base_zeroshot'].append({
                    'id': test_id,
                    'question': question,
                    'response': response_1,
                    'reference': reference_answer,
                    'generation_time': time_1,
                })
                print(f"    ✓ Generated in {time_1:.2f}s")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results['config_1_base_zeroshot'].append({
                    'id': test_id,
                    'question': question,
                    'response': f"ERROR: {str(e)}",
                    'reference': reference_answer,
                    'generation_time': 0,
                })
            
            # Config 2: Base model few-shot
            print("  → Config 2: Base few-shot...")
            try:
                start_time = time.time()
                response_2 = generate_with_base_model(
                    base_generator,
                    question,
                    few_shot_examples=few_shot_examples,
                    max_new_tokens=256,
                    temperature=0.7
                )
                time_2 = time.time() - start_time
                results['config_2_base_fewshot'].append({
                    'id': test_id,
                    'question': question,
                    'response': response_2,
                    'reference': reference_answer,
                    'generation_time': time_2,
                })
                print(f"    ✓ Generated in {time_2:.2f}s")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results['config_2_base_fewshot'].append({
                    'id': test_id,
                    'question': question,
                    'response': f"ERROR: {str(e)}",
                    'reference': reference_answer,
                    'generation_time': 0,
                })
        
        # Unload base model
        print("\n" + "="*80)
        print("Phase 1 Complete: Base model configs finished")
        print("="*80)
        unload_base_model()
    
    # ========================================================================
    # PHASE 2: Fine-tuned Model Configurations (3, 4, 5)
    # ========================================================================
    if args.only_config5 or (not args.skip_finetuned):
        print("\n" + "="*80)
        print("PHASE 2: Fine-tuned Model Configurations (3, 4, 5)")
        print("="*80)
        
        # Load fine-tuned model once
        print("\n📦 Loading fine-tuned model for configs 3, 4 & 5...")
        finetuned_generator, rag_pipeline = load_finetuned_model()
        
        # Process all questions for configs 3, 5 (or just 5 if --only-config5)
        # Note: Config 4 (finetuned-fewshot) is removed
        if args.only_config5:
            print("\n📝 Processing all questions for Config 5 only...")
            configs_to_run = [5]
        else:
            print("\n📝 Processing all questions for Config 3 & 5...")
            configs_to_run = [3, 5]
        
        for idx, test_item in enumerate(tqdm(test_data, desc=f"Config {', '.join(map(str, configs_to_run))}"), 1):
            question = test_item['question']
            test_id = test_item['id']
            reference_answer = reference_answers.get(test_id, "لا توجد إجابة متاحة في قاعدة المعرفة.")
            
            print(f"\n--- Question {idx}/{len(test_data)} (ID: {test_id}) ---")
            print(f"Question: {question[:100]}...")
            
            # Config 3: Fine-tuned model zero-shot
            if 3 in configs_to_run:
                print("  → Config 3: Fine-tuned zero-shot...")
                try:
                    start_time = time.time()
                    response_3 = generate_with_finetuned_model(
                        finetuned_generator,
                        question,
                        few_shot_examples=None,
                        max_new_tokens=256,
                        temperature=0.7
                    )
                    time_3 = time.time() - start_time
                    results['config_3_finetuned_zeroshot'].append({
                        'id': test_id,
                        'question': question,
                        'response': response_3,
                        'reference': reference_answer,
                        'generation_time': time_3,
                    })
                    print(f"    ✓ Generated in {time_3:.2f}s")
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    results['config_3_finetuned_zeroshot'].append({
                        'id': test_id,
                        'question': question,
                        'response': f"ERROR: {str(e)}",
                        'reference': reference_answer,
                        'generation_time': 0,
                    })
            
            # Config 4 removed (finetuned-fewshot)
            
            # Config 5: Fine-tuned model with RAG
            if 5 in configs_to_run:
                print("  → Config 5: Fine-tuned RAG...")
                try:
                    start_time = time.time()
                    rag_result = generate_with_rag(
                        rag_pipeline,
                        question,
                        top_k=args.rag_top_k,
                        alpha=args.rag_alpha,
                        rerank_threshold=args.rerank_threshold,
                        max_new_tokens=256,
                        temperature=0.7
                    )
                    time_5 = time.time() - start_time
                    response_5 = rag_result['response']
                    results['config_5_finetuned_rag'].append({
                        'id': test_id,
                        'question': question,
                        'response': response_5,
                        'reference': reference_answer,
                        'generation_time': time_5,
                        'used_kb_context': rag_result.get('used_kb_context', False),
                        'rerank_discarded': rag_result.get('rerank_discarded', False),
                        'num_chunks': rag_result.get('num_chunks', 0),
                        'max_rerank_score_norm': rag_result.get('max_rerank_score_norm'),
                    })
                    mn = rag_result.get('max_rerank_score_norm')
                    print(f"    ✓ Generated in {time_5:.2f}s (KB: {rag_result.get('used_kb_context', False)}, max_rerank_norm: {f'{mn:.4f}' if mn is not None else 'n/a'})")
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    results['config_5_finetuned_rag'].append({
                        'id': test_id,
                        'question': question,
                        'response': f"ERROR: {str(e)}",
                        'reference': reference_answer,
                        'generation_time': 0,
                    })
        
        # Unload fine-tuned model
        print("\n" + "="*80)
        print("Phase 2 Complete: Fine-tuned model configs finished")
        print("="*80)
        unload_finetuned_model()
    else:
        # Skip fine-tuned configs
        print("\n⏭️  Skipping fine-tuned model configs (--skip-finetuned flag)")
        for test_item in test_data:
            test_id = test_item['id']
            question = test_item['question']
            reference_answer = reference_answers.get(test_id, "لا توجد إجابة متاحة في قاعدة المعرفة.")
            
            results['config_3_finetuned_zeroshot'].append({
                'id': test_id,
                'question': question,
                'response': "SKIPPED",
                'reference': reference_answer,
                'generation_time': 0,
            })
            results['config_5_finetuned_rag'].append({
                'id': test_id,
                'question': question,
                'response': "SKIPPED",
                'reference': reference_answer,
                'generation_time': 0,
            })
    
    # Calculate BERTScore for each configuration
    print("\n" + "="*80)
    print("Calculating BERTScore Metrics")
    print("="*80)
    
    bertscore_results = {}
    
    for config_name, config_results in results.items():
        print(f"\n📊 Calculating BERTScore for: {config_name}")
        
        # Extract generated and reference answers
        generated = [r['response'] for r in config_results if not r['response'].startswith('ERROR')]
        references = [r['reference'] for r in config_results if not r['response'].startswith('ERROR')]
        
        if len(generated) == 0:
            print(f"  ⚠️  No valid responses for {config_name}")
            bertscore_results[config_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_valid': 0,
            }
            continue
        
        # Calculate BERTScore
        bertscore = calculate_bertscore(generated, references)
        bertscore_results[config_name] = {
            'precision': bertscore['precision'],
            'recall': bertscore['recall'],
            'f1': bertscore['f1'],
            'num_valid': len(generated),
            'precision_scores': bertscore['precision_scores'],
            'recall_scores': bertscore['recall_scores'],
            'f1_scores': bertscore['f1_scores'],
        }
        
        print(f"  ✓ Precision: {bertscore['precision']:.4f}")
        print(f"  ✓ Recall: {bertscore['recall']:.4f}")
        print(f"  ✓ F1: {bertscore['f1']:.4f}")
    
    # Calculate average generation times
    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)
    
    for config_name, config_results in results.items():
        valid_times = [r['generation_time'] for r in config_results if r['generation_time'] > 0]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            print(f"\n{config_name}:")
            print(f"  - Average generation time: {avg_time:.2f}s")
            print(f"  - Total valid responses: {len(valid_times)}/{len(config_results)}")
    
    # Save results
    output_dir = Path(__file__).parent / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / "model_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'bertscore_metrics': bertscore_results,
            'configurations': {
                'config_1': 'Base ALLAM7B - Zero-shot',
                'config_2': 'Base ALLAM7B - Few-shot',
                'config_3': 'Fine-tuned ALLAM7B - Zero-shot',
                'config_5': f"Fine-tuned ALLAM7B - RAG (top_k={args.rag_top_k}, alpha={args.rag_alpha}, rerank_threshold={args.rerank_threshold})",
            },
            'rag_settings': {
                'top_k': args.rag_top_k,
                'alpha': args.rag_alpha,
                'rerank_threshold': args.rerank_threshold,
            },
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("BERTScore Summary Table")
    print("="*80)
    print(f"{'Configuration':<40} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)
    for config_name, metrics in bertscore_results.items():
        config_label = {
            'config_1_base_zeroshot': '1. Base - Zero-shot',
            'config_2_base_fewshot': '2. Base - Few-shot',
            'config_3_finetuned_zeroshot': '3. Fine-tuned - Zero-shot',
            'config_5_finetuned_rag': '4. Fine-tuned - RAG',
        }.get(config_name, config_name)
        print(f"{config_label:<40} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # RAG diagnostics: when does RAG help vs hurt?
    run_rag_diagnostics(results, bertscore_results)
    
    print("\n" + "="*80)
    print("✅ Comparison Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
