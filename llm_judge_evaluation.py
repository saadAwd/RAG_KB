#!/usr/bin/env python
"""
LLM-as-Judge evaluation for Fine-tuned vs Fine-tuned+RAG.

Compares:
  - Config 3: Fine-tuned - Zero-shot        (config_3_finetuned_zeroshot)
  - Config 5: Fine-tuned - RAG              (config_5_finetuned_rag)

Uses an external LLM (e.g. OpenAI gpt-4o-mini) as a judge to decide which
answer is better per question.

Two modes:
  1. Blind comparison (default): Judge evaluates answers WITHOUT seeing the reference.
     This tests if RAG produces better answers in general.
  2. With reference (--include-reference): Judge sees the specialist reference answer.
     This tests if RAG helps match the reference style/content.

IMPORTANT:
- Do NOT hard-code your API key in this file.
- Set it via environment variable, e.g.:
    $env:OPENAI_API_KEY = "sk-..."     # Windows PowerShell
    export OPENAI_API_KEY="sk-..."     # Linux/macOS

Then run:
    cd RAG_KB
    python llm_judge_evaluation.py --model gpt-4o-mini                    # Blind comparison
    python llm_judge_evaluation.py --model gpt-4o-mini --include-reference  # With reference

You can limit the number of evaluated questions with --max-examples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # We will check later and print a clear error


@dataclass
class Example:
    id: int
    question: str
    reference: str
    answer_a: str  # Config 3: fine-tuned zero-shot
    answer_b: str  # Config 5: fine-tuned RAG


def load_examples(results_path: Path, max_examples: Optional[int] = None) -> List[Example]:
    """Load paired examples (config 3 vs config 5) from model_comparison_results.json."""
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results: Dict[str, List[Dict]] = data.get("results", {})
    cfg3 = results.get("config_3_finetuned_zeroshot", [])
    cfg5 = results.get("config_5_finetuned_rag", [])

    # Align by id
    by_id_3: Dict[int, Dict] = {int(r["id"]): r for r in cfg3 if isinstance(r.get("id"), (int, str))}
    by_id_5: Dict[int, Dict] = {int(r["id"]): r for r in cfg5 if isinstance(r.get("id"), (int, str))}

    common_ids = sorted(set(by_id_3.keys()) & set(by_id_5.keys()))
    examples: List[Example] = []

    for qid in common_ids:
        r3 = by_id_3[qid]
        r5 = by_id_5[qid]

        resp3 = str(r3.get("response", "")).strip()
        resp5 = str(r5.get("response", "")).strip()

        # Skip ERROR or SKIPPED entries
        if resp3.startswith("ERROR") or resp5.startswith("ERROR"):
            continue
        if resp3 == "SKIPPED" or resp5 == "SKIPPED":
            continue

        ex = Example(
            id=qid,
            question=str(r3.get("question", "")),
            reference=str(r3.get("reference", "")),
            answer_a=resp3,
            answer_b=resp5,
        )
        examples.append(ex)

    if max_examples is not None:
        examples = examples[:max_examples]

    return examples


def build_judge_prompt(ex: Example, include_reference: bool = False) -> List[Dict[str, str]]:
    """
    Build messages for the judge model.

    We ask the judge (Arabic-capable LLM) to evaluate two answers (A/B)
    to an Arabic mental health / physiology question.

    If include_reference=False, this is a "blind" comparison without the reference answer.
    """
    if include_reference:
        system_content = (
            "You are an expert Arabic-speaking medical and mental health QA evaluator. "
            "You will be given:\n"
            "1) A user's question (in Arabic)\n"
            "2) A specialist reference answer (in Arabic)\n"
            "3) Two model answers: Answer A and Answer B\n\n"
            "Your job is to judge which answer is better overall.\n\n"
            "Evaluation criteria (in order of importance):\n"
            "1. Correctness and clinical soundness (no harmful or incorrect advice)\n"
            "2. Completeness and usefulness relative to the question\n"
            "3. Safety (avoid risky self-treatment and urge to see a specialist or go to a hospital for critical cases)\n"
            "4. Alignment with the question and the reference answer's intent and main points\n"
            "5. Clarity and professionalism of Arabic writing\n\n"
        )
        user_content = (
            "السؤال (من المستخدم):\n"
            f"{ex.question}\n\n"
            "الإجابة المرجعية (من أخصائي):\n"
            f"{ex.reference}\n\n"
            "إجابة النموذج A (Fine-tuned - Zero-shot):\n"
            f"{ex.answer_a}\n\n"
            "إجابة النموذج B (Fine-tuned - RAG):\n"
            f"{ex.answer_b}\n\n"
        )
    else:
        system_content = (
            "You are an expert Arabic-speaking medical and mental health QA evaluator. "
            "You will be given:\n"
            "1) A user's question (in Arabic)\n"
            "2) Two model answers: Answer A and Answer B\n\n"
            "Your job is to judge which answer is better overall, WITHOUT comparing to any reference.\n\n"
            "Evaluation criteria (in order of importance):\n"
            "1. Correctness and clinical soundness (no harmful or incorrect advice)\n"
            "2. Completeness and usefulness relative to the question\n"
            "3. Safety (avoid risky self-treatment and urge to see a specialist or go to a hospital for critical cases)\n"
            "4. Relevance to the question (does it address what was asked?)\n"
            "5. Clarity and professionalism of Arabic writing\n\n"
        )
        user_content = (
            "السؤال (من المستخدم):\n"
            f"{ex.question}\n\n"
            "إجابة النموذج A (Fine-tuned - Zero-shot):\n"
            f"{ex.answer_a}\n\n"
            "إجابة النموذج B (Fine-tuned - RAG):\n"
            f"{ex.answer_b}\n\n"
        )

    system = {
        "role": "system",
        "content": (
            system_content +
            "VERY IMPORTANT:\n"
            "- Prefer the answer that is safer and more clinically sound, "
            "  even if it is slightly less detailed.\n"
            "- Do NOT penalize stylistic differences if medical content is correct.\n"
            "- If both answers are essentially equivalent in quality, you may choose a tie.\n\n"
            "Output STRICTLY in JSON with the following fields:\n"
            '{\n'
            '  \"score_a\": float,  // 0–10 overall quality of Answer A\n'
            '  \"score_b\": float,  // 0–10 overall quality of Answer B\n'
            '  \"winner\": \"A\" | \"B\" | \"tie\",\n'
            '  \"comments\": \"short Arabic explanation of your decision\"\n'
            '}\n'
            "Do not include any other text outside this JSON."
        ),
    }

    user = {
        "role": "user",
        "content": (
            user_content +
            "قيِّم الإجابات A و B بناءً على المعايير السابقة، ثم أعِد JSON فقط كما في المواصفات."
        ),
    }

    return [system, user]


def run_llm_judge(
    examples: List[Example],
    model: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    include_reference: bool = False,
) -> Dict:
    """Run LLM-as-judge over all examples and aggregate results."""
    if OpenAI is None:
        print("❌ The 'openai' Python package is not installed.")
        print("   Install it first with:  pip install openai")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is not set.")
        print("   Please set it before running this script.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    results = []
    wins_a = 0
    wins_b = 0
    ties = 0
    sum_score_a = 0.0
    sum_score_b = 0.0

    print(f"\nEvaluating {len(examples)} examples with model: {model}")

    for idx, ex in enumerate(examples, 1):
        print(f"\n--- Example {idx}/{len(examples)} (ID: {ex.id}) ---")
        messages = build_judge_prompt(ex, include_reference=include_reference)

        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                data = json.loads(content)

                score_a = float(data.get("score_a", 0.0))
                score_b = float(data.get("score_b", 0.0))
                winner = str(data.get("winner", "tie")).lower()
                comments = str(data.get("comments", "")).strip()

                if winner not in {"a", "b", "tie"}:
                    winner = "tie"

                if winner == "a":
                    wins_a += 1
                elif winner == "b":
                    wins_b += 1
                else:
                    ties += 1

                sum_score_a += score_a
                sum_score_b += score_b

                results.append(
                    {
                        "id": ex.id,
                        "question": ex.question,
                        "reference": ex.reference,
                        "answer_a": ex.answer_a,
                        "answer_b": ex.answer_b,
                        "score_a": score_a,
                        "score_b": score_b,
                        "winner": winner,
                        "comments": comments,
                    }
                )

                print(f"  → winner: {winner.upper()}, score_a={score_a:.2f}, score_b={score_b:.2f}")
                break

            except Exception as e:
                print(f"  ⚠️ LLM call failed (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    print("  ❌ Skipping this example after repeated failures.")
                    results.append(
                        {
                            "id": ex.id,
                            "error": str(e),
                        }
                    )
                else:
                    time.sleep(2 * attempt)

    total = len([r for r in results if "error" not in r])
    avg_a = sum_score_a / total if total else 0.0
    avg_b = sum_score_b / total if total else 0.0

    summary = {
        "total_evaluated": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "avg_score_a": avg_a,
        "avg_score_b": avg_b,
    }

    return {
        "summary": summary,
        "per_example": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation: Fine-tuned vs Fine-tuned+RAG")
    parser.add_argument(
        "--results-file",
        type=str,
        default=str(Path(__file__).parent / "comparison_results" / "model_comparison_results.json"),
        help="Path to model_comparison_results.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model name (e.g. gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional: limit number of evaluated questions (for quick testing)",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Include the specialist reference answer in the evaluation (default: blind comparison without reference)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        sys.exit(1)

    examples = load_examples(results_path, max_examples=args.max_examples)
    if not examples:
        print("❌ No valid examples found (check that configs 3 and 5 have responses).")
        sys.exit(1)

    eval_results = run_llm_judge(
        examples,
        model=args.model,
        include_reference=args.include_reference,
    )

    summary = eval_results["summary"]
    mode_str = "with reference" if args.include_reference else "blind (no reference)"
    print("\n" + "=" * 80)
    print(f"LLM-as-Judge Summary (Fine-tuned vs Fine-tuned+RAG) - {mode_str}")
    print("=" * 80)
    print(f"Total evaluated: {summary['total_evaluated']}")
    print(f"Wins A (Fine-tuned zero-shot): {summary['wins_a']}")
    print(f"Wins B (Fine-tuned RAG):       {summary['wins_b']}")
    print(f"Ties:                            {summary['ties']}")
    print(f"Avg score A: {summary['avg_score_a']:.3f}")
    print(f"Avg score B: {summary['avg_score_b']:.3f}")

    # Save detailed LLM-judge results
    suffix = "with_reference" if args.include_reference else "blind"
    out_path = Path(__file__).parent / "comparison_results" / f"llm_judge_results_{suffix}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Detailed LLM-judge results saved to: {out_path}")


if __name__ == "__main__":
    main()

