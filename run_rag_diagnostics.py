#!/usr/bin/env python3
"""
Run RAG diagnostics on existing model_comparison_results.json.

Use this to understand when RAG helps vs hurts without re-running the full comparison:
  - KB usage (with vs without KB context)
  - BERTScore by used_kb_context
  - Per-question delta: RAG F1 - Fine-tuned F1
  - Recommendations

Usage:
  python run_rag_diagnostics.py
  python run_rag_diagnostics.py --file comparison_results/model_comparison_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _is_valid_response(r: dict) -> bool:
    """Same filter as BERTScore loop: exclude ERROR (for index alignment with f1_scores)."""
    return not str(r.get("response", "")).startswith("ERROR")


def run_rag_diagnostics(
    results: Dict[str, List[dict]],
    bertscore_results: Dict[str, dict],
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

    valid_3 = [r for r in results[c3] if _is_valid_response(r)]
    valid_5 = [r for r in results[c5] if _is_valid_response(r)]
    if not valid_3 or not valid_5:
        return

    f1_3 = bertscore_results[c3]["f1_scores"]
    f1_5 = bertscore_results[c5]["f1_scores"]
    if len(f1_3) != len(valid_3) or len(f1_5) != len(valid_5):
        return

    by_id_3 = {r["id"]: f1_3[i] for i, r in enumerate(valid_3)}
    by_id_5 = {}
    for i, r in enumerate(valid_5):
        uid = r["id"]
        used = r.get("used_kb_context", False)
        by_id_5[uid] = (f1_5[i], used)

    common = [uid for uid in by_id_3 if uid in by_id_5]
    if not common:
        return

    with_kb = [by_id_5[uid][0] for uid in common if by_id_5[uid][1]]
    without_kb = [by_id_5[uid][0] for uid in common if not by_id_5[uid][1]]

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

    print("\n" + "=" * 80)
    print("RAG Diagnostics: When Does RAG Help or Hurt?")
    print("=" * 80)

    print("\n1) KB usage (Config 5 - Fine-tuned RAG)")
    print(f"   - With KB context:   {n_with:3d} / {n}  ({100*n_with/n:.1f}%)")
    if n_with:
        print(f"     Mean BERTScore F1: {sum(with_kb)/n_with:.4f}")
    print(f"   - Without KB (below rerank threshold): {n_without:3d} / {n}  ({100*n_without/n:.1f}%)")
    if n_without:
        print(f"     Mean BERTScore F1: {sum(without_kb)/n_without:.4f}")

    print("\n2) RAG vs Fine-tuned (no RAG) - per-question F1 delta")
    print(f"   - Mean delta (RAG - Fine-tuned): {sum(deltas)/n:+.4f}")
    print(f"   - RAG better (d > 0.001):  {n_better:3d}  ({100*n_better/n:.1f}%)")
    print(f"   - RAG worse  (d < -0.001): {n_worse:3d}  ({100*n_worse/n:.1f}%)")
    print(f"   - ~same     (|d| <= 0.001): {n_same:3d}  ({100*n_same/n:.1f}%)")
    if deltas_with_kb:
        print(f"   - When RAG used KB:    mean d = {sum(deltas_with_kb)/len(deltas_with_kb):+.4f}  (n={len(deltas_with_kb)})")
    if deltas_without_kb:
        print(f"   - When RAG skipped KB: mean d = {sum(deltas_without_kb)/len(deltas_without_kb):+.4f}  (n={len(deltas_without_kb)})")

    print("\n3) Why RAG can have lower BERTScore than Fine-tuned (no RAG)")
    print("   - BERTScore measures lexical/semantic overlap with the reference.")
    print("   - Specialist references may use different wording than your KB.")
    print("   - RAG adds KB phrasing -> less overlap with reference -> lower BERTScore.")
    print("   - Retrieval can be off-topic for this test set -> extra noise.")
    print("   - So: lower BERTScore does NOT always mean worse answers.")

    print("\n4) How to check if RAG is really better")
    print("   - Human evaluation: correctness, completeness, safety (gold standard).")
    print("   - If 'With KB' F1 > 'Without KB' F1: retrieval helps when used; consider")
    print("     raising rerank_threshold to use KB only when very confident.")
    print("   - If 'With KB' F1 < 'Without KB' F1: retrieval may be hurting; try")
    print("     higher rerank_threshold (e.g. 0.20-0.25) or check KB/retriever.")
    print("   - Ablation: run with --only-config5 and try different rerank_threshold,")
    print("     top_k, or alpha in the script to find better RAG settings.")
    print("=" * 80)


def main() -> None:
    p = argparse.ArgumentParser(description="RAG diagnostics from existing comparison results")
    p.add_argument(
        "--file",
        default=None,
        help="Path to model_comparison_results.json (default: comparison_results/model_comparison_results.json)",
    )
    args = p.parse_args()

    path = Path(args.file) if args.file else Path(__file__).parent / "comparison_results" / "model_comparison_results.json"
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    bertscore_results = data.get("bertscore_metrics", {})
    if not results or not bertscore_results:
        print("No 'results' or 'bertscore_metrics' in the JSON.")
        return

    run_rag_diagnostics(results, bertscore_results)


if __name__ == "__main__":
    main()
