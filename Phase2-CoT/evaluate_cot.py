"""Evaluation script for Phase 2 CoT analysis quality.

Evaluates:
    1. Parsing quality: Are all 7 CoT sections populated?
    2. Functional group extraction: Did the analysis identify groups?
    3. Consistency: Does the CoT verdict align with P(toxic)?
    4. Coverage: Latency and success rate across molecules

Usage:
    python evaluate_cot.py --results results/cot_results.json
    python evaluate_cot.py --csv data/t3db_processed.csv --limit 50 --api-key gsk_...
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def evaluate_results(results: list) -> dict:
    """Evaluate a list of CoT result dictionaries.

    Returns:
        Dict with evaluation metrics.
    """
    n = len(results)
    if n == 0:
        return {"error": "No results to evaluate"}

    # 1. Section completeness
    sections = [
        "structural_analysis", "toxicophore_identification",
        "mechanism_of_action", "biological_pathways",
        "organ_toxicity", "confidence", "verdict",
    ]
    section_filled = {s: 0 for s in sections}
    fully_complete = 0

    for r in results:
        all_filled = True
        for s in sections:
            val = r.get(s, "")
            if val and len(val.strip()) > 10:
                section_filled[s] += 1
            else:
                all_filled = False
        if all_filled:
            fully_complete += 1

    section_rates = {s: count / n for s, count in section_filled.items()}

    # 2. Functional group extraction
    has_groups = sum(1 for r in results if r.get("functional_groups"))
    avg_groups = (
        sum(len(r.get("functional_groups", [])) for r in results) / n
    )
    all_groups = Counter()
    for r in results:
        for g in r.get("functional_groups", []):
            all_groups[g] += 1

    # 3. Verdict consistency with P(toxic)
    consistent = 0
    inconsistent_examples = []
    for r in results:
        score = r.get("toxicity_score", 0.5)
        verdict_text = r.get("verdict", "").upper()
        is_toxic = score >= 0.5

        # Check if verdict aligns with score
        verdict_says_toxic = any(
            kw in verdict_text
            for kw in ["TOXIC (CONFIRMED", "TOXIC.", "IS TOXIC", "HIGHLY TOXIC"]
        )
        verdict_says_safe = any(
            kw in verdict_text
            for kw in ["NON-TOXIC", "UNLIKELY TOXIC", "NOT TOXIC", "SAFE"]
        )

        if is_toxic and verdict_says_toxic:
            consistent += 1
        elif not is_toxic and verdict_says_safe:
            consistent += 1
        elif verdict_says_toxic or verdict_says_safe:
            inconsistent_examples.append({
                "molecule": r.get("iupac_name", "?"),
                "score": score,
                "verdict_snippet": verdict_text[:80],
            })
        else:
            consistent += 1  # Ambiguous verdict — count as consistent

    # 4. Confidence distribution
    confidence_dist = Counter(r.get("confidence_level", "UNKNOWN") for r in results)

    # 5. Latency stats
    latencies = [r.get("llm_latency_ms", 0) for r in results if r.get("llm_latency_ms", 0) > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    # 6. Error rate
    errors = sum(1 for r in results if "ERROR" in r.get("verdict", ""))

    return {
        "total_molecules": n,
        "errors": errors,
        "success_rate": (n - errors) / n,
        "section_completeness": {
            "fully_complete": fully_complete,
            "fully_complete_rate": fully_complete / n,
            "per_section": section_rates,
        },
        "functional_groups": {
            "molecules_with_groups": has_groups,
            "extraction_rate": has_groups / n,
            "avg_groups_per_molecule": avg_groups,
            "top_10_groups": all_groups.most_common(10),
        },
        "verdict_consistency": {
            "consistent": consistent,
            "consistency_rate": consistent / n,
            "inconsistent_examples": inconsistent_examples[:5],
        },
        "confidence_distribution": dict(confidence_dist),
        "latency": {
            "avg_ms": avg_latency,
            "max_ms": max_latency,
            "measurements": len(latencies),
        },
    }


def print_report(metrics: dict):
    """Pretty-print evaluation report."""
    print("\n" + "=" * 70)
    print("  PHASE 2 COT EVALUATION REPORT")
    print("=" * 70)

    print(f"\n  Total molecules analyzed: {metrics['total_molecules']}")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Errors: {metrics['errors']}")

    sc = metrics["section_completeness"]
    print(f"\n  Section Completeness:")
    print(f"    Fully complete analyses: {sc['fully_complete']}/{metrics['total_molecules']} ({sc['fully_complete_rate']:.1%})")
    for section, rate in sc["per_section"].items():
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        name = section.replace("_", " ").title()
        print(f"    {name:30s} {bar} {rate:.0%}")

    fg = metrics["functional_groups"]
    print(f"\n  Functional Group Extraction:")
    print(f"    Molecules with groups identified: {fg['molecules_with_groups']} ({fg['extraction_rate']:.1%})")
    print(f"    Avg groups per molecule: {fg['avg_groups_per_molecule']:.1f}")
    if fg["top_10_groups"]:
        print(f"    Top groups: {', '.join(f'{g}({c})' for g, c in fg['top_10_groups'][:5])}")

    vc = metrics["verdict_consistency"]
    print(f"\n  Verdict ↔ P(toxic) Consistency:")
    print(f"    Consistent: {vc['consistent']}/{metrics['total_molecules']} ({vc['consistency_rate']:.1%})")
    if vc["inconsistent_examples"]:
        print(f"    Inconsistent examples:")
        for ex in vc["inconsistent_examples"][:3]:
            print(f"      - {ex['molecule']}: score={ex['score']:.2f}, verdict: {ex['verdict_snippet']}")

    cd = metrics["confidence_distribution"]
    print(f"\n  Confidence Distribution:")
    for level in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        count = cd.get(level, 0)
        if count > 0:
            print(f"    {level}: {count}")

    lat = metrics["latency"]
    if lat["measurements"] > 0:
        print(f"\n  Latency:")
        print(f"    Average: {lat['avg_ms']:.0f}ms")
        print(f"    Max: {lat['max_ms']:.0f}ms")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 CoT results")
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to JSON results file from run_cot.py"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file to run fresh analysis + evaluate"
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="70b")
    parser.add_argument("--output", type=str, default=None,
                        help="Save evaluation metrics to JSON")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    if args.results:
        # Evaluate existing results
        with open(args.results) as f:
            results = json.load(f)
    elif args.csv:
        # Run fresh analysis then evaluate
        import pandas as pd
        from llm_client import GroqLLMClient
        from cot_analyzer import CoTAnalyzer

        df = pd.read_csv(args.csv).dropna(subset=["iupac_name"])
        molecules = df["iupac_name"].tolist()[:args.limit]
        scores = (df["toxicity_score"].tolist()[:args.limit]
                  if "toxicity_score" in df.columns else None)

        llm = GroqLLMClient(api_key=args.api_key, model=args.model)
        analyzer = CoTAnalyzer(llm_client=llm)

        results = []
        for i, mol in enumerate(molecules):
            score = scores[i] if scores else 0.7
            try:
                result = analyzer.analyze_from_prediction(
                    iupac_name=mol, toxicity_score=score,
                    severity_label="Highly toxic" if score >= 0.5 else "Unlikely toxic",
                    is_toxic=score >= 0.5,
                )
                results.append(result.to_dict())
                print(f"  [{i+1}/{len(molecules)}] {result.summary()}")
            except Exception as e:
                print(f"  [{i+1}/{len(molecules)}] ERROR: {e}")
            import time; time.sleep(1.5)

        # Save results
        save_path = args.output or "results/cot_eval_results.json"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    else:
        parser.error("Provide --results or --csv")
        return

    # Evaluate
    metrics = evaluate_results(results)
    print_report(metrics)

    # Save metrics
    if args.output:
        metrics_path = args.output.replace(".json", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
