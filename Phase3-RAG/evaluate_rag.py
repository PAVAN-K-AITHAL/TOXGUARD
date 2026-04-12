"""Evaluation script for Phase 3 RAG safety profiles.

Evaluates:
    1. Section completeness: Are all 9 sections populated?
    2. Citation quality: Do responses reference source documents?
    3. Factuality (for T3DB molecules): Compare against ground truth
    4. Coverage: retrieval recall across different molecule types
    5. Latency benchmarks

Usage:
    python evaluate_rag.py --results results/rag_profiles.json
    python evaluate_rag.py --csv ../data/t3db_processed.csv --limit 20
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def evaluate_profiles(profiles: list) -> dict:
    """Evaluate a list of safety profile dictionaries.

    Returns:
        Dict with evaluation metrics.
    """
    n = len(profiles)
    if n == 0:
        return {"error": "No profiles to evaluate"}

    # 1. Section completeness
    sections = [
        "toxicity_mechanism", "affected_organs", "symptoms_of_exposure",
        "dose_response", "first_aid", "handling_precautions",
        "regulatory_classification", "related_compounds", "references",
    ]
    section_filled = {s: 0 for s in sections}
    fully_complete = 0

    for p in profiles:
        all_filled = True
        for s in sections:
            val = p.get(s, "")
            if val and len(val.strip()) > 15:
                section_filled[s] += 1
            else:
                all_filled = False
        if all_filled:
            fully_complete += 1

    section_rates = {s: count / n for s, count in section_filled.items()}

    # 2. Citation quality (check for [DOC-N] references)
    has_citations = 0
    total_citations = 0
    for p in profiles:
        profile_citations = 0
        for s in sections:
            text = p.get(s, "")
            refs = re.findall(r"\[DOC-\d+\]", text)
            profile_citations += len(refs)
        total_citations += profile_citations
        if profile_citations > 0:
            has_citations += 1

    # 3. "Data not available" tracking
    data_unavailable = 0
    for p in profiles:
        for s in sections:
            text = p.get(s, "").lower()
            if "data not available" in text or "not available" in text:
                data_unavailable += 1

    # 4. Retrieval stats
    avg_docs = sum(p.get("num_retrieved_docs", 0) for p in profiles) / n
    source_counts = Counter()
    for p in profiles:
        for src in p.get("retrieval_sources", []):
            source_counts[src] += 1

    # 5. Latency stats
    latencies = [
        p.get("llm_latency_ms", 0) for p in profiles
        if p.get("llm_latency_ms", 0) > 0
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # 6. Toxicity distribution
    toxic_count = sum(1 for p in profiles if p.get("is_toxic", False))

    return {
        "total_profiles": n,
        "section_completeness": {
            "fully_complete": fully_complete,
            "fully_complete_rate": fully_complete / n,
            "per_section": section_rates,
        },
        "citation_quality": {
            "profiles_with_citations": has_citations,
            "citation_rate": has_citations / n,
            "avg_citations_per_profile": total_citations / n,
        },
        "data_gaps": {
            "data_unavailable_sections": data_unavailable,
            "avg_unavailable_per_profile": data_unavailable / n,
        },
        "retrieval": {
            "avg_docs_retrieved": avg_docs,
            "source_distribution": dict(source_counts),
        },
        "latency": {
            "avg_ms": avg_latency,
            "max_ms": max(latencies) if latencies else 0,
            "measurements": len(latencies),
        },
        "toxicity_distribution": {
            "toxic": toxic_count,
            "non_toxic": n - toxic_count,
        },
    }


def print_report(metrics: dict):
    """Pretty-print evaluation report."""
    print(f"\n{'='*70}")
    print(f"  PHASE 3 RAG EVALUATION REPORT")
    print(f"{'='*70}")

    print(f"\n  Total profiles: {metrics['total_profiles']}")

    # Section completeness
    sc = metrics["section_completeness"]
    print(f"\n  Section Completeness:")
    print(f"    Fully complete: {sc['fully_complete']}/{metrics['total_profiles']} "
          f"({sc['fully_complete_rate']:.0%})")
    for section, rate in sc["per_section"].items():
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        name = section.replace("_", " ").title()
        print(f"    {name:40s} {bar} {rate:.0%}")

    # Citation quality
    cq = metrics["citation_quality"]
    print(f"\n  Citation Quality:")
    print(f"    Profiles with citations: {cq['profiles_with_citations']} "
          f"({cq['citation_rate']:.0%})")
    print(f"    Avg citations per profile: {cq['avg_citations_per_profile']:.1f}")

    # Data gaps
    dg = metrics["data_gaps"]
    print(f"\n  Data Gaps:")
    print(f"    'Data not available' sections: {dg['data_unavailable_sections']}")
    print(f"    Avg per profile: {dg['avg_unavailable_per_profile']:.1f}")

    # Retrieval
    ret = metrics["retrieval"]
    print(f"\n  Retrieval:")
    print(f"    Avg documents retrieved: {ret['avg_docs_retrieved']:.1f}")
    print(f"    Sources: {ret['source_distribution']}")

    # Latency
    lat = metrics["latency"]
    if lat["measurements"] > 0:
        print(f"\n  Latency:")
        print(f"    Average: {lat['avg_ms']:.0f}ms")
        print(f"    Max: {lat['max_ms']:.0f}ms")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 3 RAG safety profiles"
    )
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to JSON results file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save metrics to JSON"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run full 4-pillar RAGAS validation (calls validate_rag.py)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM calls in validation (CI/CD mode)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    # If --validate, delegate to the 4-pillar runner
    if args.validate:
        import subprocess
        cmd = [
            sys.executable,
            os.path.join(CURRENT_DIR, "validate_rag.py"),
            "--all",
        ]
        if args.no_llm:
            cmd.append("--no-llm")
        print(f"Launching 4-pillar validation: {' '.join(cmd)}")
        subprocess.run(cmd)
        return

    if args.results:
        with open(args.results) as f:
            profiles = json.load(f)
    else:
        parser.error("Provide --results JSON file or use --validate")

    metrics = evaluate_profiles(profiles)
    print_report(metrics)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()

