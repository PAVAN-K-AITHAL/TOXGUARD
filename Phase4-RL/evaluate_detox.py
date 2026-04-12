"""Evaluation metrics for Phase 4 molecule detoxification.

Computes and reports:
    1. Validity rate — % of generated IUPAC names that resolve to valid SMILES
    2. Detoxification rate — % of seeds where a less-toxic candidate is found
    3. Mean ΔP(toxic) — average toxicity reduction
    4. Mean Tanimoto — average structural similarity to seed
    5. Mean QED — average drug-likeness of candidates
    6. Agent convergence — mean rounds needed

Usage:
    python Phase4-RL/evaluate_detox.py --results outputs/detox_results.json
"""

import json
import logging
import os
import sys
from typing import Dict, List

logger = logging.getLogger(__name__)


def evaluate_reports(reports: List[Dict]) -> Dict:
    """Compute evaluation metrics from a list of DetoxReport dicts.

    Args:
        reports: List of serialized DetoxReport dicts.

    Returns:
        Dict with aggregated metrics.
    """
    n_total = len(reports)
    if n_total == 0:
        return {"error": "No reports to evaluate"}

    n_success = sum(1 for r in reports if r.get("success", False))
    total_generated = sum(r.get("total_generated", 0) for r in reports)
    total_valid = sum(r.get("total_valid", 0) for r in reports)
    total_less_toxic = sum(r.get("total_less_toxic", 0) for r in reports)
    total_rounds = sum(r.get("rounds", 0) for r in reports)
    total_time = sum(r.get("time_s", 0) for r in reports)

    # Delta P(toxic) for successful cases
    delta_toxicities = []
    tanimotos = []
    qeds = []
    for r in reports:
        bc = r.get("best_candidate")
        if bc and r.get("success"):
            delta = r.get("seed_p_toxic", 0) - bc.get("p_toxic", 0)
            delta_toxicities.append(delta)
            tanimotos.append(bc.get("tanimoto", 0))
            qeds.append(bc.get("qed", 0))

    metrics = {
        # Overview
        "n_molecules": n_total,
        "n_success": n_success,
        "detoxification_rate": n_success / n_total if n_total > 0 else 0,

        # Generation quality
        "total_candidates_generated": total_generated,
        "total_valid": total_valid,
        "validity_rate": total_valid / total_generated if total_generated > 0 else 0,
        "total_less_toxic": total_less_toxic,

        # Toxicity reduction (successful cases)
        "mean_delta_p_toxic": (
            sum(delta_toxicities) / len(delta_toxicities)
            if delta_toxicities else 0
        ),
        "max_delta_p_toxic": max(delta_toxicities) if delta_toxicities else 0,

        # Structural properties (successful cases)
        "mean_tanimoto": sum(tanimotos) / len(tanimotos) if tanimotos else 0,
        "mean_qed": sum(qeds) / len(qeds) if qeds else 0,

        # Agent efficiency
        "mean_rounds": total_rounds / n_total if n_total > 0 else 0,
        "mean_time_s": total_time / n_total if n_total > 0 else 0,
    }

    return metrics


def format_metrics(metrics: Dict) -> str:
    """Format metrics dict as a readable report."""
    lines = [
        "=" * 60,
        "  PHASE 4 DETOXIFICATION EVALUATION",
        "=" * 60,
        "",
        "  OVERVIEW",
        "-" * 60,
        f"  Molecules tested:       {metrics.get('n_molecules', 0)}",
        f"  Successfully detoxified: {metrics.get('n_success', 0)}",
        f"  Detoxification rate:    {metrics.get('detoxification_rate', 0):.1%}",
        "",
        "  GENERATION QUALITY",
        "-" * 60,
        f"  Total candidates:       {metrics.get('total_candidates_generated', 0)}",
        f"  Valid candidates:       {metrics.get('total_valid', 0)}",
        f"  Validity rate:          {metrics.get('validity_rate', 0):.1%}",
        f"  Less-toxic candidates:  {metrics.get('total_less_toxic', 0)}",
        "",
        "  TOXICITY REDUCTION (successful cases)",
        "-" * 60,
        f"  Mean ΔP(toxic):         {metrics.get('mean_delta_p_toxic', 0):.4f}",
        f"  Max ΔP(toxic):          {metrics.get('max_delta_p_toxic', 0):.4f}",
        "",
        "  STRUCTURAL PROPERTIES (successful cases)",
        "-" * 60,
        f"  Mean Tanimoto:          {metrics.get('mean_tanimoto', 0):.4f}",
        f"  Mean QED:               {metrics.get('mean_qed', 0):.4f}",
        "",
        "  AGENT EFFICIENCY",
        "-" * 60,
        f"  Mean rounds:            {metrics.get('mean_rounds', 0):.1f}",
        f"  Mean time per mol:      {metrics.get('mean_time_s', 0):.1f}s",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate detoxification results")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("-o", "--output", help="Output metrics JSON path")
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    # Handle both single result and list of results
    if isinstance(data, dict):
        data = [data]

    metrics = evaluate_reports(data)

    print(format_metrics(metrics))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
