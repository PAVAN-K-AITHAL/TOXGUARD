"""Feedback Loop — Expert annotations → tuning recommendations.

Processes expert annotation CSVs (from eval_human.py) and generates
structured tuning recommendations for the RAG pipeline.

Usage:
    python feedback_loop.py --annotations ./eval_results/ --output tuning_report.json
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Annotation Loading
# ═══════════════════════════════════════════════════════════════════════

def load_annotations(annotations_dir: str) -> List[Dict]:
    """Load expert annotation CSVs from a directory.

    Reads all .csv files in the directory and combines them.

    Returns:
        List of annotation row dicts.
    """
    annotations = []

    if os.path.isfile(annotations_dir):
        # Single file
        with open(annotations_dir, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations.append(dict(row))
        logger.info(f"Loaded {len(annotations)} annotations from {annotations_dir}")
        return annotations

    # Directory
    for fname in sorted(os.listdir(annotations_dir)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(annotations_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations.append(dict(row))
        logger.info(f"Loaded {len(annotations)} annotations from {fname}")

    logger.info(f"Total annotations: {len(annotations)}")
    return annotations


# ═══════════════════════════════════════════════════════════════════════
#  Failure Analysis
# ═══════════════════════════════════════════════════════════════════════

def identify_retrieval_failures(annotations: List[Dict]) -> Dict:
    """Identify retrieval failures from expert annotations.

    Looks for "wrong_molecule", "omission", and "analogue_retrieval_failure"
    failure types in expert notes.

    Returns:
        Dict with failure patterns and suggested parameter tuning.
    """
    failure_types = Counter()
    molecules_with_failures = defaultdict(list)

    for ann in annotations:
        failure = ann.get("failure_type", "").strip().lower()
        molecule = ann.get("molecule", "")

        if failure in ("wrong_molecule", "omission", "analogue_retrieval_failure"):
            failure_types[failure] += 1
            molecules_with_failures[molecule].append(failure)

    n = len(annotations)
    suggestions = []

    wrong_mol_rate = failure_types.get("wrong_molecule", 0) / n if n else 0
    omission_rate = failure_types.get("omission", 0) / n if n else 0
    analogue_rate = failure_types.get("analogue_retrieval_failure", 0) / n if n else 0

    if wrong_mol_rate > 0.10:
        suggestions.append({
            "parameter": "min_relevance_threshold",
            "current_hint": 0.35,
            "recommended": 0.45,
            "reason": f"Wrong-molecule rate ({wrong_mol_rate:.0%}) is high. "
                      "Raise threshold to filter irrelevant semantic matches.",
        })

    if omission_rate > 0.15:
        suggestions.append({
            "parameter": "section_diversity_bonus",
            "current_hint": 0.08,
            "recommended": 0.12,
            "reason": f"Omission rate ({omission_rate:.0%}) is high. "
                      "Increase diversity bonus to retrieve more varied sections.",
        })

    if analogue_rate > 0.20:
        suggestions.append({
            "parameter": "exact_match_boost",
            "current_hint": 0.30,
            "recommended": 0.20,
            "reason": f"Analogue retrieval failure rate ({analogue_rate:.0%}) is high. "
                      "Reduce exact_match_boost to let semantic results compete "
                      "when exact match is absent (Phase 4 candidates).",
        })

    return {
        "failure_types": dict(failure_types),
        "affected_molecules": {
            mol: list(set(fails))
            for mol, fails in molecules_with_failures.items()
        },
        "retriever_suggestions": suggestions,
    }


def identify_generation_failures(annotations: List[Dict]) -> Dict:
    """Identify LLM generation failures from expert annotations.

    Looks for "hallucination", "dosage_error", "inappropriate_first_aid"
    failure types.

    Returns:
        Dict with hallucination patterns and prompt suggestions.
    """
    failure_types = Counter()
    section_failures = defaultdict(lambda: Counter())

    for ann in annotations:
        failure = ann.get("failure_type", "").strip().lower()
        section = ann.get("section", "")

        if failure in ("hallucination", "dosage_error", "inappropriate_first_aid"):
            failure_types[failure] += 1
            section_failures[section][failure] += 1

    suggestions = []

    if failure_types.get("hallucination", 0) > 5:
        # Find which sections hallucinate most
        worst_sections = sorted(
            section_failures.items(),
            key=lambda x: x[1].get("hallucination", 0),
            reverse=True,
        )[:3]

        for section, counts in worst_sections:
            if counts.get("hallucination", 0) > 2:
                suggestions.append({
                    "type": "prompt_engineering",
                    "section": section,
                    "change": f"Add explicit instruction: 'For {section}, only state "
                              "facts directly found in the retrieved documents. "
                              "If no relevant data is available, state \"Data not available\".'",
                })

    if failure_types.get("dosage_error", 0) > 3:
        suggestions.append({
            "type": "temperature_reduction",
            "section": "dose_response",
            "change": "Reduce temperature from 0.3 to 0.1 for dose_response section "
                      "to minimize creative fabrication of dose values.",
        })

    if failure_types.get("inappropriate_first_aid", 0) > 2:
        suggestions.append({
            "type": "prompt_engineering",
            "section": "first_aid",
            "change": "Add guard: 'First aid information must come from authoritative "
                      "sources in the retrieved documents. Do not invent procedures.'",
        })

    return {
        "failure_types": dict(failure_types),
        "section_breakdown": {
            section: dict(counts)
            for section, counts in section_failures.items()
        },
        "prompt_suggestions": suggestions,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Tuning Report
# ═══════════════════════════════════════════════════════════════════════

def generate_tuning_report(
    annotations: List[Dict],
    output_path: Optional[str] = None,
) -> Dict:
    """Generate a structured tuning report from expert annotations.

    Returns:
        Dict with retriever params, prompt changes, KB gaps, reranking weights.
    """
    retrieval = identify_retrieval_failures(annotations)
    generation = identify_generation_failures(annotations)

    # Expert rating analysis
    ratings = []
    for ann in annotations:
        rating_str = ann.get("expert_rating_1_5", "").strip()
        if rating_str and rating_str.isdigit():
            ratings.append(int(rating_str))

    # Knowledge base gap detection
    kb_gaps = []
    for ann in annotations:
        notes = ann.get("expert_notes", "").lower()
        if "missing" in notes or "not in kb" in notes or "not in database" in notes:
            mol = ann.get("molecule", "")
            section = ann.get("section", "")
            kb_gaps.append({"molecule": mol, "section": section, "note": notes[:200]})

    report = {
        "summary": {
            "total_annotations": len(annotations),
            "expert_ratings": {
                "count": len(ratings),
                "mean": sum(ratings) / len(ratings) if ratings else 0,
                "distribution": dict(Counter(ratings)),
            },
        },
        "retriever_params": retrieval["retriever_suggestions"],
        "prompt_changes": generation["prompt_suggestions"],
        "kb_gaps": kb_gaps[:20],
        "retrieval_failures": retrieval,
        "generation_failures": generation,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Tuning report saved to {output_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Feedback Loop: Expert annotations → tuning recommendations"
    )
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to annotations CSV or directory")
    parser.add_argument("--output", type=str, default="tuning_report.json",
                        help="Output path for tuning report JSON")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    annotations = load_annotations(args.annotations)

    if not annotations:
        logger.warning("No annotations found — nothing to process")
        return

    report = generate_tuning_report(annotations, output_path=args.output)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  FEEDBACK LOOP — TUNING REPORT")
    print(f"{'='*60}")
    print(f"\n  Annotations processed: {report['summary']['total_annotations']}")

    if report["summary"]["expert_ratings"]["count"]:
        print(f"  Mean expert rating: {report['summary']['expert_ratings']['mean']:.1f}/5")

    if report["retriever_params"]:
        print(f"\n  Retriever Parameter Suggestions:")
        for s in report["retriever_params"]:
            print(f"    → {s['parameter']}: {s.get('current_hint', '?')} → {s['recommended']}")
            print(f"      Reason: {s['reason']}")

    if report["prompt_changes"]:
        print(f"\n  Prompt Engineering Suggestions:")
        for s in report["prompt_changes"]:
            print(f"    → [{s['section']}] {s['change'][:100]}")

    if report["kb_gaps"]:
        print(f"\n  Knowledge Base Gaps: {len(report['kb_gaps'])} identified")
        for gap in report["kb_gaps"][:5]:
            print(f"    → {gap['molecule']} / {gap['section']}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
