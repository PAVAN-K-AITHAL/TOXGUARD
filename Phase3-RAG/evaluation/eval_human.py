"""Pillar 3 — Human Evaluation & Golden Test Set.

Auto-scores generated profiles against the golden test set, runs RAGAS
full evaluation, and exports annotation sheets for expert review.

Solo annotator workflow: expert only reviews RAGAS-flagged failures
(faithfulness <= 0.70) rather than all profiles.

Usage:
    python eval_human.py --golden golden_test_set.json --db-dir ./chroma_db
    python eval_human.py --golden golden_test_set.json --output-dir ./eval_results
    python eval_human.py --golden golden_test_set.json --no-llm
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional

# ── Path setup ─────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PHASE3_DIR)
sys.path.insert(0, PHASE3_DIR)
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Faithfulness threshold below which we flag for expert review
EXPERT_REVIEW_THRESHOLD = 0.70


# ═══════════════════════════════════════════════════════════════════════
#  Golden Test Set Loading
# ═══════════════════════════════════════════════════════════════════════

def load_golden_set(path: str) -> Dict:
    """Load and validate the golden test set JSON."""
    with open(path) as f:
        data = json.load(f)

    molecules = data.get("molecules", [])
    logger.info(
        f"Loaded golden test set v{data.get('version', '?')}: "
        f"{len(molecules)} molecules"
    )

    # Separate by tier
    tiers = {}
    for mol in molecules:
        tier = mol.get("tier", "unknown")
        tiers.setdefault(tier, []).append(mol)

    for tier, mols in tiers.items():
        logger.info(f"  {tier}: {len(mols)} molecules")

    return data


# ═══════════════════════════════════════════════════════════════════════
#  Auto-Scoring Against Golden Set
# ═══════════════════════════════════════════════════════════════════════

def auto_score_against_golden(
    profile: Dict,
    golden_entry: Dict,
    use_ground_truth: bool = False,
) -> Dict:
    """Auto-score a generated profile against golden expectations.

    Checks:
        - Keyword coverage: expected mechanism keywords in output
        - Organ coverage: expected target organs mentioned
        - LD50 accuracy: reported values match expected
        - GHS match: regulatory classification correct

    Args:
        profile: Generated profile dict (may be empty in --no-llm mode).
        golden_entry: Golden test set entry with expected values.
        use_ground_truth: If True and profile is empty, fall back to
            ground_truth_summary for scoring (validates golden set consistency).

    Returns:
        Dict with per-check scores and overall auto_score.
    """
    expected = golden_entry.get("expected", {})

    # Build text to score against
    profile_text = _build_full_text(profile)
    if not profile_text.strip() and use_ground_truth:
        # No-LLM mode: use ground_truth_summary as a proxy
        profile_text = expected.get("ground_truth_summary", "")
    full_text = profile_text.lower()

    # 1. Keyword coverage
    expected_keywords = expected.get("mechanism_keywords", [])
    if expected_keywords:
        found = sum(1 for kw in expected_keywords if kw.lower() in full_text)
        keyword_coverage = found / len(expected_keywords)
    else:
        keyword_coverage = 1.0

    # 2. Organ coverage
    expected_organs = expected.get("target_organs", [])
    if expected_organs:
        found = sum(1 for org in expected_organs if org.lower() in full_text)
        organ_coverage = found / len(expected_organs)
    else:
        organ_coverage = 1.0

    # 3. LD50 accuracy
    expected_ld50 = expected.get("ld50_values", [])
    if expected_ld50:
        found = 0
        for val in expected_ld50:
            if val.lower() in full_text:
                found += 1
            else:
                # Try numeric match (strip units and check number)
                nums = re.findall(r"\d+[\d,.]*", val)
                for num in nums:
                    if num in full_text:
                        found += 1
                        break
        ld50_accuracy = found / len(expected_ld50)
    else:
        ld50_accuracy = 1.0

    # 4. GHS match
    expected_ghs = expected.get("ghs_category", [])
    if expected_ghs:
        found = sum(1 for g in expected_ghs if g.lower() in full_text)
        ghs_match = found / len(expected_ghs)
    else:
        ghs_match = 1.0

    # Overall auto-score (equal weights)
    auto_score = (keyword_coverage + organ_coverage + ld50_accuracy + ghs_match) / 4

    return {
        "keyword_coverage": keyword_coverage,
        "organ_coverage": organ_coverage,
        "ld50_accuracy": ld50_accuracy,
        "ghs_match": ghs_match,
        "auto_score": auto_score,
        "expected_keywords_found": [
            kw for kw in expected_keywords if kw.lower() in full_text
        ],
        "expected_keywords_missing": [
            kw for kw in expected_keywords if kw.lower() not in full_text
        ],
    }


def auto_score_phase4_pair(
    seed_profile: Dict,
    candidate_profile: Dict,
    golden_pair: Dict,
) -> Dict:
    """Auto-score a Phase 4 detox pair against golden expectations.

    Checks:
        - Did retriever find analogue docs for the candidate?
        - Does candidate profile reference parent toxicology?
        - Structural relationship explained?
        - Candidate mechanism keywords present?
    """
    expected = golden_pair.get("expected", {})
    candidate_text = _build_full_text(candidate_profile).lower()
    seed_text = _build_full_text(seed_profile).lower()

    # 1. Candidate mechanism keywords
    cand_keywords = expected.get("candidate_mechanism_keywords", [])
    if cand_keywords:
        found = sum(1 for kw in cand_keywords if kw.lower() in candidate_text)
        keyword_score = found / len(cand_keywords)
    else:
        keyword_score = 1.0

    # 2. Candidate target organs
    cand_organs = expected.get("candidate_target_organs", [])
    if cand_organs:
        found = sum(1 for org in cand_organs if org.lower() in candidate_text)
        organ_score = found / len(cand_organs)
    else:
        organ_score = 1.0

    # 3. Analogue reference (does candidate profile mention the parent?)
    analogue_name = expected.get("candidate_should_retrieve_analogue", "")
    analogue_referenced = analogue_name.lower() in candidate_text if analogue_name else True

    # 4. Structural relationship mentioned
    relationship = expected.get("structural_relationship", "")
    rel_keywords = relationship.lower().split() if relationship else []
    rel_mentioned = any(kw in candidate_text for kw in rel_keywords) if rel_keywords else True

    auto_score = (keyword_score + organ_score + (1.0 if analogue_referenced else 0.0) + (1.0 if rel_mentioned else 0.0)) / 4

    return {
        "keyword_score": keyword_score,
        "organ_score": organ_score,
        "analogue_referenced": analogue_referenced,
        "relationship_mentioned": rel_mentioned,
        "auto_score": auto_score,
    }


def _build_full_text(profile: Dict) -> str:
    """Build full text from a profile dict for matching."""
    sections = [
        "toxicity_mechanism", "affected_organs", "symptoms_of_exposure",
        "dose_response", "first_aid", "handling_precautions",
        "regulatory_classification", "related_compounds", "references",
    ]
    parts = []
    for s in sections:
        val = profile.get(s, "")
        if val:
            parts.append(val)
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  Profile Generation via Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_golden_test(
    golden_set_path: str,
    db_dir: str,
    output_dir: str,
    groq_api_key: str = "",
    delay: float = 3.0,
) -> List[Dict]:
    """Generate profiles for all golden molecules and save results.

    Returns:
        List of dicts, each with 'golden_entry', 'profile', 'auto_scores'.
    """
    golden = load_golden_set(golden_set_path)
    molecules = golden.get("molecules", [])

    os.makedirs(output_dir, exist_ok=True)

    # Only generate for non-phase4 entries (phase4 pairs are handled separately)
    standard_mols = [m for m in molecules if m.get("tier") != "phase4_detox_pair"]

    results = []

    # Try to use the RAG pipeline if available
    pipeline = None
    if groq_api_key:
        try:
            from rag_pipeline import RAGPipeline
            pipeline = RAGPipeline(
                vector_store_dir=db_dir,
                groq_api_key=groq_api_key,
                enable_pubchem=True,
            )
            logger.info("RAG pipeline initialized for golden test")
        except Exception as e:
            logger.warning(f"Could not initialize RAG pipeline: {e}")

    for i, entry in enumerate(standard_mols):
        mol_name = entry.get("common_name") or entry.get("iupac_name", "")
        logger.info(f"[{i+1}/{len(standard_mols)}] Processing: {mol_name}")

        profile_dict = {}
        if pipeline:
            try:
                profile = pipeline.generate_safety_profile(
                    iupac_name=entry.get("iupac_name", ""),
                    common_name=entry.get("common_name", ""),
                    cas_number=entry.get("cas_number", ""),
                )
                profile_dict = profile.to_dict()
            except Exception as e:
                logger.warning(f"  Profile generation failed: {e}")
                profile_dict = {"iupac_name": entry.get("iupac_name", ""), "error": str(e)}

            if i < len(standard_mols) - 1:
                time.sleep(delay)

        # Auto-score
        auto_scores = auto_score_against_golden(profile_dict, entry)

        results.append({
            "golden_entry": entry,
            "profile": profile_dict,
            "auto_scores": auto_scores,
        })

        logger.info(
            f"  Auto-score: {auto_scores['auto_score']:.3f} "
            f"(kw={auto_scores['keyword_coverage']:.2f} "
            f"org={auto_scores['organ_coverage']:.2f} "
            f"ld50={auto_scores['ld50_accuracy']:.2f} "
            f"ghs={auto_scores['ghs_match']:.2f})"
        )

    # Save profiles
    profiles_path = os.path.join(output_dir, "golden_profiles.json")
    with open(profiles_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {len(results)} profiles to {profiles_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Annotation Sheet Export
# ═══════════════════════════════════════════════════════════════════════

def generate_annotation_sheet(
    results: List[Dict],
    ragas_scores: Optional[List[Dict]] = None,
    output_path: str = "annotation_sheet.csv",
):
    """Export annotation CSV for expert review.

    Solo annotator workflow: only RAGAS-flagged failures (faith <= 0.70)
    are included for expert review. Other profiles are included but
    marked as 'auto-passed'.

    Args:
        results: List from run_golden_test with profiles and auto_scores.
        ragas_scores: Optional RAGAS scores per profile.
        output_path: CSV output path.
    """
    rows = []

    sections = [
        "toxicity_mechanism", "affected_organs", "symptoms_of_exposure",
        "dose_response", "first_aid", "handling_precautions",
        "regulatory_classification", "related_compounds",
    ]

    for i, result in enumerate(results):
        entry = result["golden_entry"]
        profile = result["profile"]
        auto = result["auto_scores"]
        mol_name = entry.get("common_name") or entry.get("iupac_name", "")

        # Get RAGAS score if available
        faith_score = 0.0
        if ragas_scores and i < len(ragas_scores):
            faith_score = ragas_scores[i].get("faithfulness", 0.0)

        # Determine if expert review needed
        needs_review = faith_score <= EXPERT_REVIEW_THRESHOLD if ragas_scores else True

        for section_key in sections:
            text = profile.get(section_key, "")[:500]  # Truncate for CSV

            rows.append({
                "molecule": mol_name,
                "tier": entry.get("tier", ""),
                "section": section_key.replace("_", " ").title(),
                "generated_text": text,
                "ragas_faithfulness": f"{faith_score:.3f}" if ragas_scores else "N/A",
                "auto_score": f"{auto['auto_score']:.3f}",
                "keyword_coverage": f"{auto['keyword_coverage']:.3f}",
                "needs_expert_review": "YES" if needs_review else "auto-passed",
                "expert_rating_1_5": "",
                "expert_notes": "",
                "failure_type": "",
            })

    # Write CSV
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    review_count = sum(1 for r in rows if r["needs_expert_review"] == "YES")
    logger.info(
        f"Annotation sheet: {len(rows)} rows, "
        f"{review_count} flagged for expert review → {output_path}"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Feedback Report
# ═══════════════════════════════════════════════════════════════════════

def generate_feedback_report(results: List[Dict], output_path: str = None) -> Dict:
    """Analyze golden test results and generate feedback.

    Reports:
        - Which retriever pathway failed most
        - Which sections hallucinate most
        - Phase 4 analogue retrieval success rate
        - Prompt improvements suggested
    """
    tier_scores = {}
    section_scores = {}

    for result in results:
        entry = result["golden_entry"]
        auto = result["auto_scores"]
        tier = entry.get("tier", "unknown")

        tier_scores.setdefault(tier, []).append(auto["auto_score"])

        # Track per-section performance
        for key in ["keyword_coverage", "organ_coverage", "ld50_accuracy", "ghs_match"]:
            section_scores.setdefault(key, []).append(auto.get(key, 0))

    report = {
        "tier_performance": {
            tier: {
                "count": len(scores),
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
            }
            for tier, scores in tier_scores.items()
        },
        "section_performance": {
            key: sum(vals) / len(vals) if vals else 0
            for key, vals in section_scores.items()
        },
        "recommendations": [],
    }

    # Generate recommendations
    sp = report["section_performance"]
    if sp.get("ld50_accuracy", 1) < 0.80:
        report["recommendations"].append(
            "LD50 accuracy below 80%. Consider adding explicit dose-response "
            "extraction in the RAG prompt or augmenting the knowledge base with "
            "standardized dose tables."
        )
    if sp.get("keyword_coverage", 1) < 0.75:
        report["recommendations"].append(
            "Mechanism keyword coverage below 75%. The retriever may be missing "
            "mechanism-of-action documents. Check section weighting in reranker."
        )
    if sp.get("ghs_match", 1) < 0.75:
        report["recommendations"].append(
            "GHS classification match below 75%. Ensure PubChem GHS data is being "
            "indexed and that the prompt includes regulatory classification requirements."
        )

    tp = report["tier_performance"]
    if "semantic_analogue" in tp and tp["semantic_analogue"]["mean_score"] < 0.60:
        report["recommendations"].append(
            "Semantic analogue tier performing poorly. Consider lowering "
            "min_relevance_threshold or adding section_diversity_bonus "
            "for semantic-only retrievals."
        )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Feedback report saved to {output_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════
#  Report Printing
# ═══════════════════════════════════════════════════════════════════════

def print_golden_report(results: List[Dict]):
    """Pretty-print Pillar 3 golden test report."""
    sep = "═" * 66
    thin = "─" * 66

    print(f"\n  {sep}")
    print(f"  ║{'PILLAR 3: GOLDEN TEST SET':^62}║")
    print(f"  {sep}\n")

    if not results:
        print("  No results to report.")
        return

    # Aggregate auto-scores
    n = len(results)
    kw = sum(r["auto_scores"]["keyword_coverage"] for r in results) / n
    org = sum(r["auto_scores"]["organ_coverage"] for r in results) / n
    ld50 = sum(r["auto_scores"]["ld50_accuracy"] for r in results) / n
    ghs = sum(r["auto_scores"]["ghs_match"] for r in results) / n

    print(f"  Auto-Scored ({n} molecules):")
    print(f"  {thin}")

    skw = "✓" if kw >= 0.75 else "⚠"
    sorg = "✓" if org >= 0.80 else "⚠"
    sld = "✓" if ld50 >= 0.80 else "⚠"
    sghs = "✓" if ghs >= 0.75 else "⚠"

    print(f"    Keyword Coverage  : {kw*100:5.1f}%   {skw}  (≥ 75%)")
    print(f"    Organ Coverage    : {org*100:5.1f}%   {sorg}  (≥ 80%)")
    print(f"    LD50 Accuracy     : {ld50*100:5.1f}%   {sld}  (≥ 80%)")
    print(f"    GHS Match         : {ghs*100:5.1f}%   {sghs}  (≥ 75%)")

    # Tier breakdown
    tiers = {}
    for r in results:
        tier = r["golden_entry"].get("tier", "unknown")
        tiers.setdefault(tier, []).append(r["auto_scores"]["auto_score"])

    if len(tiers) > 1:
        print(f"\n  By Tier:")
        print(f"  {thin}")
        for tier, scores in sorted(tiers.items()):
            avg = sum(scores) / len(scores)
            print(f"    {tier:25s}: {avg*100:5.1f}%  ({len(scores)} molecules)")

    print(f"\n  Expert Annotations: Solo annotator — reviewing RAGAS-flagged failures only")
    print(f"\n  {sep}\n")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pillar 3: Human Evaluation & Golden Test Set"
    )
    parser.add_argument("--golden", type=str, required=True,
                        help="Path to golden_test_set.json")
    parser.add_argument("--db-dir", type=str, default="./chroma_db",
                        help="Path to ChromaDB directory")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Directory for output files")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM calls — auto-score only (no profile generation)")
    parser.add_argument("--profiles", type=str, default=None,
                        help="Pre-generated profiles JSON (skip generation)")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Delay between API calls (seconds)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Groq key
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    groq_key = os.getenv("GROQ_API_KEY", "")

    if args.profiles:
        # Load pre-generated profiles
        with open(args.profiles) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} pre-generated profiles")
    elif args.no_llm:
        # Auto-score mode: score ground_truth_summary against golden expectations
        # This validates golden set consistency without generating profiles
        golden = load_golden_set(args.golden)
        results = []
        for entry in golden.get("molecules", []):
            if entry.get("tier") == "phase4_detox_pair":
                continue
            results.append({
                "golden_entry": entry,
                "profile": {},  # No generated profile
                "auto_scores": auto_score_against_golden(
                    {}, entry, use_ground_truth=True
                ),
            })
        logger.info(
            f"Auto-score mode (ground truth proxy): "
            f"{len(results)} molecules (no profiles generated)"
        )
    else:
        # Full generation + scoring
        db_path = os.path.join(PHASE3_DIR, args.db_dir) if not os.path.isabs(args.db_dir) else args.db_dir
        results = run_golden_test(
            golden_set_path=args.golden,
            db_dir=db_path,
            output_dir=args.output_dir,
            groq_api_key=groq_key,
            delay=args.delay,
        )

    # Print report
    print_golden_report(results)

    # Generate feedback report
    feedback = generate_feedback_report(
        results,
        output_path=os.path.join(args.output_dir, "feedback_report.json"),
    )
    if feedback["recommendations"]:
        print("  Recommendations:")
        for rec in feedback["recommendations"]:
            print(f"    → {rec}")

    # Generate annotation sheet
    annotation_path = os.path.join(args.output_dir, "annotation_sheet.csv")
    generate_annotation_sheet(results, output_path=annotation_path)


if __name__ == "__main__":
    main()
