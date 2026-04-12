"""Unified 4-Pillar RAG Validation Runner.

Orchestrates all evaluation pillars:
    Pillar 1: Retrieval Quality (P@K, R@K, MRR + RAGAS context metrics)
    Pillar 2: LLM Faithfulness (RAGAS faithfulness + domain hard gates)
    Pillar 3: Golden Test Set (auto-scoring + expert annotation export)
    Pillar 4: Phase 4 Analogue Retrieval (structural-analogy tests)

Usage:
    python validate_rag.py --all                     # Full 4-pillar validation
    python validate_rag.py --retrieval               # Pillar 1 only
    python validate_rag.py --faithfulness             # Pillar 2 (RAGAS + domain)
    python validate_rag.py --golden                   # Pillar 3
    python validate_rag.py --phase4                   # Pillar 4
    python validate_rag.py --retrieval --no-ragas     # Pillar 1: IR metrics only
    python validate_rag.py --faithfulness --no-llm    # Pillar 2: domain checks only
    python validate_rag.py --no-llm                   # All pillars, skip ALL LLM calls
"""

import argparse
import json
import logging
import os
import sys
import time

# ── Path setup ─────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Default golden test set location
DEFAULT_GOLDEN = os.path.join(CURRENT_DIR, "golden_test_set.json")


def print_header():
    """Print the validation report header."""
    print()
    print("  " + "═" * 66)
    print("  ║" + "TOXGUARD RAG — VALIDATION REPORT (RAGAS-Powered)".center(64) + "║")
    print("  " + "═" * 66)
    print()


def print_footer(total_time: float):
    """Print the validation report footer."""
    print("  " + "═" * 66)
    print(f"  Total validation time: {total_time:.1f}s")
    print("  " + "═" * 66)
    print()


def run_pillar1(args, store, retriever, groq_key):
    """Run Pillar 1: Retrieval Quality Metrics."""
    from eval_retrieval import (
        build_relevance_judgments,
        evaluate_batch,
        analyze_failure_modes,
        print_retrieval_report,
    )

    logger.info("━━━ PILLAR 1: Retrieval Quality ━━━")

    relevance_map = build_relevance_judgments(store)

    # Select molecules
    golden_path = args.golden if args.golden and os.path.exists(args.golden) else DEFAULT_GOLDEN
    if os.path.exists(golden_path):
        with open(golden_path) as f:
            golden = json.load(f)
        molecules = [
            m.get("common_name") or m.get("iupac_name", "")
            for m in golden.get("molecules", [])
            if m.get("tier") != "phase4_detox_pair"
        ]
    else:
        molecules = list(relevance_map.keys())

    molecules = molecules[:args.limit]
    ir_results = evaluate_batch(retriever, molecules, relevance_map, K=args.k)

    # RAGAS context metrics (optional)
    ragas_results = None
    if not args.no_ragas and not args.no_llm and groq_key:
        try:
            import asyncio
            from eval_retrieval import compute_ragas_context_metrics

            ragas_samples = []
            if os.path.exists(golden_path):
                with open(golden_path) as f:
                    golden = json.load(f)
                for entry in golden.get("molecules", [])[:10]:
                    name = entry.get("common_name") or entry.get("iupac_name", "")
                    ref = entry.get("expected", {}).get("ground_truth_summary", "")
                    if name and ref:
                        results = retriever.retrieve(query_name=name, fetch_pubchem=False)
                        ragas_samples.append({
                            "user_input": f"Generate toxicological safety profile for {name}",
                            "retrieved_contexts": [r.content for r in results[:args.k]],
                            "response": ref,
                            "reference": ref,
                        })

            if ragas_samples:
                ragas_results = asyncio.run(
                    compute_ragas_context_metrics(ragas_samples, groq_key)
                )
        except Exception as e:
            logger.warning(f"RAGAS context metrics failed: {e}")

    print_retrieval_report(ir_results, ragas_results)
    return {"ir_metrics": ir_results, "ragas_context": ragas_results}


def run_pillar2(args, groq_key):
    """Run Pillar 2: LLM Faithfulness."""
    from eval_faithfulness import evaluate_faithfulness, print_faithfulness_report

    logger.info("━━━ PILLAR 2: LLM Faithfulness ━━━")

    # Try to load pre-generated profiles
    profiles_path = None
    if args.profiles:
        profiles_path = args.profiles
    else:
        # Check if golden test produced profiles
        candidate_path = os.path.join(
            args.output_dir, "golden_profiles.json"
        )
        if os.path.exists(candidate_path):
            profiles_path = candidate_path

    profiles = []
    if profiles_path and os.path.exists(profiles_path):
        with open(profiles_path) as f:
            data = json.load(f)
        # Handle both raw profiles and golden-test-results format
        if isinstance(data, list) and data:
            if "profile" in data[0]:
                profiles = [r["profile"] for r in data]
            else:
                profiles = data
        logger.info(f"Loaded {len(profiles)} profiles from {profiles_path}")
    else:
        logger.info("No profiles available — running domain checks on empty profiles")
        golden_path = args.golden if args.golden and os.path.exists(args.golden) else DEFAULT_GOLDEN
        if os.path.exists(golden_path):
            with open(golden_path) as f:
                golden = json.load(f)
            profiles = [
                {"iupac_name": m.get("iupac_name", ""), "common_name": m.get("common_name", "")}
                for m in golden.get("molecules", [])
                if m.get("tier") != "phase4_detox_pair"
            ]

    run_ragas = not args.no_llm and bool(groq_key)

    results = evaluate_faithfulness(
        profiles=profiles,
        run_ragas=run_ragas,
        groq_api_key=groq_key,
    )

    print_faithfulness_report(results)
    return results


def run_pillar3(args, groq_key):
    """Run Pillar 3: Golden Test Set."""
    from eval_human import (
        load_golden_set,
        run_golden_test,
        auto_score_against_golden,
        print_golden_report,
        generate_annotation_sheet,
        generate_feedback_report,
    )

    logger.info("━━━ PILLAR 3: Golden Test Set ━━━")

    golden_path = args.golden if args.golden and os.path.exists(args.golden) else DEFAULT_GOLDEN

    if args.no_llm:
        # Auto-score using ground_truth_summary as proxy (validates golden set)
        golden = load_golden_set(golden_path)
        results = []
        for entry in golden.get("molecules", []):
            if entry.get("tier") == "phase4_detox_pair":
                continue
            results.append({
                "golden_entry": entry,
                "profile": {},
                "auto_scores": auto_score_against_golden(
                    {}, entry, use_ground_truth=True
                ),
            })
    else:
        db_path = os.path.join(CURRENT_DIR, args.db_dir) if not os.path.isabs(args.db_dir) else args.db_dir
        results = run_golden_test(
            golden_set_path=golden_path,
            db_dir=db_path,
            output_dir=args.output_dir,
            groq_api_key=groq_key,
            delay=args.delay,
        )

    print_golden_report(results)

    # Generate annotation sheet
    annotation_path = os.path.join(args.output_dir, "annotation_sheet.csv")
    generate_annotation_sheet(results, output_path=annotation_path)

    # Generate feedback report
    generate_feedback_report(
        results,
        output_path=os.path.join(args.output_dir, "feedback_report.json"),
    )

    return results


def run_pillar4(args, store, retriever):
    """Run Pillar 4: Phase 4 Analogue Retrieval."""
    from eval_phase4_retrieval import evaluate_phase4

    logger.info("━━━ PILLAR 4: Phase 4 Analogue Retrieval ━━━")

    golden_path = args.golden if args.golden and os.path.exists(args.golden) else DEFAULT_GOLDEN

    results = evaluate_phase4(retriever, golden_path, K=args.k)
    return results


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ToxGuard RAG — Unified 4-Pillar Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python validate_rag.py --all                     Full 4-pillar validation
    python validate_rag.py --retrieval               Pillar 1 only
    python validate_rag.py --faithfulness             Pillar 2 (RAGAS + domain)
    python validate_rag.py --golden                   Pillar 3
    python validate_rag.py --phase4                   Pillar 4
    python validate_rag.py --retrieval --no-ragas     IR metrics only, no LLM
    python validate_rag.py --faithfulness --no-llm    Domain checks only, no RAGAS
    python validate_rag.py --no-llm                   All pillars, skip ALL LLM calls
        """,
    )

    # Pillar selection
    parser.add_argument("--all", action="store_true",
                        help="Run all 4 pillars")
    parser.add_argument("--retrieval", action="store_true",
                        help="Pillar 1: Retrieval Quality")
    parser.add_argument("--faithfulness", action="store_true",
                        help="Pillar 2: LLM Faithfulness")
    parser.add_argument("--golden", action="store_true",
                        help="Pillar 3: Golden Test Set")
    parser.add_argument("--phase4", action="store_true",
                        help="Pillar 4: Phase 4 Analogue Retrieval")

    # LLM control
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip RAGAS LLM calls in Pillar 1")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip ALL LLM calls across all pillars (CI/CD mode)")

    # Paths
    parser.add_argument("--db-dir", type=str, default="./chroma_db",
                        help="Path to ChromaDB directory")
    parser.add_argument("--golden-set", type=str, default=None,
                        dest="golden_set_path",
                        help="Path to golden_test_set.json")
    parser.add_argument("--profiles", type=str, default=None,
                        help="Pre-generated profiles JSON for Pillar 2")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Output directory for results")

    # Parameters
    parser.add_argument("--k", type=int, default=12, help="Top-K for metrics")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max molecules to evaluate")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Delay between API calls (seconds)")

    args = parser.parse_args()

    # Resolve golden path — the argparse dest 'golden_set_path' should
    # be accessible but we also use 'golden' as a pillar flag, so copy:
    if args.golden_set_path:
        args.golden = args.golden_set_path
    elif not hasattr(args, 'golden') or isinstance(args.golden, bool):
        # --golden was used as a flag, not a path
        run_golden_pillar = args.golden if isinstance(args.golden, bool) else False
        args.golden = DEFAULT_GOLDEN
        # Restore the flag
        if run_golden_pillar:
            args._run_golden = True
        else:
            args._run_golden = False

    # If no pillar selected, show help
    run_any = args.all or args.retrieval or args.faithfulness or args.phase4
    run_golden = args.all or getattr(args, '_run_golden', False)

    # Handle case where --golden is the pillar flag
    if isinstance(args.golden, bool):
        run_golden = args.golden or args.all
        args.golden = DEFAULT_GOLDEN
    else:
        run_golden = args.all or run_golden

    if not run_any and not run_golden:
        parser.print_help()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load API key
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    groq_key = os.getenv("GROQ_API_KEY", "")

    if args.no_llm:
        logger.info("═══ CI/CD MODE: All LLM calls disabled ═══")
    elif not groq_key:
        logger.warning("GROQ_API_KEY not found — LLM-dependent features will be skipped")

    # Initialize store and retriever (shared)
    from vector_store import ToxVectorStore
    from retriever import HybridRetriever

    db_path = os.path.join(CURRENT_DIR, args.db_dir) if not os.path.isabs(args.db_dir) else args.db_dir
    store = ToxVectorStore(persist_dir=db_path)
    retriever = HybridRetriever(vector_store=store)

    t0 = time.time()
    print_header()

    all_results = {}

    # Pillar 1
    if args.all or args.retrieval:
        try:
            all_results["pillar1"] = run_pillar1(args, store, retriever, groq_key)
        except Exception as e:
            logger.error(f"Pillar 1 failed: {e}")
            all_results["pillar1"] = {"error": str(e)}

    # Pillar 2
    if args.all or args.faithfulness:
        try:
            all_results["pillar2"] = run_pillar2(args, groq_key)
        except Exception as e:
            logger.error(f"Pillar 2 failed: {e}")
            all_results["pillar2"] = {"error": str(e)}

    # Pillar 3
    if run_golden:
        try:
            all_results["pillar3"] = run_pillar3(args, groq_key)
        except Exception as e:
            logger.error(f"Pillar 3 failed: {e}")
            all_results["pillar3"] = {"error": str(e)}

    # Pillar 4
    if args.all or args.phase4:
        try:
            all_results["pillar4"] = run_pillar4(args, store, retriever)
        except Exception as e:
            logger.error(f"Pillar 4 failed: {e}")
            all_results["pillar4"] = {"error": str(e)}

    total_time = time.time() - t0
    print_footer(total_time)

    # Save combined results
    output_path = os.path.join(args.output_dir, "validation_results.json")
    try:
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"All results saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save results: {e}")


if __name__ == "__main__":
    main()
