"""CLI entry point for Phase 3 RAG safety profile generation.

Usage:
    # Single molecule (standalone — no Phase 1/2 model needed)
    python run_rag.py "nitrobenzene" --score 0.91

    # With common name for better retrieval
    python run_rag.py "nitrobenzene" --common-name "Nitrobenzene" --score 0.91

    # Batch from CSV
    python run_rag.py --csv ../data/t3db_processed.csv --limit 5

    # Full pipeline with Phase 1 + 2 integration
    python run_rag.py "nitrobenzene" --checkpoint ../iupacGPT/iupac-gpt/checkpoints/iupac
"""

import argparse
import json
import logging
import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Phase3-RAG must be FIRST so its prompts.py isn't shadowed by Phase2-CoT/prompts.py
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(1, PROJECT_ROOT)
# Phase1/Phase2 appended at END — only needed for optional LLM client import
sys.path.append(os.path.join(PROJECT_ROOT, "Phase2-CoT"))
sys.path.append(os.path.join(PROJECT_ROOT, "Phase1-IUPACGPT"))


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: RAG Toxicological Safety Profile Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    parser.add_argument(
        "molecules", nargs="*", default=[],
        help="IUPAC names to analyze"
    )
    parser.add_argument(
        "--common-name", type=str, default=None,
        help="Common name of the molecule (improves T3DB retrieval)"
    )
    parser.add_argument(
        "--score", nargs="*", type=float, default=None,
        help="Pre-computed P(toxic) scores (one per molecule)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="CSV file with 'iupac_name' column for batch analysis"
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Max molecules from CSV (default: 5)"
    )

    # Configuration
    parser.add_argument(
        "--db-dir", type=str,
        default=os.path.join(CURRENT_DIR, "chroma_db"),
        help="ChromaDB directory"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Groq API key"
    )
    parser.add_argument(
        "--model", type=str, default="llama-3.3-70b-versatile",
        help="Groq model name"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM temperature"
    )
    parser.add_argument(
        "--no-pubchem", action="store_true",
        help="Disable PubChem fetching"
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full detailed reports"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between API calls"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.molecules and not args.csv:
        parser.error("Provide molecule names or --csv file")

    # Initialize pipeline
    from rag_pipeline import RAGPipeline

    print(f"\n{'='*60}")
    print(f"  Phase 3: RAG Toxicological Safety Profile")
    print(f"  Model: {args.model}")
    print(f"  Vector Store: {args.db_dir}")
    print(f"{'='*60}\n")

    pipeline = RAGPipeline(
        vector_store_dir=args.db_dir,
        groq_api_key=args.api_key,
        groq_model=args.model,
        enable_pubchem=not args.no_pubchem,
        temperature=args.temperature,
    )

    # Check vector store
    store_count = pipeline.store.count()
    if store_count == 0:
        print("WARNING: Vector store is empty! Run ingest_t3db.py first.")
        print("  python Phase3-RAG/ingest_t3db.py\n")

    # Collect molecules
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv).dropna(subset=["iupac_name"])
        molecules = []
        for i, row in df.iterrows():
            if len(molecules) >= args.limit:
                break
            mol = {"iupac_name": row["iupac_name"]}
            if "common_name" in df.columns:
                mol["common_name"] = str(row.get("common_name", ""))
            if "toxicity_score" in df.columns:
                mol["toxicity_score"] = float(row.get("toxicity_score", 0.5))
            if "smiles" in df.columns:
                mol["smiles"] = str(row.get("smiles", ""))
            molecules.append(mol)
        print(f"Loaded {len(molecules)} molecules from {args.csv}\n")
    else:
        molecules = []
        for i, name in enumerate(args.molecules):
            mol = {"iupac_name": name}
            if args.common_name and i == 0:
                mol["common_name"] = args.common_name
            if args.score and i < len(args.score):
                score = args.score[i]
                mol["toxicity_score"] = score
                mol["is_toxic"] = score >= 0.5
                if score < 0.20:
                    mol["severity_label"] = "Non-toxic"
                elif score < 0.50:
                    mol["severity_label"] = "Unlikely toxic"
                elif score < 0.65:
                    mol["severity_label"] = "Likely toxic"
                elif score < 0.80:
                    mol["severity_label"] = "Moderately toxic"
                else:
                    mol["severity_label"] = "Highly toxic"
            molecules.append(mol)

    # Run analysis
    profiles = []
    for i, mol in enumerate(molecules):
        iupac = mol.get("iupac_name", "?")
        print(f"[{i+1}/{len(molecules)}] Analyzing: {iupac}")

        try:
            profile = pipeline.generate_safety_profile(**mol)
            profiles.append(profile)

            if args.verbose:
                print(profile.detailed_report())
            else:
                print(f"  → {profile.summary()}\n")

        except Exception as e:
            print(f"  → ERROR: {e}\n")
            logging.exception(e)

        if i < len(molecules) - 1:
            time.sleep(args.delay)

    # Save results
    if args.output and profiles:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([p.to_dict() for p in profiles], f, indent=2)
        print(f"Results saved to: {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Generated: {len(profiles)}/{len(molecules)} safety profiles")
    if profiles:
        avg_docs = sum(p.num_retrieved_docs for p in profiles) / len(profiles)
        sections = [
            sum(1 for s in p._get_sections().values() if s)
            for p in profiles
        ]
        avg_sections = sum(sections) / len(sections) if sections else 0
        print(f"  Avg retrieved docs: {avg_docs:.1f}")
        print(f"  Avg sections filled: {avg_sections:.1f}/9")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
