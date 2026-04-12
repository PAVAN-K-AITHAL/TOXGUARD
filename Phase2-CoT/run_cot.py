"""CLI entry point for Phase 2 CoT analysis.

Usage:
    # Analyze a single molecule (without Phase 1 model — standalone mode)
    python run_cot.py "nitrobenzene" --score 0.91

    # Analyze with Phase 1 model integration
    python run_cot.py "nitrobenzene" --checkpoint iupacGPT/iupac-gpt/checkpoints/iupac

    # Analyze multiple molecules
    python run_cot.py "nitrobenzene" "ethanol" "methanal" --score 0.91 0.28 0.85

    # Batch analysis from CSV
    python run_cot.py --csv data/t3db_processed.csv --limit 10 --output results/cot_results.json

    # Use 8b model for fast testing
    python run_cot.py "nitrobenzene" --score 0.91 --model 8b
"""

import argparse
import json
import logging
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root and current dir to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Phase1-IUPACGPT"))


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Chain-of-Thought Toxicity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    parser.add_argument(
        "molecules", nargs="*", default=[],
        help="IUPAC names to analyze (space-separated, use quotes)"
    )
    parser.add_argument(
        "--score", nargs="*", type=float, default=None,
        help="Pre-computed P(toxic) scores (one per molecule, for standalone mode)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file with 'iupac_name' column for batch analysis"
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Max molecules to analyze from CSV (default: 10)"
    )

    # Model options
    parser.add_argument(
        "--model", type=str, default="70b",
        choices=["70b", "8b", "mixtral"],
        help="Groq model shorthand (default: 70b = llama-3.3-70b-versatile)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Groq API key (or set GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM sampling temperature (default: 0.3)"
    )
    parser.add_argument(
        "--num-exemplars", type=int, default=3,
        help="Number of few-shot examples (default: 3)"
    )

    # Phase 1 integration (optional)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to IUPAC-GPT checkpoint dir (enables Phase 1 integration)"
    )
    parser.add_argument(
        "--lora-weights", type=str, default=None,
        help="Path to trained LoRA weights file"
    )

    # Output options
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full detailed reports"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between API calls for rate limiting (default: 1.0)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate input
    if not args.molecules and not args.csv:
        parser.error("Provide molecule names or --csv file")

    # Initialize LLM client
    from llm_client import GroqLLMClient
    from cot_analyzer import CoTAnalyzer

    print(f"\n{'='*60}")
    print(f"  Phase 2: Chain-of-Thought Toxicity Analysis")
    print(f"  Model: {GroqLLMClient.MODELS.get(args.model, args.model)}")
    print(f"{'='*60}\n")

    llm = GroqLLMClient(
        api_key=args.api_key,
        model=args.model,
    )

    # Initialize analyzer
    analyzer = CoTAnalyzer(
        llm_client=llm,
        checkpoint_dir=args.checkpoint,
        lora_weights_path=args.lora_weights,
        temperature=args.temperature,
        num_exemplars=args.num_exemplars,
    )

    # Collect molecules to analyze
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        if "iupac_name" not in df.columns:
            print(f"ERROR: CSV must have 'iupac_name' column. Found: {list(df.columns)}")
            sys.exit(1)
        df = df.dropna(subset=["iupac_name"])
        molecules = df["iupac_name"].tolist()[:args.limit]
        scores = df["toxicity_score"].tolist()[:args.limit] if "toxicity_score" in df.columns else None
        print(f"Loaded {len(molecules)} molecules from {args.csv}")
    else:
        molecules = args.molecules
        scores = args.score

    # Run analysis
    results = []
    for i, mol in enumerate(molecules):
        score = 0.5  # default
        severity = "Unknown"
        is_toxic = False

        if scores and i < len(scores):
            score = scores[i]
            is_toxic = score >= 0.5
            if score < 0.20:
                severity = "Non-toxic"
            elif score < 0.50:
                severity = "Unlikely toxic"
            elif score < 0.65:
                severity = "Likely toxic"
            elif score < 0.80:
                severity = "Moderately toxic"
            else:
                severity = "Highly toxic"

        try:
            if args.checkpoint and analyzer.predictor:
                result = analyzer.analyze(mol)
            else:
                result = analyzer.analyze_from_prediction(
                    iupac_name=mol,
                    toxicity_score=score,
                    severity_label=severity,
                    is_toxic=is_toxic,
                )
            results.append(result)

            if args.verbose:
                print(result.detailed_report())
            else:
                print(f"  [{i+1}/{len(molecules)}] {result.summary()}")

        except Exception as e:
            print(f"  [{i+1}/{len(molecules)}] ERROR analyzing {mol}: {e}")
            logging.exception(e)

        # Rate limiting
        if i < len(molecules) - 1:
            import time
            time.sleep(args.delay)

    # Save results
    if args.output and results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\n  Results saved to: {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Analyzed: {len(results)}/{len(molecules)} molecules")
    if results:
        toxic_count = sum(1 for r in results if r.is_toxic)
        high_conf = sum(1 for r in results if r.confidence_level == "HIGH")
        print(f"  Toxic: {toxic_count} | Non-toxic: {len(results)-toxic_count}")
        print(f"  High confidence analyses: {high_conf}/{len(results)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
