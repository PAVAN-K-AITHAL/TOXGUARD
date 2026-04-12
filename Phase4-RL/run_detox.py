"""CLI entry point for Phase 4: RL-guided molecule detoxification.

Supports three modes:
    1. Single molecule detoxification (with pre-trained policy)
    2. PPO training on a set of toxic molecules
    3. Evaluation of detoxification results

Usage:
    # Single molecule detox (inference mode)
    python Phase4-RL/run_detox.py detox "nitrobenzene" \\
        --checkpoint iupacGPT/iupac-gpt/checkpoints/iupac \\
        --lora Phase1-IUPACGPT/outputs/best_model/lora_adapters.pt \\
        -v

    # PPO training
    python Phase4-RL/run_detox.py train \\
        --checkpoint iupacGPT/iupac-gpt/checkpoints/iupac \\
        --lora Phase1-IUPACGPT/outputs/best_model/lora_adapters.pt \\
        --data data/t3db_processed.csv \\
        --steps 500

    # Evaluate
    python Phase4-RL/run_detox.py eval --results outputs/detox_results.json
"""

import argparse
import json
import logging
import os
import sys

# Load .env file from project root
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# Add project root and phase directories to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1_DIR = os.path.join(PROJECT_ROOT, "Phase1-IUPACGPT")
PHASE4_DIR = os.path.dirname(os.path.abspath(__file__))  # Phase4-RL

for _p in [PROJECT_ROOT, PHASE1_DIR, PHASE4_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Suppress RDKit C++ deprecation warnings (MorganGenerator etc.)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


def setup_logging(verbose: bool = False, log_file: str = "detox.log"):
    """Configure logging: file gets everything, console gets only agent output."""
    import io
    import warnings

    # Suppress Python warnings (deprecations, futures, etc.)
    warnings.filterwarnings("ignore")

    # Suppress additional RDKit warnings
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass

    # File handler: captures ALL logs for debugging
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    # Console handler: only shows important messages (WARNING+)
    # unless verbose mode, then shows agent-level INFO
    console_handler = logging.StreamHandler(
        stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    )
    console_level = logging.INFO if verbose else logging.WARNING
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter("  %(message)s")  # Clean, no timestamps
    )

    # Root logger → file only
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(file_handler)

    # Only show our agent logs on console (not rdkit, urllib3, chromadb, etc.)
    for name in ["detox_agent", "multi_agent", "detox_dossier", "__main__"]:
        agent_logger = logging.getLogger(name)
        agent_logger.addHandler(console_handler)
        agent_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Silence noisy libraries completely (even from file in non-verbose)
    for noisy in ["urllib3", "httpx", "httpcore", "chromadb", "sentence_transformers",
                   "transformers", "torch", "huggingface_hub", "filelock"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_detox(args):
    """Single molecule detoxification."""
    import torch
    from name_resolver import NameResolver
    from molecule_validator import MoleculeValidator
    from reward_function import RewardFunction
    from molecule_generator import MoleculeGenerator
    from detox_agent import DetoxAgent
    from ppo_config import PPOConfig

    # Phase 1 predictor (reward model)
    from iupacGPT_finetune.inference import ToxGuardPredictor
    from iupacGPT_finetune.lora import LoRAConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PPOConfig(device=device)

    print(f"[*] ToxGuard Phase 4: Molecule Detoxification")
    print(f"   Device: {device}")
    print(f"   Molecule: {args.molecule}")
    print()

    # Load Phase 1 predictor (reward model)
    print("Loading Phase 1 predictor (reward model)...")
    predictor = ToxGuardPredictor.from_checkpoint(
        checkpoint_dir=args.checkpoint,
        lora_weights_path=args.lora,
        tokenizer_path=args.tokenizer,
        device=device,
        lora_config=LoRAConfig(r=args.lora_rank),
    )

    # Initialize components
    print("Initializing components...")
    resolver = NameResolver(
        cache_dir=os.path.join(PROJECT_ROOT, "outputs", "name_cache"),
        use_opsin=not args.no_opsin,
    )
    validator = MoleculeValidator()

    reward_fn = RewardFunction(
        toxguard_predictor=predictor,
        name_resolver=resolver,
        molecule_validator=validator,
        config=config,
    )

    # Load molecule generator (IUPAC-GPT policy)
    print("Loading IUPAC-GPT policy network...")
    generator = MoleculeGenerator.from_checkpoint(
        checkpoint_dir=args.checkpoint,
        tokenizer_path=args.tokenizer,
        config=config,
        device=device,
    )

    # Load trained policy if available
    if args.policy_weights and os.path.exists(args.policy_weights):
        print(f"Loading trained policy: {args.policy_weights}")
        generator.load_policy(args.policy_weights)

    # Run detoxification
    print()
    print("=" * 60)
    agent = DetoxAgent(generator, reward_fn, resolver, config)

    report = agent.detoxify(
        seed_iupac=args.molecule,
        seed_smiles=args.smiles,
        seed_p_toxic=args.score,
    )

    # Output
    if args.verbose:
        print(report.detailed_report())
    else:
        print(report.summary())

    # ── Phase 4 → 2/3 Handoff: Generate full dossier ─────────────
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if api_key and report.success:
        print()
        print("=" * 60)
        print("  Generating toxicological dossier (Phase 2 CoT + Phase 3 RAG)...")
        print("=" * 60)
        try:
            from detox_dossier import DossierGenerator

            dossier_gen = DossierGenerator(
                groq_api_key=api_key,
                predictor=predictor,
            )
            dossier = dossier_gen.generate_dossier(report)
            print(dossier.format_report())

            # Save dossier alongside results
            dossier_path = args.output.replace(".json", "_dossier.txt") if args.output else None
            if dossier_path:
                os.makedirs(os.path.dirname(dossier_path) or ".", exist_ok=True)
                with open(dossier_path, "w", encoding="utf-8") as f:
                    f.write(dossier.format_report())
                print(f"Dossier saved to: {dossier_path}")
        except Exception as e:
            print(f"  Dossier generation failed: {e}")
            import traceback
            traceback.print_exc()
    elif api_key and not report.success:
        print("\n  Skipping dossier — no successful detoxification candidate found.")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "seed_iupac": report.seed_iupac,
                "seed_smiles": report.seed_smiles,
                "seed_p_toxic": report.seed_p_toxic,
                "success": report.success,
                "rounds": report.rounds_used,
                "total_generated": report.total_generated,
                "total_valid": report.total_valid,
                "total_less_toxic": report.total_less_toxic,
                "time_s": report.total_time_s,
                "best_candidate": (
                    {
                        "iupac": report.best_candidate.iupac_name,
                        "smiles": report.best_candidate.smiles,
                        "p_toxic": report.best_candidate.p_toxic,
                        "tanimoto": report.best_candidate.tanimoto,
                        "qed": report.best_candidate.qed,
                    }
                    if report.best_candidate else None
                ),
                "strategy_history": report.strategy_history,
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def cmd_train(args):
    """PPO training on toxic molecules."""
    import torch
    import pandas as pd
    from name_resolver import NameResolver
    from molecule_validator import MoleculeValidator
    from reward_function import RewardFunction
    from molecule_generator import MoleculeGenerator
    from rl_trainer import PPOTrainer
    from ppo_config import PPOConfig

    from iupacGPT_finetune.inference import ToxGuardPredictor
    from iupacGPT_finetune.lora import LoRAConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[!] WARNING: PPO training on CPU will be very slow. GPU recommended.")

    config = PPOConfig(
        device=device,
        total_train_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print(f"[*] ToxGuard Phase 4: PPO Training")
    print(f"   Device: {device}")
    print(f"   Steps: {args.steps}")
    print(f"   Batch size: {args.batch_size}")
    print()

    # Load seed molecules
    print("Loading seed molecules...")
    if args.data.endswith(".csv"):
        df = pd.read_csv(args.data)
        seeds = []
        # Expect columns: iupac_name or common_name, smiles, is_toxic/toxicity_label
        name_col = "iupac_name" if "iupac_name" in df.columns else "common_name"
        smiles_col = "smiles" if "smiles" in df.columns else None

        for _, row in df.iterrows():
            name = str(row.get(name_col, "")).strip()
            if not name:
                continue
            smiles = str(row.get(smiles_col, "")).strip() if smiles_col else ""
            seeds.append({
                "iupac": name,
                "smiles": smiles if smiles and smiles != "nan" else "",
                "p_toxic": 0.8,  # Will be re-scored by the reward model
            })
        # Filter to top N toxic seeds
        seeds = seeds[:args.max_seeds]
    elif args.data.endswith(".json"):
        with open(args.data) as f:
            seeds = json.load(f)
    else:
        print(f"Unsupported data format: {args.data}")
        return

    print(f"  Loaded {len(seeds)} seed molecules")

    # Initialize components
    print("Loading models...")
    predictor = ToxGuardPredictor.from_checkpoint(
        checkpoint_dir=args.checkpoint,
        lora_weights_path=args.lora,
        tokenizer_path=args.tokenizer,
        device=device,
        lora_config=LoRAConfig(r=args.lora_rank),
    )

    resolver = NameResolver(
        cache_dir=os.path.join(PROJECT_ROOT, "outputs", "name_cache"),
        use_opsin=not args.no_opsin,
    )
    validator = MoleculeValidator()

    reward_fn = RewardFunction(
        toxguard_predictor=predictor,
        name_resolver=resolver,
        molecule_validator=validator,
        config=config,
    )

    generator = MoleculeGenerator.from_checkpoint(
        checkpoint_dir=args.checkpoint,
        tokenizer_path=args.tokenizer,
        config=config,
        device=device,
    )

    # Score seeds first
    print("Scoring seed molecules with Phase 1...")
    for seed in seeds:
        if not seed.get("smiles"):
            seed["smiles"] = resolver.iupac_to_smiles(seed["iupac"]) or ""
        try:
            pred = predictor.predict(seed["iupac"], return_attention=False,
                                      return_egnn_vector=False)
            seed["p_toxic"] = pred.toxicity_score
        except Exception:
            seed["p_toxic"] = 0.8

    # Filter to actually toxic molecules
    toxic_seeds = [s for s in seeds if s["p_toxic"] >= 0.5]
    print(f"  {len(toxic_seeds)} molecules are toxic (P >= 0.5)")

    if not toxic_seeds:
        print("No toxic molecules to train on!")
        return

    # Train
    output_dir = args.output or os.path.join(PROJECT_ROOT, "outputs", "rl_training")
    trainer = PPOTrainer(
        generator=generator,
        reward_fn=reward_fn,
        config=config,
        output_dir=output_dir,
    )

    trainer.train(
        seed_molecules=toxic_seeds,
        num_steps=args.steps,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    summary = trainer.get_training_summary()
    print(f"\n{'='*60}")
    print(f"Training Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Outputs saved to: {output_dir}")


def cmd_eval(args):
    """Evaluate detoxification results."""
    print("[*] Evaluation mode -- coming soon")
    # TODO: Implement evaluation script


def main():
    parser = argparse.ArgumentParser(
        description="ToxGuard Phase 4: RL-Guided Molecule Detoxification"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── Detox command ─────────────────────────────────────────────────
    detox_parser = subparsers.add_parser("detox", help="Detoxify a single molecule")
    detox_parser.add_argument("molecule", help="IUPAC name of the toxic molecule")
    detox_parser.add_argument("--smiles", help="SMILES of the molecule (optional)")
    detox_parser.add_argument("--score", type=float, help="Known P(toxic) score")
    detox_parser.add_argument(
        "--checkpoint", required=True,
        help="Path to IUPAC-GPT checkpoint directory"
    )
    detox_parser.add_argument("--lora", help="Path to Phase 1 LoRA weights")
    detox_parser.add_argument("--tokenizer", help="Path to tokenizer")
    detox_parser.add_argument("--policy-weights", help="Path to trained PPO policy weights")
    detox_parser.add_argument("--lora-rank", type=int, default=32)
    detox_parser.add_argument("--no-opsin", action="store_true")
    detox_parser.add_argument("-o", "--output", help="Output JSON path")
    detox_parser.add_argument("-v", "--verbose", action="store_true")
    detox_parser.add_argument(
        "--api-key", type=str, default=None,
        help="Groq API key for Phase 2/3 dossier generation (optional)"
    )
    detox_parser.set_defaults(func=cmd_detox)

    # ── Train command ─────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="PPO training")
    train_parser.add_argument(
        "--data", required=True,
        help="Path to CSV/JSON with toxic molecules"
    )
    train_parser.add_argument(
        "--checkpoint", required=True,
        help="Path to IUPAC-GPT checkpoint"
    )
    train_parser.add_argument("--lora", help="Path to Phase 1 LoRA weights")
    train_parser.add_argument("--tokenizer", help="Path to tokenizer")
    train_parser.add_argument("--lora-rank", type=int, default=32)
    train_parser.add_argument("--steps", type=int, default=500)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.add_argument("--max-seeds", type=int, default=100)
    train_parser.add_argument("--save-every", type=int, default=50)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--no-opsin", action="store_true")
    train_parser.add_argument("-o", "--output", help="Output directory")
    train_parser.add_argument("-v", "--verbose", action="store_true")
    train_parser.set_defaults(func=cmd_train)

    # ── Eval command ──────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("eval", help="Evaluate results")
    eval_parser.add_argument("--results", required=True, help="Results JSON")
    eval_parser.add_argument("-v", "--verbose", action="store_true")
    eval_parser.set_defaults(func=cmd_eval)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(verbose=getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
