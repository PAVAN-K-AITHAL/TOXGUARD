#!/usr/bin/env python3
"""
==============================================================================
STEP 5b — Train ToxGuard (5-Fold Scaffold-Split Cross-Validation)
==============================================================================

Publication-standard evaluation of the Phase1-IUPACGPT architecture using
5-fold Bemis-Murcko scaffold-split cross-validation.

Uses **mean pooling** by default (fixes the Oxidane bug from last-token pooling).

Evaluation:
    - 5-fold scaffold-split cross-validation (identical to baselines)
    - Per-dataset breakdown (ToxCast, Tox21, hERG, DILI, Common Molecules, T3DB)
    - Reports: AUROC, AUPRC, F1, MCC, Accuracy ± std across folds

Input:
    - data/*_final.csv  (training: ToxCast, Tox21, hERG, DILI, Common Molecules)
    - data/t3db_processed.csv  (external validation)
    - iupacGPT/iupac-gpt/checkpoints/iupac/  (IUPACGPT backbone)

Output:
    - Phase1-IUPACGPT/iupacGPT_outputs/5fold_cv/iupacgpt_5fold_results.json
    - Phase1-IUPACGPT/iupacGPT_outputs/5fold_cv/fold_<N>/lora_weights.pt

Usage:
    python Phase1-IUPACGPT/steps/step5b_train_5fold_cv.py
    python Phase1-IUPACGPT/steps/step5b_train_5fold_cv.py --pooling_strategy mean
    python Phase1-IUPACGPT/steps/step5b_train_5fold_cv.py --resume

Author: ToxGuard Team
==============================================================================
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix,
)

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Add project root and Phase1 to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase1-IUPACGPT"))

from iupacGPT_finetune.model import ToxGuardModel, ToxGuardLitModel
from iupacGPT_finetune.tokenizer import get_tokenizer
from iupacGPT_finetune.lora import apply_lora_to_model, LoRAConfig, save_lora_weights
from iupacGPT_finetune.data_pipeline import (
    MoleculeDataset, T3DBDataset, ToxicityDataset, ToxicityCollator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
CHECKPOINT_DIR = os.path.join("iupacGPT", "iupac-gpt", "checkpoints", "iupac")
SPM_PATH = os.path.join("iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model")
DATA_DIR = "./data"
OUTPUT_DIR = os.path.join("Phase1-IUPACGPT", "iupacGPT_outputs", "5fold_cv")

# Training dataset specs
TRAINING_DATASETS = {
    "toxcast":          "toxcast_final.csv",
    "tox21":            "tox21_final.csv",
    "herg":             "herg_final.csv",
    "dili":             "dili_final.csv",
    "common_molecules": "common_molecules_final.csv",
}

EVAL_DATASETS = {
    "t3db": ("t3db_processed.csv", True),  # (filename, all_toxic)
}


# ---------------------------------------------------------------------------
# Scaffold k-fold splitting
# ---------------------------------------------------------------------------
def get_scaffold(smi):
    """Return generic Bemis-Murcko scaffold SMILES."""
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            return "_INVALID_"
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return "_ERROR_"


def scaffold_kfold(smiles_list, labels, n_folds=5):
    """5-fold scaffold-based cross-validation split.

    Identical algorithm to the baselines (RF, SchNet, ChemBERTa)
    for fair, apples-to-apples comparison.
    """
    scaffold_groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold_groups[get_scaffold(smi)].append(i)

    sorted_scaffolds = sorted(
        scaffold_groups.items(),
        key=lambda kv: (-len(kv[1]), kv[0])
    )

    fold_indices = [[] for _ in range(n_folds)]
    for i, (_scaffold, indices) in enumerate(sorted_scaffolds):
        fold_indices[i % n_folds].extend(indices)

    logger.info(f"  {len(scaffold_groups)} unique scaffolds → {n_folds} folds")
    for i, fold_idx in enumerate(fold_indices):
        n_toxic = int(labels[fold_idx].sum()) if len(fold_idx) > 0 else 0
        logger.info(f"    Fold {i+1}: {len(fold_idx)} molecules "
                     f"({n_toxic} toxic, {len(fold_idx)-n_toxic} non-toxic)")

    splits = []
    for fold in range(n_folds):
        test_idx = np.array(fold_indices[fold])
        train_idx = np.concatenate([np.array(fold_indices[j])
                                     for j in range(n_folds) if j != fold])
        splits.append((train_idx, test_idx))

    return splits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute standard classification metrics — same as baselines."""
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)

    metrics = {}
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = 0.5
    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auprc"] = 0.0

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    metrics["sensitivity"] = float(tp / max(tp + fn, 1))
    return metrics


# ---------------------------------------------------------------------------
# Load datasets with source tags
# ---------------------------------------------------------------------------
def load_all_training_data(data_dir, tokenizer, max_length):
    """Load all training datasets, keeping track of per-dataset source indices."""
    datasets_list = []
    source_tags = []  # parallel array: dataset name for each molecule

    for ds_name, csv_name in TRAINING_DATASETS.items():
        csv_path = os.path.join(data_dir, csv_name)
        if not os.path.exists(csv_path):
            logger.warning(f"  {csv_name} not found, skipping.")
            continue

        try:
            ds = MoleculeDataset(
                csv_path, tokenizer, max_length,
                dataset_name=ds_name
            )
            datasets_list.append(ds)
            source_tags.extend([ds_name] * len(ds))
            logger.info(f"  {ds_name}: {len(ds)} molecules")
        except Exception as e:
            logger.warning(f"  Could not load {ds_name}: {e}")

    if not datasets_list:
        raise RuntimeError("No datasets loaded! Run steps 1-3 first.")

    combined = ToxicityDataset(datasets_list)
    source_tags = np.array(source_tags)

    return combined, source_tags


def load_eval_data(data_dir, tokenizer, max_length):
    """Load external validation datasets (T3DB)."""
    eval_data = {}
    for ds_name, (csv_name, all_toxic) in EVAL_DATASETS.items():
        csv_path = os.path.join(data_dir, csv_name)
        if not os.path.exists(csv_path):
            logger.warning(f"  {csv_name} not found, skipping.")
            continue
        try:
            if all_toxic:
                ds = T3DBDataset(csv_path, tokenizer, max_length, dataset_name=ds_name)
            else:
                ds = MoleculeDataset(csv_path, tokenizer, max_length, dataset_name=ds_name)
            eval_data[ds_name] = ds
            logger.info(f"  External: {ds_name} = {len(ds)} molecules")
        except Exception as e:
            logger.warning(f"  Could not load {ds_name}: {e}")

    return eval_data


# ---------------------------------------------------------------------------
# Evaluate a model on a DataLoader
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, loader, device):
    """Run inference, return (labels, probabilities)."""
    model.eval()
    all_labels, all_probs = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["binary_labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = outputs.toxicity_score.cpu().numpy()  # sigmoid already applied
        labels_np = labels.numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())

    return np.array(all_labels), np.array(all_probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="STEP 5b: ToxGuard IUPACGPT 5-Fold Scaffold CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR)
    parser.add_argument("--tokenizer", default=SPM_PATH)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)

    # Data
    parser.add_argument("--max_length", type=int, default=300)

    # Pooling
    parser.add_argument("--pooling_strategy", default="mean",
                        choices=["mean", "cls", "last_token"],
                        help="Token pooling strategy for molecular representation")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=64.0)
    parser.add_argument("--lora_dropout", type=float, default=0.2)
    parser.add_argument("--lora_targets", default="c_attn,c_proj,c_fc")

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1200)
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accumulation", type=int, default=2)
    parser.add_argument("--precision", default="16")

    # Loss
    parser.add_argument("--use_focal_loss", action="store_true", default=True)
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--focal_alpha", type=float, default=0.45)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Early stopping
    parser.add_argument("--es_patience", type=int, default=20)
    parser.add_argument("--es_min_delta", type=float, default=1e-3)

    # CV
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Hardware
    parser.add_argument("--num_workers", type=int, default=4)

    # Resume
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from last completed fold")

    args = parser.parse_args()

    # Set working directory to project root
    os.chdir(PROJECT_ROOT)

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    resume_state_path = out_dir / "cv_resume_state.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 65)
    logger.info("STEP 5b: ToxGuard IUPACGPT — 5-Fold Scaffold CV")
    logger.info("=" * 65)
    logger.info(f"  Pooling:    {args.pooling_strategy}")
    logger.info(f"  LoRA:       rank={args.lora_rank}, alpha={args.lora_alpha}")
    logger.info(f"  Training:   lr={args.learning_rate}, focal(γ={args.focal_gamma}, α={args.focal_alpha})")
    logger.info(f"  CV:         {args.n_folds}-fold scaffold split")
    logger.info(f"  Device:     {device}")

    start_time = time.time()
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # ── Tokenizer ──
    logger.info("\nLoading tokenizer...")
    tokenizer = get_tokenizer(vocab_path=args.tokenizer)
    pad_id = tokenizer.pad_token_id or getattr(tokenizer, "eos_token_id", 0) or 0
    collator = ToxicityCollator(pad_id)

    # ── Load data ──
    logger.info("\nLoading training datasets...")
    combined, source_tags = load_all_training_data(args.data_dir, tokenizer, args.max_length)
    labels = combined.binary_labels.astype(float)
    smiles = combined.smiles
    total = len(combined)

    logger.info(f"\n  Combined: {total} molecules "
                f"({int(labels.sum())} toxic, {int((labels == 0).sum())} non-toxic)")

    # Load external validation
    logger.info("\nLoading external validation...")
    eval_data = load_eval_data(args.data_dir, tokenizer, args.max_length)

    # ── Scaffold k-fold ──
    logger.info(f"\nPerforming {args.n_folds}-fold scaffold CV...")
    splits = scaffold_kfold(smiles, labels, n_folds=args.n_folds)

    # ── Resume state ──
    fold_results = []
    all_per_dataset = {ds: [] for ds in list(TRAINING_DATASETS.keys()) + list(EVAL_DATASETS.keys())}

    if args.resume and resume_state_path.exists():
        with open(resume_state_path, "r") as f:
            resume_state = json.load(f)
        fold_results = resume_state.get("fold_results", [])
        loaded_pd = resume_state.get("all_per_dataset", {})
        for ds_name in all_per_dataset:
            all_per_dataset[ds_name] = loaded_pd.get(ds_name, [])
        logger.info(f"Resumed: {len(fold_results)} folds already completed")

    # ── Training loop ──
    _persistent = args.num_workers > 0
    _pin_memory = device.type == "cuda"

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        fold_number = fold_i + 1
        if fold_number <= len(fold_results):
            logger.info(f"\nSkipping fold {fold_number}/{args.n_folds} (already completed)")
            continue

        logger.info(f"\n{'─'*65}")
        logger.info(f"FOLD {fold_number}/{args.n_folds}")

        fold_dir = out_dir / f"fold_{fold_number}"
        os.makedirs(fold_dir, exist_ok=True)

        # Split train → train/val (90/10)
        np.random.seed(args.seed + fold_i)
        n_val = max(1, len(train_idx) // 10)
        shuffled_train = train_idx.copy()
        np.random.shuffle(shuffled_train)
        val_idx = shuffled_train[:n_val]
        actual_train_idx = shuffled_train[n_val:]

        train_ds = Subset(combined, actual_train_idx.tolist())
        val_ds = Subset(combined, val_idx.tolist())
        test_ds = Subset(combined, test_idx.tolist())

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collator, num_workers=args.num_workers,
            persistent_workers=_persistent, pin_memory=_pin_memory,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=args.num_workers,
            persistent_workers=_persistent, pin_memory=_pin_memory,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=args.num_workers,
            persistent_workers=_persistent, pin_memory=_pin_memory,
        )

        logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

        # ── Build fresh model ──
        logger.info(f"  Loading IUPACGPT backbone (pooling={args.pooling_strategy})...")
        model = ToxGuardModel.from_pretrained_iupacgpt(
            args.checkpoint,
            pooling_strategy=args.pooling_strategy,
        )
        model.config.pad_token_id = pad_id

        # Loss configuration
        model.label_smoothing = args.label_smoothing
        if args.use_focal_loss:
            model.use_focal_loss = True
            model.focal_gamma = args.focal_gamma
            model.focal_alpha = args.focal_alpha

        # LoRA
        lora_config = LoRAConfig(
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_targets.split(","),
            fan_in_fan_out=True,
        )
        model, lora_stats = apply_lora_to_model(model, lora_config)

        # Lightning module
        max_steps = len(train_loader) * args.max_epochs
        lit_model = ToxGuardLitModel(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=max_steps,
            scheduler_type=args.scheduler,
        )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_auroc",
                min_delta=args.es_min_delta,
                patience=args.es_patience,
                mode="max",
                verbose=True,
            ),
        ]

        # Trainer
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=1,
            callbacks=callbacks,
            logger=False,
            gradient_clip_val=args.grad_clip,
            accumulate_grad_batches=args.grad_accumulation,
            precision=args.precision,
            log_every_n_steps=10,
            enable_checkpointing=False,
            enable_model_summary=False,
            deterministic=False,
        )

        # Train
        trainer.fit(lit_model, train_loader, val_loader)

        # Save LoRA weights for this fold
        lora_path = str(fold_dir / "lora_weights.pt")
        save_lora_weights(model, lora_path)

        # ── Evaluate on test fold (overall) ──
        model.to(device)
        test_labels, test_probs = evaluate_model(model, test_loader, device)
        fold_metrics = compute_metrics(test_labels, test_probs)
        fold_results.append(fold_metrics)

        logger.info(f"  Test — AUROC: {fold_metrics['auroc']:.4f} | "
                     f"AUPRC: {fold_metrics['auprc']:.4f} | "
                     f"F1: {fold_metrics['f1']:.4f} | "
                     f"MCC: {fold_metrics['mcc']:.4f}")

        # ── Per-dataset breakdown on test fold ──
        sources_test = source_tags[test_idx]
        for ds_name in TRAINING_DATASETS.keys():
            ds_mask = sources_test == ds_name
            if ds_mask.sum() >= 5:
                ds_indices = test_idx[np.where(ds_mask)[0]]
                ds_subset = Subset(combined, ds_indices.tolist())
                ds_loader = DataLoader(
                    ds_subset, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collator, num_workers=0,
                )
                ds_labels, ds_probs = evaluate_model(model, ds_loader, device)
                ds_metrics = compute_metrics(ds_labels, ds_probs)
                all_per_dataset[ds_name].append(ds_metrics)
                logger.info(f"    {ds_name}: AUROC={ds_metrics['auroc']:.4f} "
                             f"n={int(ds_mask.sum())}")

        # ── External validation (T3DB) ──
        for ds_name, ev_ds in eval_data.items():
            ev_loader = DataLoader(
                ev_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=collator, num_workers=0,
            )
            ev_labels, ev_probs = evaluate_model(model, ev_loader, device)
            ev_metrics = compute_metrics(ev_labels, ev_probs)
            all_per_dataset[ds_name].append(ev_metrics)
            logger.info(f"    {ds_name} (ext): AUROC={ev_metrics['auroc']:.4f}")

        # Save resume state
        resume_data = {
            "fold_results": fold_results,
            "all_per_dataset": all_per_dataset,
            "n_folds": args.n_folds,
        }
        with open(resume_state_path, "w") as f:
            json.dump(resume_data, f, indent=2)
        logger.info(f"  Saved resume state: {resume_state_path}")

        # Free memory
        del model, lit_model, trainer
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Aggregate ──
    logger.info(f"\n{'='*65}")
    logger.info("AGGREGATED 5-FOLD RESULTS")
    logger.info("=" * 65)

    if len(fold_results) < args.n_folds:
        logger.warning(
            f"Only {len(fold_results)}/{args.n_folds} folds completed. "
            "Re-run with --resume to finish."
        )

    metric_names = ["auroc", "auprc", "accuracy", "f1", "mcc",
                     "precision", "recall", "specificity", "sensitivity"]

    overall = {}
    for m in metric_names:
        values = [fold[m] for fold in fold_results]
        overall[m] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_fold": [float(v) for v in values],
        }
        logger.info(f"  {m:15s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    per_dataset_summary = {}
    logger.info(f"\n{'─'*65}")
    logger.info("PER-DATASET RESULTS")
    logger.info("─" * 65)

    for ds_name, fold_list in all_per_dataset.items():
        if not fold_list:
            continue
        ds_summary = {}
        for m in metric_names:
            values = [fold[m] for fold in fold_list]
            ds_summary[m] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        per_dataset_summary[ds_name] = ds_summary
        logger.info(f"  {ds_name:20s}: AUROC={ds_summary['auroc']['mean']:.4f}±"
                     f"{ds_summary['auroc']['std']:.4f}  "
                     f"F1={ds_summary['f1']['mean']:.4f}  "
                     f"({len(fold_list)} folds)")

    elapsed = time.time() - start_time

    # ── Save results ──
    results = {
        "model": "ToxGuard IUPACGPT (LoRA + Mean Pooling)",
        "config": {
            "architecture": "GPT-2 (IUPACGPT) + LoRA + Binary Head",
            "pooling_strategy": args.pooling_strategy,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
            "label_smoothing": args.label_smoothing,
            "optimizer": "AdamW",
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "warmup_steps": args.warmup_steps,
            "loss": f"FocalLoss(alpha={args.focal_alpha}, gamma={args.focal_gamma})"
                    if args.use_focal_loss else "BCE",
            "max_epochs": args.max_epochs,
            "patience": args.es_patience,
            "batch_size": args.batch_size,
            "grad_accumulation": args.grad_accumulation,
            "effective_batch_size": args.batch_size * args.grad_accumulation,
            "n_folds": args.n_folds,
            "split_method": "bemis_murcko_scaffold",
            "seed": args.seed,
        },
        "overall_5fold": overall,
        "per_dataset": per_dataset_summary,
        "per_fold_raw": [
            {k: float(v) if isinstance(v, (int, float, np.floating)) else v
             for k, v in fold.items()}
            for fold in fold_results
        ],
        "n_molecules": total,
        "n_toxic": int(labels.sum()),
        "n_nontoxic": int((labels == 0).sum()),
        "completed_folds": len(fold_results),
        "elapsed_seconds": elapsed,
    }

    results_path = out_dir / "iupacgpt_5fold_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*65}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"{'='*65}")

    # Clean resume state on completion
    if len(fold_results) >= args.n_folds and resume_state_path.exists():
        os.remove(resume_state_path)
        logger.info(f"Removed resume state after successful completion")


if __name__ == "__main__":
    main()
