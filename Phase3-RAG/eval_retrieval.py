"""Pillar 1 — Retrieval Quality Metrics for ToxGuard RAG.

Two evaluation layers:
    Layer A — Custom IR Metrics (offline, no LLM):
        Precision@K, Recall@K, MRR against T3DB ground truth.
    Layer B — RAGAS Context Metrics (LLM-judged):
        context_precision, context_recall via Groq LLM judge.

Usage:
    python eval_retrieval.py --db-dir ./chroma_db --k 12
    python eval_retrieval.py --db-dir ./chroma_db --ragas
    python eval_retrieval.py --db-dir ./chroma_db --phase4 --golden golden_test_set.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# ── Path setup ─────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from vector_store import ToxVectorStore
from retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Layer A — Custom IR Metrics (no LLM)
# ═══════════════════════════════════════════════════════════════════════

def build_relevance_judgments(
    vector_store: ToxVectorStore,
) -> Dict[str, Set[str]]:
    """Build relevance mappings from the vector store metadata.

    For each molecule in the store, collect all doc_ids that belong to it.
    This serves as the ground-truth relevant set for IR metrics.

    Returns:
        Dict mapping molecule_name (lowercased) -> Set of doc_ids.
    """
    store = vector_store
    store._ensure_initialized()

    # Get all documents via peek (ChromaDB limit)
    total = store.count()
    if total == 0:
        logger.warning("Vector store is empty — no relevance judgments")
        return {}

    relevance_map: Dict[str, Set[str]] = defaultdict(set)

    # ChromaDB peek is limited; use get with offset for larger stores
    batch_size = 1000
    offset = 0

    while offset < total:
        try:
            results = store.collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )
        except Exception:
            # Fallback: older ChromaDB may not support offset
            results = store.collection.peek(limit=min(total, 10000))
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results.get("metadatas") else {}
                    mol_name = meta.get("molecule_name", "").lower().strip()
                    if mol_name:
                        relevance_map[mol_name].add(doc_id)
            break

        if not results or not results["ids"]:
            break

        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            mol_name = meta.get("molecule_name", "").lower().strip()
            if mol_name:
                relevance_map[mol_name].add(doc_id)

        offset += len(results["ids"])
        if len(results["ids"]) < batch_size:
            break

    logger.info(
        f"Built relevance judgments for {len(relevance_map)} molecules "
        f"({sum(len(v) for v in relevance_map.values())} total doc-molecule pairs)"
    )
    return dict(relevance_map)


def compute_ir_metrics(
    retriever: HybridRetriever,
    mol_name: str,
    relevant_ids: Set[str],
    K: int = 12,
) -> Dict[str, float]:
    """Compute precision@K, recall@K, and MRR for a single molecule.

    Args:
        retriever: The HybridRetriever instance.
        mol_name: Molecule name to query.
        relevant_ids: Set of doc_ids that are relevant for this molecule.
        K: Number of top results to consider.

    Returns:
        Dict with precision_at_k, recall_at_k, mrr.
    """
    results = retriever.retrieve(
        query_name=mol_name,
        fetch_pubchem=False,  # Only evaluate against existing KB
    )

    top_k_ids = [r.doc_id for r in results[:K]]
    retrieved_set = set(top_k_ids)

    # Precision@K
    relevant_in_topk = retrieved_set & relevant_ids
    precision = len(relevant_in_topk) / K if K > 0 else 0.0

    # Recall@K
    recall = len(relevant_in_topk) / len(relevant_ids) if relevant_ids else 0.0

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, doc_id in enumerate(top_k_ids, 1):
        if doc_id in relevant_ids:
            mrr = 1.0 / rank
            break

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
        "num_relevant": len(relevant_ids),
        "num_retrieved": len(results),
        "num_relevant_in_topk": len(relevant_in_topk),
    }


def evaluate_batch(
    retriever: HybridRetriever,
    molecules: List[str],
    relevance_map: Dict[str, Set[str]],
    K: int = 12,
) -> Dict:
    """Evaluate IR metrics across a batch of molecules.

    Args:
        retriever: HybridRetriever instance.
        molecules: List of molecule names to evaluate.
        relevance_map: Ground-truth relevance mapping.
        K: Top-K for metrics.

    Returns:
        Dict with per-molecule and aggregate metrics.
    """
    results = {}
    agg = {"precision_at_k": [], "recall_at_k": [], "mrr": []}

    for mol in molecules:
        mol_lower = mol.lower().strip()
        relevant_ids = relevance_map.get(mol_lower, set())

        if not relevant_ids:
            logger.warning(f"No ground-truth docs for '{mol}' — skipping IR metrics")
            continue

        metrics = compute_ir_metrics(retriever, mol, relevant_ids, K)
        results[mol] = metrics

        agg["precision_at_k"].append(metrics["precision_at_k"])
        agg["recall_at_k"].append(metrics["recall_at_k"])
        agg["mrr"].append(metrics["mrr"])

        logger.info(
            f"  {mol:30s} P@{K}={metrics['precision_at_k']:.3f} "
            f"R@{K}={metrics['recall_at_k']:.3f} "
            f"MRR={metrics['mrr']:.3f}"
        )

    # Aggregate
    n = len(agg["precision_at_k"])
    aggregate = {
        "num_molecules": n,
        "mean_precision_at_k": sum(agg["precision_at_k"]) / n if n else 0,
        "mean_recall_at_k": sum(agg["recall_at_k"]) / n if n else 0,
        "mean_mrr": sum(agg["mrr"]) / n if n else 0,
    }

    return {"per_molecule": results, "aggregate": aggregate}


def analyze_failure_modes(
    retriever: HybridRetriever,
    molecules: List[str],
    relevance_map: Dict[str, Set[str]],
    K: int = 12,
) -> Dict:
    """Identify common retrieval failure patterns.

    Returns:
        Dict with failure analysis — wrong-molecule retrievals,
        section gaps, exact-match vs semantic split.
    """
    wrong_molecule_count = 0
    missing_sections = defaultdict(int)
    method_stats = defaultdict(lambda: {"total": 0, "relevant": 0})

    for mol in molecules:
        mol_lower = mol.lower().strip()
        relevant_ids = relevance_map.get(mol_lower, set())
        if not relevant_ids:
            continue

        results = retriever.retrieve(query_name=mol, fetch_pubchem=False)

        for r in results[:K]:
            method = r.retrieval_method
            method_stats[method]["total"] += 1

            if r.doc_id in relevant_ids:
                method_stats[method]["relevant"] += 1
            else:
                retrieved_mol = r.molecule_name.lower().strip()
                if retrieved_mol and retrieved_mol != mol_lower:
                    wrong_molecule_count += 1

    return {
        "wrong_molecule_retrievals": wrong_molecule_count,
        "retrieval_method_accuracy": {
            method: {
                "total": stats["total"],
                "relevant": stats["relevant"],
                "accuracy": stats["relevant"] / stats["total"] if stats["total"] else 0,
            }
            for method, stats in method_stats.items()
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  Layer B — RAGAS Context Metrics (LLM-judged)
# ═══════════════════════════════════════════════════════════════════════

def _get_ragas_llm_and_embeddings(groq_api_key: str):
    """Initialize RAGAS LLM wrapper and embeddings.

    Uses Groq (Llama-3.3-70b) for LLM and PubMedBERT for embeddings.
    """
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings

    groq_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
    )
    ragas_llm = LangchainLLMWrapper(groq_llm)

    # Use PubMedBERT — same model as our vector store
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    return ragas_llm, ragas_embeddings


async def compute_ragas_context_metrics(
    samples: List[Dict],
    groq_api_key: str,
) -> List[Dict]:
    """Run RAGAS context_precision and context_recall on samples.

    Each sample dict must have:
        - user_input: str
        - retrieved_contexts: List[str]
        - response: str (FULL profile text)
        - reference: str (ground truth summary)

    Returns:
        List of dicts with per-sample RAGAS scores.
    """
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
    from ragas import SingleTurnSample

    ragas_llm, ragas_embeddings = _get_ragas_llm_and_embeddings(groq_api_key)

    precision_metric = LLMContextPrecisionWithReference(llm=ragas_llm)
    recall_metric = LLMContextRecall(llm=ragas_llm)

    results = []
    for i, s in enumerate(samples):
        sample = SingleTurnSample(
            user_input=s["user_input"],
            retrieved_contexts=s["retrieved_contexts"],
            response=s["response"],
            reference=s["reference"],
        )

        try:
            precision = await precision_metric.single_turn_ascore(sample)
        except Exception as e:
            logger.warning(f"Context precision failed for sample {i}: {e}")
            precision = 0.0

        try:
            recall = await recall_metric.single_turn_ascore(sample)
        except Exception as e:
            logger.warning(f"Context recall failed for sample {i}: {e}")
            recall = 0.0

        results.append({
            "user_input": s["user_input"],
            "context_precision": precision,
            "context_recall": recall,
        })

        logger.info(
            f"  RAGAS [{i+1}/{len(samples)}] "
            f"precision={precision:.3f} recall={recall:.3f}"
        )

        # Rate limiting for Groq free tier
        await asyncio.sleep(2)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def print_retrieval_report(ir_results: Dict, ragas_results: Optional[List] = None):
    """Pretty-print Pillar 1 retrieval report."""
    sep = "═" * 66
    thin = "─" * 66

    print(f"\n  {sep}")
    print(f"  ║{'PILLAR 1: RETRIEVAL QUALITY':^62}║")
    print(f"  {sep}\n")

    agg = ir_results["aggregate"]

    print(f"  Custom IR Metrics (offline):")
    print(f"  {thin}")

    p = agg["mean_precision_at_k"]
    r = agg["mean_recall_at_k"]
    m = agg["mean_mrr"]

    status_p = "✓" if p >= 0.70 else "✗"
    status_r = "✓" if r >= 0.80 else "✗"
    status_m = "✓" if m >= 0.85 else "✗"

    print(f"    Precision@K      : {p:.3f}    {status_p}  (≥ 0.70)")
    print(f"    Recall@K         : {r:.3f}    {status_r}  (≥ 0.80)")
    print(f"    MRR              : {m:.3f}    {status_m}  (≥ 0.85)")
    print(f"    Molecules tested : {agg['num_molecules']}")

    if ragas_results:
        print(f"\n  RAGAS Context Metrics (LLM-judged):")
        print(f"  {thin}")

        avg_prec = sum(r["context_precision"] for r in ragas_results) / len(ragas_results)
        avg_rec = sum(r["context_recall"] for r in ragas_results) / len(ragas_results)

        sp = "✓" if avg_prec >= 0.75 else "✗"
        sr = "✓" if avg_rec >= 0.75 else "✗"

        print(f"    Context Precision : {avg_prec:.3f}    {sp}  (≥ 0.75)")
        print(f"    Context Recall    : {avg_rec:.3f}    {sr}  (≥ 0.75)")

    print(f"\n  {sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pillar 1: Retrieval Quality Metrics"
    )
    parser.add_argument("--db-dir", type=str, default="./chroma_db",
                        help="Path to ChromaDB directory")
    parser.add_argument("--k", type=int, default=12, help="Top-K for metrics")
    parser.add_argument("--ragas", action="store_true",
                        help="Include RAGAS context metrics (requires LLM)")
    parser.add_argument("--molecules", type=str, nargs="*",
                        help="Specific molecules to test (default: sample from store)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max molecules to evaluate")
    parser.add_argument("--golden", type=str, default=None,
                        help="Path to golden_test_set.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Initialize store and retriever
    db_path = os.path.join(CURRENT_DIR, args.db_dir) if not os.path.isabs(args.db_dir) else args.db_dir
    store = ToxVectorStore(persist_dir=db_path)
    retriever = HybridRetriever(vector_store=store)

    # Build relevance judgments
    logger.info("Building relevance judgments from vector store...")
    relevance_map = build_relevance_judgments(store)

    # Select molecules to evaluate
    if args.molecules:
        molecules = args.molecules
    elif args.golden and os.path.exists(args.golden):
        with open(args.golden) as f:
            golden = json.load(f)
        molecules = []
        for entry in golden.get("molecules", []):
            name = entry.get("common_name") or entry.get("iupac_name", "")
            if name and entry.get("tier") != "phase4_detox_pair":
                molecules.append(name)
        molecules = molecules[:args.limit]
    else:
        molecules = list(relevance_map.keys())[:args.limit]

    logger.info(f"Evaluating {len(molecules)} molecules with K={args.k}")

    # Layer A: IR Metrics
    ir_results = evaluate_batch(retriever, molecules, relevance_map, K=args.k)

    # Layer B: RAGAS (optional)
    ragas_results = None
    if args.ragas:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            logger.error("GROQ_API_KEY not found — cannot run RAGAS metrics")
        else:
            # Build RAGAS samples from golden set or fallback
            ragas_samples = []
            if args.golden and os.path.exists(args.golden):
                with open(args.golden) as f:
                    golden = json.load(f)
                for entry in golden.get("molecules", [])[:10]:
                    name = entry.get("common_name") or entry.get("iupac_name", "")
                    expected = entry.get("expected", {})
                    ref = expected.get("ground_truth_summary", "")
                    if name and ref:
                        results = retriever.retrieve(query_name=name, fetch_pubchem=False)
                        ragas_samples.append({
                            "user_input": f"Generate toxicological safety profile for {name}",
                            "retrieved_contexts": [r.content for r in results[:args.k]],
                            "response": ref,  # Use ground truth as proxy response
                            "reference": ref,
                        })

            if ragas_samples:
                ragas_results = asyncio.run(
                    compute_ragas_context_metrics(ragas_samples, groq_key)
                )

    # Failure analysis
    failures = analyze_failure_modes(retriever, molecules[:20], relevance_map, K=args.k)

    # Print report
    print_retrieval_report(ir_results, ragas_results)

    if failures.get("wrong_molecule_retrievals", 0) > 0:
        print(f"  ⚠ Wrong-molecule retrievals: {failures['wrong_molecule_retrievals']}")

    # Save results
    if args.output:
        output = {
            "ir_metrics": ir_results,
            "ragas_metrics": ragas_results,
            "failure_analysis": failures,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
