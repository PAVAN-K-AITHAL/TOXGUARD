"""Pillar 4 — Phase 4 Structural-Analogy Retrieval Tests.

Tests whether the RAG correctly retrieves analogue toxicology when queried
with novel Phase 4 candidate molecules. This is the critical gap that the
original evaluation missed — Phase 4 queries by structural analogy, not by
exact molecule name.

Metrics:
    - Analogue Retrieval Rate: % of candidates where correct analogue found
    - Analogue MRR: mean rank of first analogue doc in retrieved set
    - Dossier Keyword Accuracy: % of candidate profiles with expected keywords
    - Cross-contamination Rate: % citing data from unrelated molecules

Usage:
    python eval_phase4_retrieval.py --db-dir ./chroma_db --golden golden_test_set.json
    python eval_phase4_retrieval.py --db-dir ./chroma_db --golden golden_test_set.json --no-llm
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# ── Path setup ─────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PHASE3_DIR)
sys.path.insert(0, PHASE3_DIR)
sys.path.insert(0, PROJECT_ROOT)

from vector_store import ToxVectorStore
from retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════

class DetoxPair:
    """A seed → candidate detoxification pair from the golden set."""

    def __init__(self, entry: Dict):
        self.id = entry.get("id", "")
        self.seed_iupac = entry.get("seed_iupac_name", "")
        self.candidate_iupac = entry.get("candidate_iupac_name", "")
        self.seed_cas = entry.get("seed_cas", "")
        self.candidate_cas = entry.get("candidate_cas", "")

        expected = entry.get("expected", {})
        self.expected_seed_retrieval = expected.get("seed_retrieval", "")
        self.expected_candidate_retrieval = expected.get("candidate_retrieval", "")
        self.expected_analogue = expected.get("candidate_should_retrieve_analogue", "")
        self.candidate_mechanism_keywords = expected.get("candidate_mechanism_keywords", [])
        self.candidate_target_organs = expected.get("candidate_target_organs", [])
        self.structural_relationship = expected.get("structural_relationship", "")
        self.ground_truth = expected.get("ground_truth_summary", "")

    def __repr__(self):
        return f"DetoxPair({self.seed_iupac} → {self.candidate_iupac})"


def load_detox_pairs(golden_set_path: str) -> List[DetoxPair]:
    """Load Phase 4 detox pair entries from golden_test_set.json."""
    with open(golden_set_path) as f:
        golden = json.load(f)

    pairs = []
    for entry in golden.get("molecules", []):
        if entry.get("tier") == "phase4_detox_pair":
            pairs.append(DetoxPair(entry))

    logger.info(f"Loaded {len(pairs)} Phase 4 detox pairs from golden set")
    return pairs


# ═══════════════════════════════════════════════════════════════════════
#  Candidate Retrieval Test
# ═══════════════════════════════════════════════════════════════════════

def test_candidate_retrieval(
    retriever: HybridRetriever,
    candidate_name: str,
    expected_analogue: str,
    seed_name: str = "",
    K: int = 12,
) -> Dict:
    """Test if retriever finds the expected analogue's docs for a candidate.

    Args:
        retriever: HybridRetriever instance.
        candidate_name: IUPAC/common name of the candidate.
        expected_analogue: Name of the molecule whose docs should be retrieved.
        seed_name: Parent toxic molecule to exclude from results.
        K: Number of top results to check.

    Returns:
        Dict with:
            analogue_found: bool
            analogue_rank: int (0 if not found)
            retrieved_molecules: list of unique molecule names in top-K
            top_k_methods: list of retrieval methods used
    """
    results = retriever.retrieve(
        query_name=candidate_name,
        fetch_pubchem=False,  # Test only against existing KB
        exclude_molecule=seed_name if seed_name else None,
    )

    top_k = results[:K]
    analogue_lower = expected_analogue.lower().strip()

    # Find the first occurrence of the analogue's docs
    analogue_rank = 0
    analogue_found = False
    retrieved_molecules = []
    top_k_methods = []

    for rank, r in enumerate(top_k, 1):
        mol_name = r.molecule_name.lower().strip()
        if mol_name and mol_name not in retrieved_molecules:
            retrieved_molecules.append(mol_name)

        top_k_methods.append(r.retrieval_method)

        if mol_name == analogue_lower and not analogue_found:
            analogue_rank = rank
            analogue_found = True

    # Also check partial matches (e.g., "nitrobenzene" in "Nitrobenzene")
    if not analogue_found:
        for rank, r in enumerate(top_k, 1):
            mol_name = r.molecule_name.lower().strip()
            if analogue_lower in mol_name or mol_name in analogue_lower:
                analogue_rank = rank
                analogue_found = True
                break

    return {
        "analogue_found": analogue_found,
        "analogue_rank": analogue_rank,
        "retrieved_molecules": retrieved_molecules[:10],
        "top_k_methods": list(set(top_k_methods)),
        "num_retrieved": len(results),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Dossier Accuracy Test
# ═══════════════════════════════════════════════════════════════════════

def test_dossier_keyword_accuracy(
    retriever: HybridRetriever,
    pair: DetoxPair,
    K: int = 12,
) -> Dict:
    """Check if retrieved docs for candidate contain expected mechanism keywords.

    This is a no-LLM test — we check the retrieved document texts directly,
    not the generated profile. This tells us if the right information is
    being provided to the LLM before generation.

    Returns:
        Dict with keyword_accuracy and matched/missing keywords.
    """
    results = retriever.retrieve(
        query_name=pair.candidate_iupac,
        fetch_pubchem=False,
        exclude_molecule=pair.seed_iupac if pair.seed_iupac else None,
    )

    # Combine all retrieved doc texts
    combined = " ".join(r.content.lower() for r in results[:K])

    # Check mechanism keywords
    keywords = pair.candidate_mechanism_keywords
    if not keywords:
        return {"keyword_accuracy": 1.0, "matched": [], "missing": []}

    matched = [kw for kw in keywords if kw.lower() in combined]
    missing = [kw for kw in keywords if kw.lower() not in combined]

    accuracy = len(matched) / len(keywords) if keywords else 1.0

    return {
        "keyword_accuracy": accuracy,
        "matched": matched,
        "missing": missing,
    }


def check_cross_contamination(
    retriever: HybridRetriever,
    pair: DetoxPair,
    K: int = 12,
) -> Dict:
    """Check if candidate profile pulls data from unrelated molecules.

    Cross-contamination occurs when the retriever surfaces docs from
    molecules that are neither the candidate nor the expected analogue.

    Returns:
        Dict with contamination_rate and contaminant molecules.
    """
    results = retriever.retrieve(
        query_name=pair.candidate_iupac,
        fetch_pubchem=False,
        exclude_molecule=pair.seed_iupac if pair.seed_iupac else None,
    )

    top_k = results[:K]
    if not top_k:
        return {"contamination_rate": 0.0, "contaminants": []}

    candidate_lower = pair.candidate_iupac.lower().strip()
    analogue_lower = pair.expected_analogue.lower().strip()

    # Also accept the seed molecule as non-contaminant
    seed_lower = pair.seed_iupac.lower().strip()
    acceptable = {candidate_lower, analogue_lower, seed_lower, ""}

    contaminant_names = set()
    contaminant_count = 0

    for r in top_k:
        mol_name = r.molecule_name.lower().strip()

        # Check if this molecule is acceptable
        is_acceptable = (
            mol_name in acceptable
            or any(acc in mol_name or mol_name in acc for acc in acceptable if acc)
        )

        if not is_acceptable:
            contaminant_count += 1
            if mol_name:
                contaminant_names.add(mol_name)

    rate = contaminant_count / len(top_k) if top_k else 0.0

    return {
        "contamination_rate": rate,
        "contaminant_count": contaminant_count,
        "total_docs": len(top_k),
        "contaminants": list(contaminant_names)[:5],
    }


# ═══════════════════════════════════════════════════════════════════════
#  Aggregate Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_phase4_metrics(
    retrieval_results: List[Dict],
    keyword_results: List[Dict],
    contamination_results: List[Dict],
) -> Dict:
    """Compute aggregate Phase 4 metrics.

    Returns:
        Dict with all Phase 4 metrics and pass/fail status.
    """
    n = len(retrieval_results)
    if n == 0:
        return {"error": "No results to compute metrics"}

    # Analogue Retrieval Rate
    found_count = sum(1 for r in retrieval_results if r["analogue_found"])
    retrieval_rate = found_count / n

    # Analogue MRR
    mrr_values = []
    for r in retrieval_results:
        if r["analogue_found"] and r["analogue_rank"] > 0:
            mrr_values.append(1.0 / r["analogue_rank"])
        else:
            mrr_values.append(0.0)
    mean_mrr = sum(mrr_values) / n

    # Dossier Keyword Accuracy
    kw_scores = [r["keyword_accuracy"] for r in keyword_results]
    mean_kw_accuracy = sum(kw_scores) / n

    # Cross-contamination Rate
    contam_rates = [r["contamination_rate"] for r in contamination_results]
    mean_contamination = sum(contam_rates) / n

    return {
        "num_pairs": n,
        "analogue_retrieval_rate": retrieval_rate,
        "analogue_mrr": mean_mrr,
        "dossier_keyword_accuracy": mean_kw_accuracy,
        "cross_contamination_rate": mean_contamination,
        "pass_retrieval_rate": retrieval_rate >= 0.70,
        "pass_mrr": mean_mrr >= 0.50,
        "pass_keyword_accuracy": mean_kw_accuracy >= 0.65,
        "pass_contamination": mean_contamination <= 0.10,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Report Printing
# ═══════════════════════════════════════════════════════════════════════

def print_phase4_report(
    metrics: Dict,
    retrieval_results: List[Dict],
    pairs: List[DetoxPair],
):
    """Pretty-print Pillar 4 report."""
    sep = "═" * 66
    thin = "─" * 66

    print(f"\n  {sep}")
    print(f"  ║{'PILLAR 4: PHASE 4 ANALOGUE RETRIEVAL':^62}║")
    print(f"  {sep}\n")

    n = metrics["num_pairs"]

    rr = metrics["analogue_retrieval_rate"]
    mrr = metrics["analogue_mrr"]
    kw = metrics["dossier_keyword_accuracy"]
    cc = metrics["cross_contamination_rate"]

    sr = "✓" if metrics["pass_retrieval_rate"] else "✗"
    sm = "✓" if metrics["pass_mrr"] else "✗"
    sk = "✓" if metrics["pass_keyword_accuracy"] else "✗"
    sc = "✓" if metrics["pass_contamination"] else "✗"

    print(f"  Detox Pairs Tested: {n}")
    print(f"  {thin}")
    print(f"    Analogue Retrieval Rate : {rr*100:5.1f}%   {sr}  (≥ 70%)")
    print(f"    Analogue MRR            : {mrr:.3f}    {sm}  (≥ 0.50)")
    print(f"    Dossier Keyword Accuracy: {kw*100:5.1f}%   {sk}  (≥ 65%)")
    print(f"    Cross-contamination     : {cc*100:5.1f}%   {sc}  (≤ 10%)")

    # Per-pair details
    print(f"\n  Per-Pair Details:")
    print(f"  {thin}")
    for pair, res in zip(pairs, retrieval_results):
        status = "✓" if res["analogue_found"] else "✗"
        rank = f"rank={res['analogue_rank']}" if res["analogue_found"] else "NOT FOUND"
        print(
            f"    {status} {pair.seed_iupac} → {pair.candidate_iupac} "
            f"| analogue='{pair.expected_analogue}' {rank}"
        )
        if not res["analogue_found"] and res["retrieved_molecules"]:
            top3 = ", ".join(res["retrieved_molecules"][:3])
            print(f"      Retrieved instead: {top3}")

    print(f"\n  {sep}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════

def evaluate_phase4(
    retriever: HybridRetriever,
    golden_set_path: str,
    K: int = 12,
) -> Dict:
    """Run full Phase 4 evaluation pipeline.

    Returns:
        Dict with metrics, per-pair results, and pass/fail status.
    """
    pairs = load_detox_pairs(golden_set_path)
    if not pairs:
        logger.warning("No Phase 4 detox pairs found in golden set")
        return {"error": "No detox pairs found"}

    retrieval_results = []
    keyword_results = []
    contamination_results = []

    for i, pair in enumerate(pairs):
        logger.info(f"[{i+1}/{len(pairs)}] Testing: {pair.seed_iupac} → {pair.candidate_iupac}")

        # Test 1: Can retriever find the analogue?
        ret = test_candidate_retrieval(
            retriever, pair.candidate_iupac, pair.expected_analogue,
            seed_name=pair.seed_iupac, K=K,
        )
        retrieval_results.append(ret)

        status = "FOUND" if ret["analogue_found"] else "MISSED"
        logger.info(f"  Analogue retrieval: {status} (rank={ret['analogue_rank']})")

        # Test 2: Do retrieved docs contain expected keywords?
        kw = test_dossier_keyword_accuracy(retriever, pair, K)
        keyword_results.append(kw)
        logger.info(f"  Keyword accuracy: {kw['keyword_accuracy']:.2f}")

        # Test 3: Cross-contamination check
        cc = check_cross_contamination(retriever, pair, K)
        contamination_results.append(cc)
        if cc["contamination_rate"] > 0:
            logger.info(f"  Cross-contamination: {cc['contamination_rate']:.2f}")

    # Compute aggregate metrics
    metrics = compute_phase4_metrics(
        retrieval_results, keyword_results, contamination_results
    )

    # Print report
    print_phase4_report(metrics, retrieval_results, pairs)

    return {
        "metrics": metrics,
        "per_pair": [
            {
                "pair": str(pair),
                "seed": pair.seed_iupac,
                "candidate": pair.candidate_iupac,
                "expected_analogue": pair.expected_analogue,
                "retrieval": ret,
                "keywords": kw,
                "contamination": cc,
            }
            for pair, ret, kw, cc in zip(
                pairs, retrieval_results, keyword_results, contamination_results
            )
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pillar 4: Phase 4 Structural-Analogy Retrieval Tests"
    )
    parser.add_argument("--db-dir", type=str, default="./chroma_db",
                        help="Path to ChromaDB directory")
    parser.add_argument("--golden", type=str, required=True,
                        help="Path to golden_test_set.json")
    parser.add_argument("--k", type=int, default=12,
                        help="Top-K for retrieval")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    db_path = os.path.join(PHASE3_DIR, args.db_dir) if not os.path.isabs(args.db_dir) else args.db_dir
    store = ToxVectorStore(persist_dir=db_path)
    retriever = HybridRetriever(vector_store=store)

    results = evaluate_phase4(retriever, args.golden, K=args.k)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
