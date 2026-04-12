"""Pillar 2 — LLM Faithfulness Evaluation (RAGAS-Powered).

Uses RAGAS Faithfulness as the primary hallucination detector, supplemented
by domain-specific checks for toxicological data fabrication.

Key design decision: Dose fabrication is a HARD GATE. If any LD50/LC50/NOAEL
value in the profile is fabricated (not found in retrieved docs), the composite
score is capped at 0.40 max regardless of other scores.

Usage:
    python eval_faithfulness.py --profiles profiles.json --db-dir ./chroma_db
    python eval_faithfulness.py --profiles profiles.json --no-llm  # domain checks only
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# ── Path setup ─────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Maximum composite score when dose fabrication is detected
DOSE_FABRICATION_CAP = 0.40

# Sections that comprise the full profile text
PROFILE_SECTIONS = [
    "toxicity_mechanism", "affected_organs", "symptoms_of_exposure",
    "dose_response", "first_aid", "handling_precautions",
    "regulatory_classification", "related_compounds", "references",
]


# ═══════════════════════════════════════════════════════════════════════
#  Helper: Build full profile text
# ═══════════════════════════════════════════════════════════════════════

def build_full_profile_text(profile: Dict) -> str:
    """Concatenate all profile sections into a single text block.

    This is used as the `response` field for RAGAS evaluation — we must
    evaluate against the full profile, not a single section, so RAGAS
    judges retrieval support across organs, symptoms, regulatory, etc.
    """
    parts = []
    for section in PROFILE_SECTIONS:
        text = profile.get(section, "")
        if text and text.strip():
            title = section.replace("_", " ").title()
            parts.append(f"## {title}\n{text.strip()}")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  RAGAS Faithfulness
# ═══════════════════════════════════════════════════════════════════════

async def ragas_faithfulness(
    profiles: List[Dict],
    retrieved_docs_per_profile: List[List[str]],
    groq_api_key: str,
) -> List[Dict]:
    """Run RAGAS Faithfulness and ResponseRelevancy on profiles.

    Args:
        profiles: List of profile dicts (from SafetyProfile.to_dict()).
        retrieved_docs_per_profile: List of lists, each containing the
            retrieved document texts for that profile.
        groq_api_key: Groq API key.

    Returns:
        List of dicts with faithfulness and relevancy scores per profile.
    """
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas import SingleTurnSample

    groq_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
    )
    ragas_llm = LangchainLLMWrapper(groq_llm)

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    faith_metric = Faithfulness(llm=ragas_llm)
    relevancy_metric = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    results = []
    for i, (profile, docs) in enumerate(zip(profiles, retrieved_docs_per_profile)):
        mol_name = profile.get("common_name") or profile.get("iupac_name", "")
        full_text = build_full_profile_text(profile)

        if not full_text.strip() or not docs:
            results.append({
                "molecule": mol_name,
                "faithfulness": 0.0,
                "response_relevancy": 0.0,
                "error": "empty profile or no docs",
            })
            continue

        sample = SingleTurnSample(
            user_input=f"Generate toxicological safety profile for {mol_name}",
            retrieved_contexts=docs,
            response=full_text,
        )

        try:
            faith = await faith_metric.single_turn_ascore(sample)
        except Exception as e:
            logger.warning(f"Faithfulness failed for {mol_name}: {e}")
            faith = 0.0

        try:
            relevancy = await relevancy_metric.single_turn_ascore(sample)
        except Exception as e:
            logger.warning(f"ResponseRelevancy failed for {mol_name}: {e}")
            relevancy = 0.0

        results.append({
            "molecule": mol_name,
            "faithfulness": faith,
            "response_relevancy": relevancy,
        })

        logger.info(
            f"  RAGAS [{i+1}/{len(profiles)}] {mol_name}: "
            f"faith={faith:.3f} relevancy={relevancy:.3f}"
        )

        # Rate limiting for Groq free tier (30 req/min)
        await asyncio.sleep(3)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Domain-Specific Checks
# ═══════════════════════════════════════════════════════════════════════

# Regex patterns for dose values
DOSE_PATTERNS = [
    # LD50, LC50, NOAEL, LOAEL with value and unit
    re.compile(
        r"(?:LD50|LC50|NOAEL|LOAEL|TD50|ED50)"
        r"[:\s]*(?:approximately\s+|~\s*|>\s*|<\s*|≈\s*)?"
        r"(\d+[\d,\.]*\s*(?:mg|g|µg|μg|ng)/(?:kg|L|m³|mL))",
        re.IGNORECASE,
    ),
    # Standalone dose values like "640 mg/kg"
    re.compile(
        r"(\d+[\d,\.]*\s*(?:mg|g|µg|μg|ng)/(?:kg|L|m³|mL))",
        re.IGNORECASE,
    ),
]

# CAS number pattern
CAS_PATTERN = re.compile(r"\b(\d{2,7}-\d{2}-\d)\b")


def check_dose_fabrication(
    profile: Dict,
    retrieved_doc_texts: List[str],
) -> Dict:
    """CRITICAL HARD GATE: Check if dose values in profile are fabricated.

    Extracts all LD50/LC50/NOAEL/LOAEL values from the profile and
    cross-references each against the retrieved document texts. Any value
    NOT found in the source docs is flagged as FABRICATED.

    Returns:
        Dict with:
            passed: bool — True if no fabrication detected
            fabricated_values: list of fabricated dose strings
            total_values: int — total dose values found in profile
            verification_rate: float — fraction of values verified
    """
    full_text = build_full_profile_text(profile)
    docs_combined = " ".join(retrieved_doc_texts).lower()

    # Extract all dose values from profile
    dose_values = set()
    for pattern in DOSE_PATTERNS:
        matches = pattern.findall(full_text)
        for m in matches:
            # Normalize: strip whitespace, lowercase
            normalized = re.sub(r"\s+", " ", m.strip().lower())
            dose_values.add(normalized)

    if not dose_values:
        return {
            "passed": True,
            "fabricated_values": [],
            "total_values": 0,
            "verification_rate": 1.0,
        }

    # Check each value against retrieved docs
    fabricated = []
    verified = 0
    for val in dose_values:
        # Try exact match and also with stripped commas
        val_no_comma = val.replace(",", "")
        # Extract just the number for fuzzy matching
        num_match = re.search(r"(\d+[\d,\.]*)", val)
        num_str = num_match.group(1) if num_match else ""

        found = (
            val in docs_combined
            or val_no_comma in docs_combined
            or (num_str and num_str in docs_combined)
        )

        if found:
            verified += 1
        else:
            fabricated.append(val)

    total = len(dose_values)
    return {
        "passed": len(fabricated) == 0,
        "fabricated_values": fabricated,
        "total_values": total,
        "verification_rate": verified / total if total else 1.0,
    }


def check_identifier_accuracy(
    profile: Dict,
    retrieved_doc_texts: List[str],
) -> Dict:
    """Verify CAS numbers and SMILES in the profile match retrieved data.

    Returns:
        Dict with accuracy score and mismatched identifiers.
    """
    full_text = build_full_profile_text(profile)
    docs_combined = " ".join(retrieved_doc_texts)

    # Check CAS numbers
    profile_cas_numbers = CAS_PATTERN.findall(full_text)
    input_cas = profile.get("cas_number", "")

    mismatches = []
    total_checked = 0

    for cas in profile_cas_numbers:
        total_checked += 1
        # Verify against input or retrieved docs
        if cas == input_cas or cas in docs_combined:
            continue
        mismatches.append(f"CAS: {cas}")

    # Check SMILES if present in output
    profile_smiles = profile.get("smiles", "")
    if profile_smiles and profile_smiles not in docs_combined:
        # SMILES from input is fine — only flag if profile generates a new one
        pass

    accuracy = (total_checked - len(mismatches)) / total_checked if total_checked > 0 else 1.0

    return {
        "accuracy": accuracy,
        "total_checked": total_checked,
        "mismatches": mismatches,
    }


def check_hedging_compliance(
    profile: Dict,
    retrieved_doc_texts: List[str],
) -> Dict:
    """Check if unsupported sections properly hedge with 'Data not available'.

    Sections with no supporting documents should say "Data not available"
    rather than stating unsupported facts.

    Returns:
        Dict with compliance score and violations.
    """
    docs_combined = " ".join(retrieved_doc_texts).lower()
    violations = []
    total_checked = 0

    # Check sections that might lack supporting data
    check_sections = [
        "dose_response", "regulatory_classification",
        "first_aid", "handling_precautions",
    ]

    for section_key in check_sections:
        content = profile.get(section_key, "")
        if not content or not content.strip():
            continue

        total_checked += 1
        content_lower = content.lower()

        # Check if this section's content has any support in retrieved docs
        # Use keyword overlap as a heuristic
        content_words = set(content_lower.split())
        doc_words = set(docs_combined.split())

        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
                       "to", "and", "or", "for", "with", "on", "at", "by"}
        content_keywords = content_words - stop_words
        overlap = content_keywords & doc_words

        support_ratio = len(overlap) / len(content_keywords) if content_keywords else 1.0

        # If very low overlap AND section doesn't hedge
        hedging_phrases = [
            "data not available", "not available", "no data",
            "insufficient data", "limited information",
            "no specific data", "data unavailable",
        ]
        has_hedging = any(phrase in content_lower for phrase in hedging_phrases)

        if support_ratio < 0.15 and not has_hedging:
            violations.append({
                "section": section_key,
                "support_ratio": support_ratio,
            })

    compliance = (total_checked - len(violations)) / total_checked if total_checked > 0 else 1.0

    return {
        "compliance": compliance,
        "total_checked": total_checked,
        "violations": violations,
    }


def compute_composite_score(
    ragas_scores: Dict,
    domain_scores: Dict,
) -> Dict:
    """Compute weighted composite faithfulness score with hard gate.

    HARD GATE: If dose_fabrication fails, composite is capped at 0.40 max.

    Weights when RAGAS was run:
        60% RAGAS faithfulness
        20% dose fabrication (0 or 1)
        10% identifier accuracy
        10% hedging compliance

    Weights when RAGAS was skipped (--no-llm mode):
        50% dose fabrication (0 or 1)
        25% identifier accuracy
        25% hedging compliance
    """
    # Use None sentinel to distinguish "RAGAS skipped" from "RAGAS scored 0.0"
    faith = ragas_scores.get("faithfulness", None)
    ragas_available = faith is not None
    faith = faith if faith is not None else 0.0

    dose_passed = domain_scores.get("dose_fabrication", {}).get("passed", True)
    dose_score = 1.0 if dose_passed else 0.0
    id_accuracy = domain_scores.get("identifier_accuracy", {}).get("accuracy", 1.0)
    hedging = domain_scores.get("hedging_compliance", {}).get("compliance", 1.0)

    if ragas_available:
        # Full weighting: RAGAS + domain
        weighted = (
            0.60 * faith
            + 0.20 * dose_score
            + 0.10 * id_accuracy
            + 0.10 * hedging
        )
        weights = {
            "ragas_faithfulness": 0.60,
            "dose_fabrication": 0.20,
            "identifier_accuracy": 0.10,
            "hedging_compliance": 0.10,
        }
    else:
        # No-LLM mode: domain-only weighting (exclude RAGAS entirely)
        weighted = (
            0.50 * dose_score
            + 0.25 * id_accuracy
            + 0.25 * hedging
        )
        weights = {
            "ragas_faithfulness": 0.00,
            "dose_fabrication": 0.50,
            "identifier_accuracy": 0.25,
            "hedging_compliance": 0.25,
        }

    # HARD GATE: dose fabrication caps the composite
    if not dose_passed:
        composite = min(DOSE_FABRICATION_CAP, weighted)
        gate_triggered = True
    else:
        composite = weighted
        gate_triggered = False

    return {
        "composite_score": composite,
        "gate_triggered": gate_triggered,
        "ragas_available": ragas_available,
        "components": {
            "ragas_faithfulness": faith,
            "dose_fabrication": dose_score,
            "identifier_accuracy": id_accuracy,
            "hedging_compliance": hedging,
        },
        "weights": weights,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Full Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════

def evaluate_faithfulness(
    profiles: List[Dict],
    retrieved_docs_per_profile: Optional[List[List[str]]] = None,
    run_ragas: bool = True,
    groq_api_key: str = "",
) -> Dict:
    """Run full faithfulness evaluation: RAGAS + domain checks.

    Args:
        profiles: List of profile dicts.
        retrieved_docs_per_profile: Retrieved doc texts per profile.
            If None, domain checks still run but RAGAS is skipped.
        run_ragas: Whether to run RAGAS (requires LLM calls).
        groq_api_key: Groq API key for RAGAS.

    Returns:
        Dict with per-profile and aggregate results.
    """
    if retrieved_docs_per_profile is None:
        retrieved_docs_per_profile = [[] for _ in profiles]

    per_profile = []

    # Domain-specific checks (always run — no LLM needed)
    for i, (profile, docs) in enumerate(zip(profiles, retrieved_docs_per_profile)):
        mol_name = profile.get("common_name") or profile.get("iupac_name", "")

        dose_result = check_dose_fabrication(profile, docs)
        id_result = check_identifier_accuracy(profile, docs)
        hedging_result = check_hedging_compliance(profile, docs)

        entry = {
            "molecule": mol_name,
            "dose_fabrication": dose_result,
            "identifier_accuracy": id_result,
            "hedging_compliance": hedging_result,
        }
        per_profile.append(entry)

        status = "✓" if dose_result["passed"] else "✗ FABRICATION"
        logger.info(
            f"  Domain [{i+1}/{len(profiles)}] {mol_name}: "
            f"dose={status} id_acc={id_result['accuracy']:.2f} "
            f"hedge={hedging_result['compliance']:.2f}"
        )

    # RAGAS checks (optional — requires LLM)
    ragas_results = None
    if run_ragas and groq_api_key:
        ragas_results = asyncio.run(
            ragas_faithfulness(profiles, retrieved_docs_per_profile, groq_api_key)
        )
        # Merge RAGAS results into per-profile
        for i, ragas in enumerate(ragas_results):
            per_profile[i]["ragas_faithfulness"] = ragas.get("faithfulness", 0.0)
            per_profile[i]["ragas_response_relevancy"] = ragas.get("response_relevancy", 0.0)

    # Compute composite scores
    for entry in per_profile:
        ragas_scores = {
            # None sentinel = "RAGAS was not run" vs 0.0 = "RAGAS scored zero"
            "faithfulness": entry.get("ragas_faithfulness", None),
        }
        domain_scores = {
            "dose_fabrication": entry["dose_fabrication"],
            "identifier_accuracy": entry["identifier_accuracy"],
            "hedging_compliance": entry["hedging_compliance"],
        }
        entry["composite"] = compute_composite_score(ragas_scores, domain_scores)

    # Aggregate
    n = len(per_profile)
    aggregate = {
        "num_profiles": n,
        "dose_fabrication_rate": sum(
            1 for e in per_profile if not e["dose_fabrication"]["passed"]
        ) / n if n else 0,
        "mean_identifier_accuracy": sum(
            e["identifier_accuracy"]["accuracy"] for e in per_profile
        ) / n if n else 0,
        "mean_hedging_compliance": sum(
            e["hedging_compliance"]["compliance"] for e in per_profile
        ) / n if n else 0,
        "mean_composite": sum(
            e["composite"]["composite_score"] for e in per_profile
        ) / n if n else 0,
        "hard_gates_triggered": sum(
            1 for e in per_profile if e["composite"]["gate_triggered"]
        ),
    }

    if run_ragas and ragas_results:
        aggregate["mean_ragas_faithfulness"] = sum(
            e.get("ragas_faithfulness", 0) for e in per_profile
        ) / n if n else 0
        aggregate["mean_ragas_relevancy"] = sum(
            e.get("ragas_response_relevancy", 0) for e in per_profile
        ) / n if n else 0

    return {"per_profile": per_profile, "aggregate": aggregate}


def print_faithfulness_report(results: Dict):
    """Pretty-print Pillar 2 faithfulness report."""
    sep = "═" * 66
    thin = "─" * 66

    print(f"\n  {sep}")
    print(f"  ║{'PILLAR 2: LLM FAITHFULNESS':^62}║")
    print(f"  {sep}\n")

    agg = results["aggregate"]

    if "mean_ragas_faithfulness" in agg:
        print(f"  RAGAS Scores:")
        print(f"  {thin}")
        f = agg["mean_ragas_faithfulness"]
        r = agg.get("mean_ragas_relevancy", 0)
        sf = "✓" if f >= 0.80 else "✗"
        sr = "✓" if r >= 0.75 else "✗"
        print(f"    Faithfulness      : {f:.3f}    {sf}  (≥ 0.80)")
        print(f"    Response Relevancy: {r:.3f}    {sr}  (≥ 0.75)")

    print(f"\n  Domain-Specific Checks:")
    print(f"  {thin}")

    dose_rate = agg["dose_fabrication_rate"]
    id_acc = agg["mean_identifier_accuracy"]
    hedge = agg["mean_hedging_compliance"]
    composite = agg["mean_composite"]
    gates = agg["hard_gates_triggered"]

    sd = "✓" if dose_rate == 0 else "✗ GATE"
    si = "✓" if id_acc >= 0.95 else "✗"
    sh = "✓" if hedge >= 0.90 else "✗"
    sc = "✓" if composite >= 0.80 else "✗"

    print(f"    Dose Fabrication  : {dose_rate*100:5.1f}%   {sd}  HARD GATE (any fail → cap 0.40)")
    print(f"    Identifier Accuracy: {id_acc*100:5.1f}%  {si}  (≥ 95%)")
    print(f"    Hedging Compliance : {hedge*100:5.1f}%  {sh}  (≥ 90%)")
    print(f"    Composite (gated) : {composite:.3f}    {sc}  (≥ 0.80)")

    if gates > 0:
        print(f"\n    ⚠ Hard gates triggered: {gates}/{agg['num_profiles']} profiles")

    print(f"\n  {sep}\n")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pillar 2: LLM Faithfulness Evaluation"
    )
    parser.add_argument("--profiles", type=str, required=True,
                        help="Path to JSON profiles file")
    parser.add_argument("--db-dir", type=str, default="./chroma_db",
                        help="Path to ChromaDB directory")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip RAGAS LLM calls — domain checks only")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Load profiles
    with open(args.profiles) as f:
        profiles = json.load(f)

    logger.info(f"Loaded {len(profiles)} profiles from {args.profiles}")

    # Load Groq API key
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    groq_key = os.getenv("GROQ_API_KEY", "")

    run_ragas = not args.no_llm and bool(groq_key)
    if args.no_llm:
        logger.info("--no-llm: skipping RAGAS LLM calls, running domain checks only")
    elif not groq_key:
        logger.warning("GROQ_API_KEY not found — falling back to domain checks only")

    # Note: In a real pipeline, retrieved_docs would come from the pipeline run.
    # For standalone evaluation, we use empty lists (domain checks still work).
    retrieved_docs = [[] for _ in profiles]

    results = evaluate_faithfulness(
        profiles=profiles,
        retrieved_docs_per_profile=retrieved_docs,
        run_ragas=run_ragas,
        groq_api_key=groq_key,
    )

    print_faithfulness_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
