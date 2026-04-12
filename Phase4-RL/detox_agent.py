"""Agentic detoxification workflow for Phase 4.

Orchestrates an iterative refinement loop:
    Round 1: Generate N candidates → Validate → Score → Select best
    Round 2: If no valid less-toxic candidate, analyze failures → Adjust
             strategy → Regenerate with modified parameters
    ...
    Round K: Return best candidate or report failure (max 5 rounds)

The agent adaptively adjusts:
    - Temperature (higher on retry)
    - Prefix fraction (shorter prefix for more diversity)
    - Batch size (larger on retry)
    - Similarity threshold (relaxed if too strict)

Usage:
    agent = DetoxAgent(
        generator=generator,
        reward_fn=reward_fn,
        resolver=resolver,
        config=config,
    )
    report = agent.detoxify("nitrobenzene", seed_p_toxic=0.91)
    print(report)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Explanation is now handled by Phase 2/3 handoff (detox_dossier.py)
# ExplainerAgent removed — use detox_dossier.generate_dossier() instead

logger = logging.getLogger(__name__)

try:
    from .ppo_config import PPOConfig
    from .molecule_generator import MoleculeGenerator
    from .reward_function import RewardFunction, RewardInfo
    from .name_resolver import NameResolver
    from .molecule_validator import MoleculeValidator
    from .multi_agent import AnalystAgent, VerifierAgent, ReviewerAgent
    from .property_matcher import PropertyMatcher
    from .conversion_efficiency import ConversionEfficiency
    from .scaffold_detox import ScaffoldDetox
except ImportError:
    from ppo_config import PPOConfig
    from molecule_generator import MoleculeGenerator
    from reward_function import RewardFunction, RewardInfo
    from name_resolver import NameResolver
    from molecule_validator import MoleculeValidator
    from multi_agent import AnalystAgent, VerifierAgent, ReviewerAgent
    from property_matcher import PropertyMatcher
    from conversion_efficiency import ConversionEfficiency
    from scaffold_detox import ScaffoldDetox


@dataclass
class DetoxCandidate:
    """A single detoxification candidate."""
    iupac_name: str = ""
    smiles: str = ""
    p_toxic: float = 1.0
    reward: float = 0.0
    tanimoto: float = 0.0
    qed: float = 0.0
    sa_score: float = 0.0
    property_score: float = 0.0        # NEW: chemical property preservation
    conversion_score: float = 0.0      # NEW: conversion efficiency (MCS)
    delta_toxicity: float = 0.0
    valid: bool = False
    verified: bool = False             # NEW: passed verifier agent
    toxicophore_removed: bool = False  # NEW: toxicophore checked by verifier
    round_generated: int = 0
    source: str = "gpt"                # "scaffold" or "gpt"


@dataclass
class DetoxReport:
    """Complete detoxification report for a molecule."""

    # Seed molecule
    seed_iupac: str = ""
    seed_smiles: str = ""
    seed_p_toxic: float = 0.0
    seed_severity: str = ""

    # Best candidate
    best_candidate: Optional[DetoxCandidate] = None
    success: bool = False

    # All candidates across rounds
    all_candidates: List[DetoxCandidate] = field(default_factory=list)
    valid_candidates: List[DetoxCandidate] = field(default_factory=list)
    less_toxic_candidates: List[DetoxCandidate] = field(default_factory=list)

    # Agent stats
    rounds_used: int = 0
    total_generated: int = 0
    total_valid: int = 0
    total_less_toxic: int = 0
    total_time_s: float = 0.0

    # Explanation of detoxification (populated by Phase 2/3 handoff)
    explanation: Optional[Dict] = None

    # Strategy adjustments made
    strategy_history: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.success and self.best_candidate:
            bc = self.best_candidate
            return (
                f"[OK] Detoxified: {self.seed_iupac} -> {bc.iupac_name}\n"
                f"   P(toxic): {self.seed_p_toxic:.3f} -> {bc.p_toxic:.3f} "
                f"(Δ = -{bc.delta_toxicity:.3f})\n"
                f"   Tanimoto: {bc.tanimoto:.3f} | QED: {bc.qed:.3f} | "
                f"SA: {bc.sa_score:.3f}\n"
                f"   Rounds: {self.rounds_used} | "
                f"Candidates: {self.total_generated} "
                f"({self.total_valid} valid, {self.total_less_toxic} less-toxic)\n"
                f"   Time: {self.total_time_s:.1f}s"
            )
        else:
            return (
                f"[FAIL] Failed to detoxify: {self.seed_iupac}\n"
                f"   Seed P(toxic): {self.seed_p_toxic:.3f}\n"
                f"   Rounds: {self.rounds_used} | "
                f"Candidates: {self.total_generated} "
                f"({self.total_valid} valid)\n"
                f"   Time: {self.total_time_s:.1f}s"
            )

    def detailed_report(self) -> str:
        """Multi-line detailed report for display."""
        sep = "=" * 60
        thin = "-" * 60
        lines = [
            sep,
            "  MOLECULE DETOXIFICATION REPORT",
            sep,
            "",
            "  SEED MOLECULE",
            thin,
            f"  IUPAC:     {self.seed_iupac}",
            f"  SMILES:    {self.seed_smiles}",
            f"  P(toxic):  {self.seed_p_toxic:.4f}",
            f"  Severity:  {self.seed_severity}",
            "",
        ]

        if self.success and self.best_candidate:
            bc = self.best_candidate
            lines.extend([
                "  [OK] PROPOSED DETOXIFIED MOLECULE",
                thin,
                f"  IUPAC:       {bc.iupac_name}",
                f"  SMILES:      {bc.smiles}",
                f"  P(toxic):    {bc.p_toxic:.4f}  (delta = -{bc.delta_toxicity:.4f})",
                f"  Tanimoto:    {bc.tanimoto:.4f}  (structural similarity)",
                f"  QED:         {bc.qed:.4f}  (drug-likeness)",
                f"  SA Score:    {bc.sa_score:.4f}  (synthetic accessibility)",
                f"  PropScore:   {bc.property_score:.4f}  (chemical property preservation)",
                f"  ConvScore:   {bc.conversion_score:.4f}  (conversion efficiency)",
                f"  Reward:      {bc.reward:.4f}",
                f"  Source:      {bc.source}",
                f"  Verified:    {'Yes' if bc.verified else 'No'}",
                "",
            ])
            # Explanation is generated by Phase 2/3 handoff (detox_dossier.py)
            if self.explanation:
                lines.append(str(self.explanation))
        else:
            lines.extend([
                "  [FAIL] NO VALID LESS-TOXIC CANDIDATE FOUND",
                thin,
                "",
            ])

        # Top 5 less toxic candidates
        if self.less_toxic_candidates:
            lines.append("  TOP DETOXIFICATION CANDIDATES")
            lines.append(thin)
            for i, c in enumerate(self.less_toxic_candidates[:5], 1):
                lines.append(
                    f"  {i}. [{c.source}] {c.iupac_name}"
                )
                lines.append(
                    f"     P(tox)={c.p_toxic:.3f} "
                    f"Tan={c.tanimoto:.3f} "
                    f"Prop={c.property_score:.3f} "
                    f"Conv={c.conversion_score:.3f} "
                    f"R={c.reward:.3f} "
                    f"{'[verified]' if c.verified else ''} "
                    f"(round {c.round_generated})"
                )
            lines.append("")

        # Agent stats
        lines.extend([
            "  AGENT STATISTICS",
            thin,
            f"  Rounds:           {self.rounds_used}",
            f"  Total generated:  {self.total_generated}",
            f"  Valid:             {self.total_valid}",
            f"  Less toxic:       {self.total_less_toxic}",
            f"  Time:             {self.total_time_s:.1f}s",
        ])

        if self.strategy_history:
            lines.append(f"  Strategy adjustments:")
            for s in self.strategy_history:
                lines.append(f"    • {s}")

        lines.extend(["", sep])
        return "\n".join(lines)


class DetoxAgent:
    """Agentic detoxification workflow.

    Manages an iterative loop that generates, validates, and refines
    molecular candidates to find a less-toxic, structurally similar
    variant of a given toxic molecule.

    The agent adapts its strategy based on failure analysis:
        - Too few valid candidates → increase temperature
        - Valid but still toxic → try different prefix fraction
        - Low similarity → tighten prefix constraint
        - No improvement after 2 rounds → switch to scratch generation

    Args:
        generator: MoleculeGenerator (IUPAC-GPT policy).
        reward_fn: RewardFunction for scoring candidates.
        resolver: NameResolver for IUPAC → SMILES conversion.
        config: PPOConfig with agent parameters.
    """

    def __init__(
        self,
        generator: MoleculeGenerator,
        reward_fn: RewardFunction,
        resolver: NameResolver,
        config: Optional[PPOConfig] = None,
    ):
        self.generator = generator
        self.reward_fn = reward_fn
        self.resolver = resolver
        self.config = config or PPOConfig()

    def detoxify(
        self,
        seed_iupac: str,
        seed_smiles: Optional[str] = None,
        seed_p_toxic: Optional[float] = None,
    ) -> DetoxReport:
        """Run the full agentic detoxification workflow.

        Args:
            seed_iupac: IUPAC name of the toxic molecule.
            seed_smiles: SMILES of the seed (auto-resolved if None).
            seed_p_toxic: P(toxic) of the seed (auto-predicted if None).

        Returns:
            DetoxReport with best candidate and statistics.
        """
        t0 = time.time()
        report = DetoxReport(seed_iupac=seed_iupac)

        # ── Resolve seed SMILES ───────────────────────────────────────
        if seed_smiles is None:
            seed_smiles = self.resolver.iupac_to_smiles(seed_iupac)
            if seed_smiles is None:
                report.strategy_history.append(
                    "ERROR: Could not resolve seed IUPAC to SMILES"
                )
                report.total_time_s = time.time() - t0
                return report
        report.seed_smiles = seed_smiles

        # ── Get seed toxicity ─────────────────────────────────────────
        if seed_p_toxic is None:
            try:
                pred = self.reward_fn.predictor.predict(
                    seed_iupac, return_attention=False, return_egnn_vector=False
                )
                seed_p_toxic = pred.toxicity_score
                report.seed_severity = pred.severity_label
            except Exception as e:
                logger.warning(f"Failed to predict seed toxicity: {e}")
                seed_p_toxic = 0.5
        report.seed_p_toxic = seed_p_toxic

        if not report.seed_severity:
            import os as _os, sys as _sys
            _p1_path = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                "Phase1-IUPACGPT"
            )
            if _p1_path not in _sys.path:
                _sys.path.insert(0, _p1_path)
            from iupacGPT_finetune.model import score_to_severity_label
            report.seed_severity = score_to_severity_label(seed_p_toxic)

        logger.info(
            f"Starting detoxification: {seed_iupac} | "
            f"P(toxic)={seed_p_toxic:.3f} | SMILES={seed_smiles[:30]}"
        )

        # ══════════════════════════════════════════════════════════════
        #  ANALYST AGENT — Analyze seed molecule
        # ══════════════════════════════════════════════════════════════
        analyst = AnalystAgent()
        analyst_report = analyst.analyze(seed_smiles, seed_iupac)
        logger.info(analyst_report.summary())
        report.strategy_history.append(
            f"[Analyst] {analyst_report.num_toxicophores} toxicophore(s) found. "
            + (", ".join(r for r in analyst_report.recommendations[:2]))
        )

        # ══════════════════════════════════════════════════════════════
        #  VERIFIER & REVIEWER setup
        # ══════════════════════════════════════════════════════════════
        prop_matcher = PropertyMatcher()
        conv_estimator = ConversionEfficiency(timeout_seconds=3)
        verifier = VerifierAgent(
            property_matcher=prop_matcher,
            conversion_estimator=conv_estimator,
            analyst_report=analyst_report,
        )
        reviewer = ReviewerAgent(analyst_report=analyst_report, config=self.config, seed_p_toxic=seed_p_toxic)

        best_candidate = None
        best_reward = float("-inf")

        def _update_best(dc):
            """Two-tier best candidate selection: prefer less-toxic-than-seed."""
            nonlocal best_candidate, best_reward
            is_detox = dc.p_toxic < seed_p_toxic
            best_is_detox = best_candidate is not None and best_candidate.p_toxic < seed_p_toxic

            if best_candidate is None:
                best_reward = dc.reward
                best_candidate = dc
            elif is_detox and not best_is_detox:
                best_reward = dc.reward
                best_candidate = dc
            elif is_detox == best_is_detox and dc.reward > best_reward:
                best_reward = dc.reward
                best_candidate = dc

        def _process_candidate(dc, round_num):
            """Run a candidate through verifier and update report."""
            report.all_candidates.append(dc)
            report.total_generated += 1

            if dc.valid:
                report.total_valid += 1
                report.valid_candidates.append(dc)

                if dc.p_toxic < seed_p_toxic:
                    report.total_less_toxic += 1
                    report.less_toxic_candidates.append(dc)

                _update_best(dc)

        # ══════════════════════════════════════════════════════════════
        #  PHASE A: Scaffold-based detoxification (round 0)
        # ══════════════════════════════════════════════════════════════
        try:
            from .scaffold_detox import ScaffoldDetox
        except ImportError:
            from scaffold_detox import ScaffoldDetox

        scaffold_detox = ScaffoldDetox(
            self.reward_fn.predictor,
            self.resolver,
            self.reward_fn.validator,
        )
        scaffold_candidates = scaffold_detox.generate_replacements(
            smiles=seed_smiles,
            seed_p_toxic=seed_p_toxic,
        )

        scaffold_verifications = []
        if scaffold_candidates:
            logger.info(
                f"Scaffold detox: {len(scaffold_candidates)} candidates "
                f"from toxicophore replacement"
            )

            for sc in scaffold_candidates:
                # Run through VERIFIER
                vr = verifier.verify(
                    candidate_smiles=sc.smiles,
                    candidate_iupac=sc.iupac_name,
                    seed_smiles=seed_smiles,
                    p_toxic=sc.p_toxic,
                    seed_p_toxic=seed_p_toxic,
                    tanimoto=sc.tanimoto,
                )
                scaffold_verifications.append(vr)

                dc = DetoxCandidate(
                    iupac_name=sc.iupac_name,
                    smiles=sc.smiles,
                    p_toxic=sc.p_toxic,
                    reward=sc.reward,
                    tanimoto=sc.tanimoto,
                    qed=sc.qed,
                    sa_score=sc.sa_score,
                    property_score=vr.property_score,
                    conversion_score=vr.conversion_score,
                    delta_toxicity=sc.delta_toxicity,
                    valid=vr.is_valid,
                    verified=vr.passed,
                    toxicophore_removed=vr.toxicophore_removed,
                    round_generated=0,
                    source="scaffold",
                )
                _process_candidate(dc, 0)

            # REVIEWER for scaffold round
            scaffold_review = reviewer.review(
                round_num=0,
                verification_results=scaffold_verifications,
                current_temperature=self.config.temperature,
                current_prefix_fraction=self.config.prefix_fraction,
                current_gen_mode="scaffold",
            )
            logger.info(scaffold_review.summary())
            report.strategy_history.append(
                f"[Reviewer R0] {scaffold_review.passed_candidates}/{len(scaffold_candidates)} "
                f"scaffold candidates verified"
            )
        else:
            logger.info("Scaffold detox: no toxicophores found")
            report.strategy_history.append(
                "[Analyst] No known toxicophores -- using GPT generation"
            )

        # Check if scaffold already found target
        if (best_candidate and best_candidate.p_toxic < self.config.target_toxicity
                and best_candidate.verified and best_candidate.tanimoto > 0.3):
            report.strategy_history.append(
                f"[Early Stop] Scaffold found verified target: "
                f"P={best_candidate.p_toxic:.3f}, Tan={best_candidate.tanimoto:.3f}"
            )
            report.success = True
            report.best_candidate = best_candidate
            report.less_toxic_candidates.sort(key=lambda c: c.reward, reverse=True)
            report.total_time_s = time.time() - t0
            self.resolver.save_cache()
            logger.info(report.summary())
            return report

        # ══════════════════════════════════════════════════════════════
        #  PHASE B: IUPAC-GPT Multi-Agent Loop
        # ══════════════════════════════════════════════════════════════
        logger.info("")
        logger.info("─" * 60)
        logger.info("  MULTI-AGENT DETOXIFICATION LOOP")
        logger.info("  Agents: [1] Analyst  [2] Generator  [3] Verifier  [4] Reviewer")
        logger.info("─" * 60)

        temperature = self.config.temperature
        prefix_fraction = self.config.prefix_fraction
        batch_size = self.config.batch_size
        gen_mode = "prefix"

        # Cross-round dedup: track all generated candidates across rounds
        seen_across_rounds = set()

        for round_num in range(1, self.config.max_agent_rounds + 1):
            report.rounds_used = round_num

            logger.info("")
            logger.info(f"╔══ Round {round_num}/{self.config.max_agent_rounds} " + "═" * 45)
            logger.info(f"║  Settings: temp={temperature:.2f} | prefix={prefix_fraction:.2f} | mode={gen_mode} | batch={batch_size}")

            # ── GENERATOR: Generate candidates ────────────────────────
            logger.info(f"║")
            logger.info(f"║  [2] Generator Agent:  Generating {batch_size} candidates...")
            candidates = self.generator.generate(
                seed_iupac=seed_iupac,
                n=batch_size,
                temperature=temperature,
                prefix_fraction=prefix_fraction,
                mode=gen_mode,
            )

            if not candidates:
                report.strategy_history.append(
                    f"Round {round_num}: 0 candidates generated"
                )
                temperature = min(1.5, temperature + 0.2)
                continue

            # Deduplicate (within round + across all previous rounds)
            unique_candidates = []
            for c in candidates:
                if c not in seen_across_rounds:
                    seen_across_rounds.add(c)
                    unique_candidates.append(c)
            candidates = unique_candidates

            if not candidates:
                report.strategy_history.append(
                    f"Round {round_num}: all candidates were duplicates from prior rounds"
                )
                temperature = min(1.5, temperature + 0.15)
                continue

            # ── SCORER: Score + Verify each candidate ──────────────────
            logger.info(f"║  [3] Verifier Agent:   Evaluating {len(candidates)} candidates...")
            round_verifications = []
            round_valid = 0
            round_less_toxic = 0
            round_verified = 0

            for cand_iupac in candidates:
                # Score with reward function
                reward, info = self.reward_fn.compute(
                    cand_iupac, seed_smiles, seed_p_toxic
                )

                if not info.is_valid:
                    continue

                # Run through VERIFIER
                vr = verifier.verify(
                    candidate_smiles=info.candidate_smiles,
                    candidate_iupac=cand_iupac,
                    seed_smiles=seed_smiles,
                    p_toxic=info.p_toxic,
                    seed_p_toxic=seed_p_toxic,
                    tanimoto=info.tanimoto,
                )
                round_verifications.append(vr)

                dc = DetoxCandidate(
                    iupac_name=cand_iupac,
                    smiles=info.candidate_smiles,
                    p_toxic=info.p_toxic,
                    reward=reward,
                    tanimoto=info.tanimoto,
                    qed=info.qed,
                    sa_score=info.sa_score,
                    property_score=vr.property_score,
                    conversion_score=vr.conversion_score,
                    delta_toxicity=(
                        seed_p_toxic - info.p_toxic if info.p_toxic >= 0 else 0
                    ),
                    valid=info.is_valid,
                    verified=vr.passed,
                    toxicophore_removed=vr.toxicophore_removed,
                    round_generated=round_num,
                    source="gpt",
                )
                _process_candidate(dc, round_num)

                if info.is_valid:
                    round_valid += 1
                if info.p_toxic < seed_p_toxic:
                    round_less_toxic += 1
                if vr.passed:
                    round_verified += 1

            logger.info(
                f"║                        → {len(candidates)} generated | "
                f"{round_valid} valid | {round_less_toxic} less-toxic | "
                f"{round_verified} verified"
            )

            # ── REVIEWER: Analyze results and provide feedback ────────
            review = reviewer.review(
                round_num=round_num,
                verification_results=round_verifications,
                current_temperature=temperature,
                current_prefix_fraction=prefix_fraction,
                current_gen_mode=gen_mode,
            )

            logger.info(f"║  [4] Reviewer Agent:   {review.passed_candidates}/{len(round_verifications)} passed verification")
            if review.feedback_items:
                for fi in review.feedback_items[:2]:
                    logger.info(f"║      → {fi}")
            logger.info(f"╚" + "═" * 59)

            # Apply reviewer's strategy adjustments
            report.strategy_history.append(
                f"[Reviewer R{round_num}] "
                + "; ".join(review.feedback_items[:2])
            )

            if review.recommended_temperature is not None:
                temperature = review.recommended_temperature
            if review.recommended_prefix_fraction is not None:
                prefix_fraction = review.recommended_prefix_fraction
            if review.recommended_gen_mode is not None:
                gen_mode = review.recommended_gen_mode

            # ── Check early stopping ──────────────────────────────────
            if (best_candidate
                    and best_candidate.p_toxic < self.config.target_toxicity
                    and best_candidate.verified):
                report.strategy_history.append(
                    f"[Early Stop R{round_num}] Verified target reached: "
                    f"P={best_candidate.p_toxic:.3f}, Tan={best_candidate.tanimoto:.3f}"
                )
                break

        # ══════════════════════════════════════════════════════════════
        #  FINALIZE REPORT
        # ══════════════════════════════════════════════════════════════
        if best_candidate and best_candidate.p_toxic < seed_p_toxic:
            report.success = True
            report.best_candidate = best_candidate

            # Explanation is now generated by Phase 2/3 handoff
            # Use: detox_dossier.generate_dossier(report) after detoxification
            logger.info(
                f"Best candidate found: {best_candidate.iupac_name} "
                f"P(toxic)={best_candidate.p_toxic:.3f}. "
                f"Run Phase 2/3 handoff for full toxicological dossier."
            )

        report.less_toxic_candidates.sort(key=lambda c: c.reward, reverse=True)

        report.total_time_s = time.time() - t0

        # Save resolver cache
        self.resolver.save_cache()

        logger.info(report.summary())
        return report

    def detoxify_batch(
        self,
        molecules: List[Dict],
    ) -> List[DetoxReport]:
        """Run detoxification for multiple molecules.

        Args:
            molecules: List of dicts with "iupac", optional "smiles", "p_toxic".

        Returns:
            List of DetoxReport objects.
        """
        reports = []
        for i, mol in enumerate(molecules):
            logger.info(f"\n[{i+1}/{len(molecules)}] Processing: {mol['iupac']}")
            report = self.detoxify(
                seed_iupac=mol["iupac"],
                seed_smiles=mol.get("smiles"),
                seed_p_toxic=mol.get("p_toxic"),
            )
            reports.append(report)
        return reports
