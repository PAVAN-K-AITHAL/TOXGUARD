"""Multi-agent orchestrator for Phase 4 detoxification.

Implements a structured multi-agent pipeline inspired by MT-MOL:

    Analyst Agent  -> Identifies toxicophores, computes seed properties
    Generator Agent -> Generates candidates (scaffold detox + IUPAC-GPT)
    Verifier Agent  -> Validates candidates (properties, conversion, toxicophore removal)
    Reviewer Agent  -> Provides structured feedback for next round

The agents are NOT LLM-based (we don't call GPT-4/DeepSeek). Instead,
they are specialized RDKit-powered analysis modules that provide
structured, deterministic feedback. The LLM component is only the
IUPAC-GPT generator.

Usage:
    pipeline = MultiAgentPipeline(generator, reward_fn, resolver, config)
    report = pipeline.run("nitrobenzene")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    from .property_matcher import PropertyMatcher, PropertyComparison
    from .conversion_efficiency import ConversionEfficiency, ConversionResult
    from .scaffold_detox import ScaffoldDetox, TOXICOPHORE_REPLACEMENTS
    from .ppo_config import PPOConfig
except ImportError:
    from property_matcher import PropertyMatcher, PropertyComparison
    from conversion_efficiency import ConversionEfficiency, ConversionResult
    from scaffold_detox import ScaffoldDetox, TOXICOPHORE_REPLACEMENTS
    from ppo_config import PPOConfig


# ══════════════════════════════════════════════════════════════════════
#  ANALYST AGENT
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AnalystReport:
    """Output of the Analyst Agent."""
    seed_smiles: str = ""
    seed_iupac: str = ""

    # Toxicophore analysis
    toxicophores_found: List[Tuple[str, str]] = field(default_factory=list)
    num_toxicophores: int = 0
    suggested_replacements: List[str] = field(default_factory=list)

    # Functional group analysis
    functional_groups: Dict[str, int] = field(default_factory=dict)

    # Property profile
    mw: float = 0.0
    logp: float = 0.0
    hbd: int = 0
    hba: int = 0
    psa: float = 0.0
    num_rings: int = 0
    num_aromatic_rings: int = 0
    rotatable_bonds: int = 0

    # Strategy recommendation
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"[Analyst] Seed: {self.seed_iupac} ({self.seed_smiles})",
            f"  MW={self.mw:.1f} | LogP={self.logp:.2f} | HBD={self.hbd} | HBA={self.hba} | PSA={self.psa:.1f}",
            f"  Rings={self.num_rings} | AromaticRings={self.num_aromatic_rings} | RotBonds={self.rotatable_bonds}",
        ]
        if self.toxicophores_found:
            tox_names = [t[0] for t in self.toxicophores_found]
            lines.append(f"  Toxicophores: {', '.join(tox_names)}")
        else:
            lines.append("  Toxicophores: none found (whole-scaffold toxicity)")

        if self.functional_groups:
            fg_str = ", ".join(f"{k}={v}" for k, v in self.functional_groups.items() if v > 0)
            if fg_str:
                lines.append(f"  Functional groups: {fg_str}")

        for rec in self.recommendations:
            lines.append(f"  >> {rec}")

        return "\n".join(lines)


class AnalystAgent:
    """Analyzes seed molecule to guide generation strategy.

    Identifies toxicophores, functional groups, and physicochemical
    properties. Provides recommendations for the generator.
    """

    # RDKit fragment detectors for key functional groups
    FRAGMENT_CHECKS = [
        ("nitro", lambda m: Fragments.fr_nitro(m)),
        ("amine", lambda m: Fragments.fr_NH2(m) + Fragments.fr_NH1(m)),
        ("aldehyde", lambda m: Fragments.fr_aldehyde(m)),
        ("halide", lambda m: Fragments.fr_halogen(m)),
        ("carboxyl", lambda m: Fragments.fr_COO(m)),
        ("ester", lambda m: Fragments.fr_ester(m)),
        ("amide", lambda m: Fragments.fr_amide(m)),
        ("hydroxyl", lambda m: Fragments.fr_Al_OH(m) + Fragments.fr_Ar_OH(m)),
        ("ether", lambda m: Fragments.fr_ether(m)),
        ("benzene", lambda m: Fragments.fr_benzene(m)),
        ("epoxide", lambda m: Fragments.fr_epoxide(m)),
        ("nitrile", lambda m: Fragments.fr_nitrile(m)),
    ]

    def analyze(self, smiles: str, iupac: str = "") -> AnalystReport:
        """Analyze a seed molecule and produce a structured report.

        Args:
            smiles: SMILES of the seed molecule.
            iupac: IUPAC name of the seed molecule.

        Returns:
            AnalystReport with toxicophore analysis, properties, and
            strategy recommendations.
        """
        report = AnalystReport(seed_smiles=smiles, seed_iupac=iupac)

        if not HAS_RDKIT:
            report.recommendations.append("RDKit unavailable -- limited analysis")
            return report

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            report.recommendations.append("Invalid SMILES -- cannot analyze")
            return report

        # ── Property profile ──────────────────────────────────────
        report.mw = Descriptors.MolWt(mol)
        report.logp = Descriptors.MolLogP(mol)
        report.hbd = rdMolDescriptors.CalcNumHBD(mol)
        report.hba = rdMolDescriptors.CalcNumHBA(mol)
        report.psa = Descriptors.TPSA(mol)
        report.num_rings = rdMolDescriptors.CalcNumRings(mol)
        report.num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        report.rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # ── Functional group analysis ─────────────────────────────
        for fg_name, fg_func in self.FRAGMENT_CHECKS:
            try:
                count = fg_func(mol)
                if count > 0:
                    report.functional_groups[fg_name] = count
            except Exception:
                pass

        # ── Toxicophore detection ─────────────────────────────────
        for tox_name, tox_smarts, replacements in TOXICOPHORE_REPLACEMENTS:
            pattern = Chem.MolFromSmarts(tox_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                report.toxicophores_found.append((tox_name, tox_smarts))
                for repl_name, _ in replacements[:3]:  # Top 3 suggestions
                    report.suggested_replacements.append(
                        f"Replace {tox_name} with {repl_name}"
                    )

        report.num_toxicophores = len(report.toxicophores_found)

        # ── Strategy recommendations ──────────────────────────────
        if report.num_toxicophores > 0:
            tox_names = [t[0] for t in report.toxicophores_found]
            report.recommendations.append(
                f"Found {report.num_toxicophores} toxicophore(s): {', '.join(tox_names)}. "
                f"Scaffold detox should handle these."
            )
        else:
            report.recommendations.append(
                "No known toxicophores found. Toxicity likely from whole scaffold. "
                "Use IUPAC-GPT to generate structural analogs."
            )

        if report.mw < 200:
            report.recommendations.append(
                f"Small molecule (MW={report.mw:.0f}). "
                f"Candidates should target MW {report.mw*0.7:.0f}-{report.mw*1.3:.0f}."
            )
        elif report.mw > 500:
            report.recommendations.append(
                f"Large molecule (MW={report.mw:.0f}). "
                f"Preserve core scaffold, modify periphery only."
            )

        return report


# ══════════════════════════════════════════════════════════════════════
#  VERIFIER AGENT
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VerificationResult:
    """Result of verifying a single candidate."""
    iupac_name: str = ""
    smiles: str = ""
    passed: bool = False

    # Individual checks
    is_valid: bool = False
    toxicophore_removed: bool = False
    properties_preserved: bool = False
    conversion_efficient: bool = False

    # Scores
    p_toxic: float = 1.0
    tanimoto: float = 0.0
    property_score: float = 0.0
    conversion_score: float = 0.0

    # Details
    property_comparison: Optional[PropertyComparison] = None
    conversion_result: Optional[ConversionResult] = None
    fail_reasons: List[str] = field(default_factory=list)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.iupac_name[:40]} | "
            f"P={self.p_toxic:.3f} Tan={self.tanimoto:.3f} "
            f"Prop={self.property_score:.3f} Conv={self.conversion_score:.3f}"
        )


class VerifierAgent:
    """Validates candidate molecules against multiple criteria.

    Checks:
        1. Chemical validity (valid SMILES)
        2. Toxicity reduction (P(toxic) < seed)
        3. Toxicophore removal (if applicable)
        4. Property preservation (MW, LogP, etc.)
        5. Conversion efficiency (MCS-based)

    Candidates must pass ALL checks to be considered valid detoxifications.
    """

    def __init__(
        self,
        property_matcher: PropertyMatcher,
        conversion_estimator: ConversionEfficiency,
        analyst_report: AnalystReport,
        min_property_score: float = 0.4,
        min_conversion_score: float = 0.2,
    ):
        self.property_matcher = property_matcher
        self.conversion_estimator = conversion_estimator
        self.analyst_report = analyst_report
        self.min_property_score = min_property_score
        self.min_conversion_score = min_conversion_score

    def verify(
        self,
        candidate_smiles: str,
        candidate_iupac: str,
        seed_smiles: str,
        p_toxic: float,
        seed_p_toxic: float,
        tanimoto: float,
    ) -> VerificationResult:
        """Verify a single candidate molecule.

        Args:
            candidate_smiles: SMILES of the candidate.
            candidate_iupac: IUPAC name of the candidate.
            seed_smiles: SMILES of the original molecule.
            p_toxic: Predicted toxicity of the candidate.
            seed_p_toxic: Predicted toxicity of the seed.
            tanimoto: Tanimoto similarity to seed.

        Returns:
            VerificationResult with pass/fail status and reasons.
        """
        result = VerificationResult(
            iupac_name=candidate_iupac,
            smiles=candidate_smiles,
            p_toxic=p_toxic,
            tanimoto=tanimoto,
        )

        # Check 1: Valid molecule
        if not candidate_smiles or not HAS_RDKIT:
            result.fail_reasons.append("Invalid SMILES")
            return result
        mol = Chem.MolFromSmiles(candidate_smiles)
        if mol is None:
            result.fail_reasons.append("RDKit cannot parse SMILES")
            return result
        result.is_valid = True

        # Check 2: Toxicity reduced
        if p_toxic >= seed_p_toxic:
            result.fail_reasons.append(
                f"Not less toxic: P={p_toxic:.3f} >= seed P={seed_p_toxic:.3f}"
            )

        # Check 3: Toxicophore removal (if analyst found any)
        if self.analyst_report.toxicophores_found:
            all_removed = True
            for tox_name, tox_smarts in self.analyst_report.toxicophores_found:
                pattern = Chem.MolFromSmarts(tox_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    all_removed = False
                    result.fail_reasons.append(
                        f"Toxicophore '{tox_name}' still present"
                    )
            result.toxicophore_removed = all_removed
        else:
            result.toxicophore_removed = True  # No specific toxicophore to check

        # Check 4: Property preservation
        prop_comp = self.property_matcher.compare(seed_smiles, candidate_smiles)
        result.property_score = prop_comp.overall_score
        result.property_comparison = prop_comp
        result.properties_preserved = prop_comp.overall_score >= self.min_property_score
        if not result.properties_preserved:
            result.fail_reasons.append(
                f"Poor property preservation: {prop_comp.overall_score:.3f} < {self.min_property_score}"
            )

        # Check 5: Conversion efficiency
        conv_result = self.conversion_estimator.compute(seed_smiles, candidate_smiles)
        result.conversion_score = conv_result.conversion_score
        result.conversion_result = conv_result
        result.conversion_efficient = conv_result.conversion_score >= self.min_conversion_score
        if not result.conversion_efficient:
            result.fail_reasons.append(
                f"Poor conversion efficiency: {conv_result.conversion_score:.3f} "
                f"(MCS ratio={conv_result.mcs_ratio:.3f})"
            )

        # Overall pass: valid + less toxic + no critical failures
        result.passed = (
            result.is_valid
            and p_toxic < seed_p_toxic
            and len(result.fail_reasons) == 0
        )

        return result


# ══════════════════════════════════════════════════════════════════════
#  REVIEWER AGENT
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ReviewerFeedback:
    """Structured feedback from the Reviewer Agent."""
    round_num: int = 0
    total_candidates: int = 0
    valid_candidates: int = 0
    passed_candidates: int = 0
    less_toxic_candidates: int = 0

    # Aggregate analysis
    avg_p_toxic: float = 0.0
    avg_tanimoto: float = 0.0
    avg_property_score: float = 0.0
    avg_conversion_score: float = 0.0

    # Common failure reasons
    failure_analysis: Dict[str, int] = field(default_factory=dict)

    # Strategy adjustments
    feedback_items: List[str] = field(default_factory=list)
    recommended_temperature: Optional[float] = None
    recommended_prefix_fraction: Optional[float] = None
    recommended_gen_mode: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"[Reviewer] Round {self.round_num}: "
            f"{self.passed_candidates}/{self.total_candidates} passed verification",
            f"  Avg: P(tox)={self.avg_p_toxic:.3f} | Tan={self.avg_tanimoto:.3f} | "
            f"PropScore={self.avg_property_score:.3f} | ConvScore={self.avg_conversion_score:.3f}",
        ]

        if self.failure_analysis:
            lines.append("  Failure analysis:")
            for reason, count in sorted(
                self.failure_analysis.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {count}x: {reason}")

        for fb in self.feedback_items:
            lines.append(f"  >> {fb}")

        return "\n".join(lines)


class ReviewerAgent:
    """Analyzes verification results and provides structured feedback.

    Reviews all candidates from a round, identifies patterns in
    failures, and recommends strategy adjustments for the next round.

    Tracks round history to avoid repeating the same strategy and
    escalates through increasingly diverse approaches:
        Round 1-2: Adjust prefix fraction and temperature
        Round 3:   Switch generation mode (prefix → scratch)
        Round 4+:  Aggressive diversity (high temp, short prefix)
    """

    def __init__(self, analyst_report: AnalystReport, config: PPOConfig, seed_p_toxic: float = 0.5):
        self.analyst_report = analyst_report
        self.config = config
        self.seed_p_toxic = seed_p_toxic
        # Track history for adaptive feedback
        self._round_history: List[Dict] = []
        self._strategies_tried: List[str] = []

    def review(
        self,
        round_num: int,
        verification_results: List[VerificationResult],
        current_temperature: float,
        current_prefix_fraction: float,
        current_gen_mode: str,
    ) -> ReviewerFeedback:
        """Review a round's results and provide adaptive feedback.

        Uses round history to avoid repeating ineffective strategies.
        Escalates through increasingly diverse generation approaches.

        Args:
            round_num: Current round number.
            verification_results: List of VerificationResult from verifier.
            current_temperature: Current generation temperature.
            current_prefix_fraction: Current prefix fraction.
            current_gen_mode: Current generation mode.

        Returns:
            ReviewerFeedback with analysis and strategy recommendations.
        """
        feedback = ReviewerFeedback(
            round_num=round_num,
            total_candidates=len(verification_results),
        )

        if not verification_results:
            feedback.feedback_items.append(
                "No candidates generated. Switching to scratch mode with high diversity."
            )
            feedback.recommended_temperature = min(1.5, current_temperature + 0.3)
            feedback.recommended_prefix_fraction = max(0.1, current_prefix_fraction - 0.2)
            feedback.recommended_gen_mode = "scratch"
            self._strategies_tried.append("empty_to_scratch")
            return feedback

        # ── Aggregate statistics ──────────────────────────────────
        valid = [r for r in verification_results if r.is_valid]
        passed = [r for r in verification_results if r.passed]
        less_toxic = [r for r in valid if r.p_toxic < 0.5]

        feedback.valid_candidates = len(valid)
        feedback.passed_candidates = len(passed)
        feedback.less_toxic_candidates = len(less_toxic)

        if valid:
            feedback.avg_p_toxic = sum(r.p_toxic for r in valid) / len(valid)
            feedback.avg_tanimoto = sum(r.tanimoto for r in valid) / len(valid)
            feedback.avg_property_score = sum(r.property_score for r in valid) / len(valid)
            feedback.avg_conversion_score = sum(r.conversion_score for r in valid) / len(valid)

        # ── Failure analysis ──────────────────────────────────────
        for r in verification_results:
            for reason in r.fail_reasons:
                if "toxicophore" in reason.lower():
                    key = "Toxicophore still present"
                elif "property" in reason.lower():
                    key = "Poor property preservation"
                elif "conversion" in reason.lower():
                    key = "Poor conversion efficiency"
                elif "not less toxic" in reason.lower():
                    key = "Still toxic"
                else:
                    key = reason[:50]
                feedback.failure_analysis[key] = feedback.failure_analysis.get(key, 0) + 1

        # Save round snapshot for history tracking
        round_snapshot = {
            "round": round_num,
            "n_valid": len(valid),
            "n_passed": len(passed),
            "n_less_toxic": len(less_toxic),
            "avg_prop": feedback.avg_property_score,
            "avg_conv": feedback.avg_conversion_score,
            "avg_tox": feedback.avg_p_toxic,
            "temperature": current_temperature,
            "prefix_fraction": current_prefix_fraction,
            "gen_mode": current_gen_mode,
        }
        self._round_history.append(round_snapshot)

        # ── Determine dominant issue ──────────────────────────────
        has_property_drift = feedback.avg_property_score < 0.5 and len(valid) > 0
        has_low_conversion = feedback.avg_conversion_score < 0.3 and len(valid) > 0
        has_toxicophore = feedback.failure_analysis.get("Toxicophore still present", 0) > len(verification_results) * 0.3
        all_still_toxic = feedback.failure_analysis.get("Still toxic", 0) > len(verification_results) * 0.7
        low_validity = len(valid) < len(verification_results) * 0.4
        has_passed = len(passed) > 0

        # Check if metrics are improving across rounds
        improving = False
        if len(self._round_history) >= 2:
            prev = self._round_history[-2]
            curr = round_snapshot
            improving = (
                curr["n_passed"] > prev["n_passed"]
                or curr["avg_prop"] > prev["avg_prop"] + 0.05
                or curr["avg_conv"] > prev["avg_conv"] + 0.05
            )

        # ── Adaptive strategy based on round number ───────────────
        # Early rounds (1-2): Standard adjustments
        # Mid rounds (3): Mode switch if not improving
        # Late rounds (4+): Aggressive diversity

        if has_toxicophore:
            feedback.feedback_items.append(
                f"Toxicophore persists in {feedback.failure_analysis['Toxicophore still present']}"
                f"/{len(verification_results)} candidates. "
                f"Shortening prefix to force divergence from toxic substructure."
            )
            feedback.recommended_prefix_fraction = max(0.1, current_prefix_fraction - 0.15)
            self._strategies_tried.append("reduce_prefix_toxicophore")

        elif has_passed and improving:
            # Things are getting better — refine around current parameters
            best = max(passed, key=lambda r: r.conversion_score + (1 - r.p_toxic))
            feedback.feedback_items.append(
                f"Improvement detected. Best verified: P(tox)={best.p_toxic:.3f}, "
                f"Prop={best.property_score:.3f}, Conv={best.conversion_score:.3f}. "
                f"Fine-tuning temperature for more precise exploration."
            )
            feedback.recommended_temperature = max(0.5, current_temperature - 0.05)
            self._strategies_tried.append("refine_best")

        elif round_num <= 2 and (has_property_drift or has_low_conversion):
            # Early rounds: try prefix adjustment
            if "increase_prefix" not in self._strategies_tried:
                seed_mw = self.analyst_report.mw
                feedback.feedback_items.append(
                    f"Structural divergence from seed (MW={seed_mw:.0f}): "
                    f"PropScore={feedback.avg_property_score:.2f}, "
                    f"ConvScore={feedback.avg_conversion_score:.2f}. "
                    f"Increasing prefix fraction to anchor generation closer to seed scaffold."
                )
                feedback.recommended_prefix_fraction = min(0.8, current_prefix_fraction + 0.15)
                self._strategies_tried.append("increase_prefix")
            else:
                feedback.feedback_items.append(
                    f"Prefix adjustment insufficient (Prop={feedback.avg_property_score:.2f}). "
                    f"Reducing temperature for more conservative generation."
                )
                feedback.recommended_temperature = max(0.6, current_temperature - 0.15)
                self._strategies_tried.append("reduce_temp_conservative")

        elif round_num == 3 and not improving:
            # Mid round: switch generation mode
            if current_gen_mode == "prefix":
                feedback.feedback_items.append(
                    f"Prefix-conditioned generation plateaued after {round_num} rounds "
                    f"(Prop={feedback.avg_property_score:.2f}, Conv={feedback.avg_conversion_score:.2f}). "
                    f"Switching to scratch mode for unconstrained molecular exploration."
                )
                feedback.recommended_gen_mode = "scratch"
                feedback.recommended_temperature = 0.85
                feedback.recommended_prefix_fraction = current_prefix_fraction
                self._strategies_tried.append("switch_to_scratch")
            else:
                feedback.feedback_items.append(
                    f"Scratch mode yielded limited results. "
                    f"Returning to prefix mode with aggressive temperature boost."
                )
                feedback.recommended_gen_mode = "prefix"
                feedback.recommended_temperature = min(1.3, current_temperature + 0.2)
                feedback.recommended_prefix_fraction = 0.3
                self._strategies_tried.append("scratch_to_prefix_hot")

        elif round_num >= 4:
            # Late rounds: aggressive diversity strategies
            if "aggressive_diversity" not in self._strategies_tried:
                feedback.feedback_items.append(
                    f"Round {round_num}: Exhausting standard strategies. "
                    f"Applying aggressive diversity — low prefix (0.2), elevated temperature. "
                    f"Prioritizing novel scaffolds over structural preservation."
                )
                feedback.recommended_temperature = min(1.3, current_temperature + 0.15)
                feedback.recommended_prefix_fraction = 0.2
                feedback.recommended_gen_mode = "prefix"
                self._strategies_tried.append("aggressive_diversity")
            else:
                # Final attempt: flip mode
                new_mode = "scratch" if current_gen_mode == "prefix" else "prefix"
                feedback.feedback_items.append(
                    f"Round {round_num}: Final strategy rotation. "
                    f"Switching to {new_mode} mode with balanced parameters "
                    f"(temp=0.9, prefix=0.4) for last-chance exploration."
                )
                feedback.recommended_gen_mode = new_mode
                feedback.recommended_temperature = 0.9
                feedback.recommended_prefix_fraction = 0.4
                self._strategies_tried.append("final_rotation")

        elif all_still_toxic:
            feedback.feedback_items.append(
                f"Majority of candidates remain toxic (avg P={feedback.avg_p_toxic:.3f}). "
                f"Increasing diversity with higher temperature and shorter prefix."
            )
            feedback.recommended_temperature = min(1.3, current_temperature + 0.1)
            feedback.recommended_prefix_fraction = max(0.2, current_prefix_fraction - 0.1)
            self._strategies_tried.append("diversity_for_toxicity")

        elif low_validity:
            feedback.feedback_items.append(
                f"Low validity rate ({len(valid)}/{len(verification_results)}). "
                f"Model generating malformed IUPAC names. Reducing temperature for coherence."
            )
            feedback.recommended_temperature = max(0.5, current_temperature - 0.15)
            self._strategies_tried.append("reduce_temp_validity")

        elif has_passed:
            best = max(passed, key=lambda r: r.conversion_score + (1 - r.p_toxic))
            feedback.feedback_items.append(
                f"Found {len(passed)} verified candidate(s). Best: P(tox)={best.p_toxic:.3f}, "
                f"Prop={best.property_score:.3f}, Conv={best.conversion_score:.3f}. "
                f"Narrowing search around this chemical space."
            )
            feedback.recommended_temperature = max(0.5, current_temperature - 0.1)
            self._strategies_tried.append("narrow_search")

        # Default
        if not feedback.feedback_items:
            feedback.feedback_items.append(
                f"Round {round_num}: Mixed signals. Applying moderate parameter perturbation."
            )

        return feedback
