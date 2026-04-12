"""Chain-of-Thought toxicity analyzer — main pipeline for Phase 2.

Orchestrates:
    1. Phase 1 ToxGuard prediction (P(toxic) + attention + toxicophores)
    2. Few-shot CoT prompt construction
    3. LLM inference via Groq API
    4. Structured output parsing

Usage:
    from Phase2_CoT import CoTAnalyzer, GroqLLMClient

    llm = GroqLLMClient(api_key="gsk_...")
    analyzer = CoTAnalyzer(llm_client=llm, checkpoint_dir="iupacGPT/iupac-gpt/checkpoints/iupac")
    result = analyzer.analyze("nitrobenzene")
    print(result.summary())
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CoTResult:
    """Complete result from Chain-of-Thought toxicity analysis."""

    # Phase 1 outputs
    iupac_name: str
    toxicity_score: float
    severity_label: str
    is_toxic: bool
    top_tokens: List[dict]
    toxicophore_hits: List[dict]

    # Phase 2 CoT outputs
    structural_analysis: str = ""
    toxicophore_identification: str = ""
    mechanism_of_action: str = ""
    biological_pathways: str = ""
    organ_toxicity: str = ""
    confidence: str = ""
    verdict: str = ""

    # Extracted features
    functional_groups: List[str] = field(default_factory=list)
    metabolite_groups: List[str] = field(default_factory=list)
    confidence_level: str = "UNKNOWN"  # HIGH / MEDIUM / LOW

    # Metadata
    llm_model: str = ""
    llm_latency_ms: float = 0.0
    raw_response: str = ""

    def summary(self) -> str:
        """One-line summary of the CoT analysis."""
        toxic_str = "TOXIC" if self.is_toxic else "Non-toxic"
        groups = ", ".join(self.functional_groups[:3]) if self.functional_groups else "none identified"
        return (
            f"{self.iupac_name}: {toxic_str} "
            f"(P={self.toxicity_score:.3f}, {self.severity_label}) | "
            f"Functional groups: {groups} | "
            f"Confidence: {self.confidence_level}"
        )

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "iupac_name": self.iupac_name,
            "toxicity_score": self.toxicity_score,
            "severity_label": self.severity_label,
            "is_toxic": self.is_toxic,
            "top_tokens": self.top_tokens,
            "toxicophore_hits": self.toxicophore_hits,
            "structural_analysis": self.structural_analysis,
            "toxicophore_identification": self.toxicophore_identification,
            "mechanism_of_action": self.mechanism_of_action,
            "biological_pathways": self.biological_pathways,
            "organ_toxicity": self.organ_toxicity,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "functional_groups": self.functional_groups,
            "metabolite_groups": self.metabolite_groups,
            "confidence_level": self.confidence_level,
            "llm_model": self.llm_model,
            "llm_latency_ms": self.llm_latency_ms,
        }

    def detailed_report(self) -> str:
        """Full multi-line report for display / export."""
        separator = "=" * 70
        lines = [
            separator,
            f"  CoT TOXICITY ANALYSIS: {self.iupac_name}",
            separator,
            "",
            f"  Phase 1 Prediction:  P(toxic) = {self.toxicity_score:.3f}",
            f"  Severity:            {self.severity_label}",
            f"  Classification:      {'TOXIC' if self.is_toxic else 'NON-TOXIC'}",
            "",
        ]

        if self.top_tokens:
            toks = ", ".join(
                f"{t.get('token', '?')} ({t.get('score', 0):.2f})"
                for t in self.top_tokens[:5]
            )
            lines.append(f"  Top Attention Tokens: {toks}")

        if self.toxicophore_hits:
            hits = ", ".join(
                f"{h.get('pattern', '?')} in '{h.get('fragment', '')}'"
                for h in self.toxicophore_hits[:3]
            )
            lines.append(f"  Toxicophore Matches:  {hits}")

        lines.append("")
        lines.append("-" * 70)
        lines.append("  CHAIN-OF-THOUGHT ANALYSIS")
        lines.append("-" * 70)

        sections = [
            ("1. STRUCTURAL ANALYSIS", self.structural_analysis),
            ("2. TOXICOPHORE IDENTIFICATION", self.toxicophore_identification),
            ("3. MECHANISM OF ACTION", self.mechanism_of_action),
            ("4. BIOLOGICAL PATHWAYS", self.biological_pathways),
            ("5. ORGAN TOXICITY", self.organ_toxicity),
            ("6. CONFIDENCE", self.confidence),
            ("7. VERDICT", self.verdict),
        ]

        for title, content in sections:
            lines.append("")
            lines.append(f"  {title}:")
            if content:
                for line in content.split("\n"):
                    lines.append(f"    {line}")
            else:
                lines.append("    [Not available]")

        if self.functional_groups:
            lines.append("")
            lines.append(f"  Identified Functional Groups: {', '.join(self.functional_groups)}")

        lines.append("")
        lines.append(separator)
        lines.append(f"  LLM: {self.llm_model} | Latency: {self.llm_latency_ms:.0f}ms")
        lines.append(separator)

        return "\n".join(lines)


class CoTAnalyzer:
    """Chain-of-Thought toxicity analyzer pipeline.

    Connects Phase 1 (ToxGuard prediction) with Phase 2 (LLM reasoning)
    to provide mechanistic toxicity explanations.

    Args:
        llm_client: An LLM client instance (GroqLLMClient).
        predictor: Optional ToxGuardPredictor instance. If not provided,
            you must pass Phase 1 results directly via analyze_from_prediction().
        checkpoint_dir: Path to IUPAC-GPT checkpoint (for auto-loading predictor).
        lora_weights_path: Path to trained LoRA weights (for auto-loading predictor).
        temperature: LLM sampling temperature (0.0 - 1.0).
        max_tokens: Max tokens for LLM response.
        num_exemplars: Number of few-shot examples to include.
    """

    def __init__(
        self,
        llm_client,
        predictor=None,
        checkpoint_dir: Optional[str] = None,
        lora_weights_path: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        num_exemplars: int = 3,
    ):
        self.llm = llm_client
        self.predictor = predictor
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_exemplars = num_exemplars

        # Auto-load predictor if paths provided but no predictor given
        if self.predictor is None and checkpoint_dir is not None:
            self.predictor = self._load_predictor(
                checkpoint_dir, lora_weights_path
            )

    def _load_predictor(self, checkpoint_dir, lora_weights_path=None):
        """Load ToxGuard predictor from checkpoint."""
        # Add Phase1 to path if needed
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        phase1_path = os.path.join(project_root, "Phase1-IUPACGPT")
        if phase1_path not in sys.path:
            sys.path.insert(0, phase1_path)

        try:
            from iupacGPT_finetune.inference import ToxGuardPredictor
            
            # The actual tokenizer path
            spm_path = os.path.join(project_root, "iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model")
            
            predictor = ToxGuardPredictor.from_checkpoint(
                checkpoint_dir=checkpoint_dir,
                lora_weights_path=lora_weights_path,
                tokenizer_path=spm_path,
                device="cpu",  # Intel Arc — use CPU
            )
            logger.info("Loaded ToxGuard predictor for Phase 2 CoT")
            return predictor
        except Exception as e:
            logger.warning(
                f"Could not load ToxGuard predictor: {e}. "
                "Use analyze_from_prediction() to pass Phase 1 results directly."
            )
            return None

    def analyze(
        self,
        iupac_name: str,
        return_raw: bool = False,
    ) -> CoTResult:
        """Full pipeline: Phase 1 prediction → CoT reasoning.

        Args:
            iupac_name: IUPAC name of the molecule.
            return_raw: If True, include raw LLM response in result.

        Returns:
            CoTResult with prediction + mechanistic analysis.
        """
        if self.predictor is None:
            raise RuntimeError(
                "No ToxGuard predictor loaded. Either:\n"
                "1. Pass predictor= in constructor\n"
                "2. Pass checkpoint_dir= to auto-load\n"
                "3. Use analyze_from_prediction() with manual Phase 1 results"
            )

        # Step 1: Phase 1 prediction
        prediction = self.predictor.predict(
            iupac_name,
            return_attention=True,
            attention_top_k=10,
        )

        # Step 2: Run CoT reasoning
        return self.analyze_from_prediction(
            iupac_name=iupac_name,
            toxicity_score=prediction.toxicity_score,
            severity_label=prediction.severity_label,
            is_toxic=prediction.is_toxic,
            top_tokens=prediction.top_tokens or [],
            toxicophore_hits=prediction.toxicophore_hits or [],
            return_raw=return_raw,
        )

    def analyze_from_prediction(
        self,
        iupac_name: str,
        toxicity_score: float,
        severity_label: str,
        is_toxic: bool,
        top_tokens: Optional[List[dict]] = None,
        toxicophore_hits: Optional[List[dict]] = None,
        return_raw: bool = False,
    ) -> CoTResult:
        """Run CoT reasoning given pre-computed Phase 1 results.

        Use this when you already have Phase 1 outputs or when the
        ToxGuard predictor is not available in the current environment.

        Args:
            iupac_name: IUPAC name.
            toxicity_score: P(toxic) from Phase 1.
            severity_label: Severity string from Phase 1.
            is_toxic: Binary classification from Phase 1.
            top_tokens: List of {token, score} dicts from attention.
            toxicophore_hits: List of {pattern, fragment, score} dicts.
            return_raw: Include raw LLM response text.

        Returns:
            CoTResult with CoT analysis.
        """
        try:
            from .prompts import (
                SYSTEM_PROMPT,
                build_few_shot_prompt,
                parse_cot_response,
                extract_functional_groups,
                extract_metabolite_groups,
                extract_confidence_level,
            )
        except ImportError:
            from prompts import (
                SYSTEM_PROMPT,
                build_few_shot_prompt,
                parse_cot_response,
                extract_functional_groups,
                extract_metabolite_groups,
                extract_confidence_level,
            )

        top_tokens = top_tokens or []
        toxicophore_hits = toxicophore_hits or []

        # Format attention tokens for prompt
        tokens_str = ", ".join(
            t.get("token", "?") for t in top_tokens[:5]
        ) if top_tokens else "None"

        # Format toxicophore hits for prompt
        hits_str = ", ".join(
            f"{h.get('pattern', '?')} (score={h.get('score', 0):.2f})"
            for h in toxicophore_hits[:3]
        ) if toxicophore_hits else "None"

        # Build few-shot prompt
        user_prompt = build_few_shot_prompt(
            iupac_name=iupac_name,
            toxicity_score=toxicity_score,
            severity_label=severity_label,
            top_tokens=tokens_str,
            toxicophore_hits=hits_str,
            num_exemplars=self.num_exemplars,
        )

        # Query LLM
        logger.info(f"Running CoT analysis for: {iupac_name}")
        response = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse structured output
        sections = parse_cot_response(response.content)
        functional_groups = extract_functional_groups(sections)
        metabolite_groups = extract_metabolite_groups(sections)
        confidence_level = extract_confidence_level(sections)

        result = CoTResult(
            # Phase 1
            iupac_name=iupac_name,
            toxicity_score=toxicity_score,
            severity_label=severity_label,
            is_toxic=is_toxic,
            top_tokens=top_tokens,
            toxicophore_hits=toxicophore_hits,
            # Phase 2 CoT
            structural_analysis=sections.get("STRUCTURAL ANALYSIS", ""),
            toxicophore_identification=sections.get("TOXICOPHORE IDENTIFICATION", ""),
            mechanism_of_action=sections.get("MECHANISM OF ACTION", ""),
            biological_pathways=sections.get("BIOLOGICAL PATHWAYS", ""),
            organ_toxicity=sections.get("ORGAN TOXICITY", ""),
            confidence=sections.get("CONFIDENCE", ""),
            verdict=sections.get("VERDICT", ""),
            # Extracted
            functional_groups=functional_groups,
            metabolite_groups=metabolite_groups,
            confidence_level=confidence_level,
            # Metadata
            llm_model=response.model,
            llm_latency_ms=response.latency_ms,
            raw_response=response.content if return_raw else "",
        )

        logger.info(f"CoT analysis complete: {result.summary()}")
        return result

    def analyze_batch(
        self,
        iupac_names: List[str],
        delay_between: float = 1.0,
        save_path: Optional[str] = None,
    ) -> List[CoTResult]:
        """Analyze multiple molecules with rate-limiting.

        Args:
            iupac_names: List of IUPAC names.
            delay_between: Seconds to wait between API calls (rate limiting).
            save_path: If provided, save results to this JSON file
                incrementally (crash-safe).

        Returns:
            List of CoTResult objects.
        """
        results = []

        # Load existing results if resuming
        existing = {}
        if save_path and os.path.exists(save_path):
            try:
                with open(save_path, "r") as f:
                    existing_list = json.load(f)
                existing = {r["iupac_name"]: r for r in existing_list}
                logger.info(f"Resuming from {len(existing)} existing results")
            except Exception:
                pass

        for i, name in enumerate(iupac_names):
            # Skip already-analyzed molecules
            if name in existing:
                logger.info(f"[{i+1}/{len(iupac_names)}] Skipping {name} (cached)")
                # Reconstruct CoTResult from cached dict
                cached = existing[name]
                results.append(CoTResult(**{
                    k: v for k, v in cached.items()
                    if k in CoTResult.__dataclass_fields__
                }))
                continue

            logger.info(f"[{i+1}/{len(iupac_names)}] Analyzing: {name}")
            try:
                if self.predictor:
                    result = self.analyze(name)
                else:
                    # Without predictor, use defaults
                    result = self.analyze_from_prediction(
                        iupac_name=name,
                        toxicity_score=0.5,
                        severity_label="Unknown",
                        is_toxic=False,
                    )
                results.append(result)

                # Save incrementally
                if save_path:
                    all_results = [r.to_dict() for r in results]
                    # Include any cached results not yet re-analyzed
                    for cached_name, cached_data in existing.items():
                        if cached_name not in {r.iupac_name for r in results}:
                            all_results.append(cached_data)
                    with open(save_path, "w") as f:
                        json.dump(all_results, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to analyze {name}: {e}")
                results.append(CoTResult(
                    iupac_name=name,
                    toxicity_score=0.0,
                    severity_label="Error",
                    is_toxic=False,
                    top_tokens=[],
                    toxicophore_hits=[],
                    verdict=f"ERROR: {e}",
                ))

            # Rate limiting
            if i < len(iupac_names) - 1:
                time.sleep(delay_between)

        return results
