"""Multi-objective reward function for Phase 4 PPO training.

The reward combines four signals:

    R(molecule) = w₁ × (1 - P(toxic))        ← detoxification
               + w₂ × Tanimoto(mol, seed)     ← structural similarity
               + w₃ × QED(mol)                ← drug-likeness
               + w₄ × SA_normalized(mol)      ← synthesizability
               + validity_bonus               ← +0.5 if valid, -1.0 if invalid

Default weights: (0.40, 0.25, 0.20, 0.15) — prioritises toxicity reduction.

The reward model uses Phase 1 ToxGuardPredictor (frozen) as the toxicity
scoring oracle. RDKit computes molecular properties.

Usage:
    reward_fn = RewardFunction(
        toxguard_predictor=predictor,
        name_resolver=resolver,
        molecule_validator=validator,
    )
    reward, info = reward_fn.compute(
        candidate_iupac="aniline",
        seed_smiles="[O-][N+](=O)c1ccccc1",
    )
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    from .ppo_config import PPOConfig
    from .molecule_validator import MoleculeValidator, ValidationResult
    from .name_resolver import NameResolver
    from .property_matcher import PropertyMatcher
    from .conversion_efficiency import ConversionEfficiency
except ImportError:
    from ppo_config import PPOConfig
    from molecule_validator import MoleculeValidator, ValidationResult
    from name_resolver import NameResolver
    from property_matcher import PropertyMatcher
    from conversion_efficiency import ConversionEfficiency


@dataclass
class RewardInfo:
    """Detailed breakdown of reward components."""
    total_reward: float = 0.0
    detox_reward: float = 0.0
    similarity_reward: float = 0.0
    qed_reward: float = 0.0
    sa_reward: float = 0.0
    property_reward: float = 0.0       # NEW
    conversion_reward: float = 0.0     # NEW
    validity_bonus: float = 0.0

    # Resolution/validation details
    candidate_iupac: str = ""
    candidate_smiles: str = ""
    seed_smiles: str = ""
    p_toxic: float = -1.0
    tanimoto: float = 0.0
    qed: float = 0.0
    sa_score: float = 0.0
    property_score: float = 0.0        # NEW
    conversion_score: float = 0.0      # NEW
    is_valid: bool = False
    error: str = ""

    def __repr__(self):
        if not self.is_valid:
            return f"RewardInfo(R={self.total_reward:.3f}, valid=False, error={self.error!r})"
        return (
            f"RewardInfo(R={self.total_reward:.3f}, "
            f"P(toxic)={self.p_toxic:.3f}, "
            f"tanimoto={self.tanimoto:.3f}, "
            f"qed={self.qed:.3f}, sa={self.sa_score:.3f})"
        )


class RewardFunction:
    """Multi-objective reward function for PPO molecule detoxification.

    Combines toxicity reduction, structural similarity, drug-likeness,
    and synthetic accessibility into a single scalar reward.

    Args:
        toxguard_predictor: Phase 1 ToxGuardPredictor (frozen) for P(toxic).
        name_resolver: NameResolver for IUPAC → SMILES conversion.
        molecule_validator: MoleculeValidator for property computation.
        config: PPOConfig with reward weights.
    """

    def __init__(
        self,
        toxguard_predictor,
        name_resolver: NameResolver,
        molecule_validator: MoleculeValidator,
        config: Optional[PPOConfig] = None,
    ):
        self.predictor = toxguard_predictor
        self.resolver = name_resolver
        self.validator = molecule_validator
        self.config = config or PPOConfig()
        self.property_matcher = PropertyMatcher()
        self.conversion_estimator = ConversionEfficiency(timeout_seconds=3)

    def compute(
        self,
        candidate_iupac: str,
        seed_smiles: str,
        seed_p_toxic: Optional[float] = None,
    ) -> Tuple[float, RewardInfo]:
        """Compute reward for a generated candidate molecule.

        Args:
            candidate_iupac: Generated IUPAC name from the policy.
            seed_smiles: SMILES of the original toxic molecule.
            seed_p_toxic: P(toxic) of the seed (for relative comparison).

        Returns:
            (reward_scalar, reward_info) tuple.
        """
        info = RewardInfo(
            candidate_iupac=candidate_iupac,
            seed_smiles=seed_smiles,
        )

        w_detox = self.config.reward_detox_weight
        w_sim = self.config.reward_similarity_weight
        w_qed = self.config.reward_qed_weight
        w_sa = self.config.reward_sa_weight
        w_prop = self.config.reward_property_weight
        w_conv = self.config.reward_conversion_weight

        # ── Step 1: Convert IUPAC → SMILES ────────────────────────────
        candidate_smiles = self.resolver.iupac_to_smiles(candidate_iupac)
        if candidate_smiles is None:
            info.error = "IUPAC -> SMILES resolution failed"
            info.is_valid = False
            info.validity_bonus = self.config.invalid_penalty
            info.total_reward = self.config.invalid_penalty
            return info.total_reward, info

        info.candidate_smiles = candidate_smiles

        # ── Step 2: Validate molecule with RDKit ──────────────────────
        validation = self.validator.validate_candidate(
            candidate_smiles=candidate_smiles,
            seed_smiles=seed_smiles,
        )
        if not validation.valid:
            info.error = validation.error
            info.is_valid = False
            info.validity_bonus = self.config.invalid_penalty
            info.total_reward = self.config.invalid_penalty
            return info.total_reward, info

        info.is_valid = True
        info.validity_bonus = self.config.validity_bonus
        info.tanimoto = validation.tanimoto
        info.qed = validation.qed
        info.sa_score = validation.sa_score

        # ── Step 3: Get toxicity score from Phase 1 ───────────────────
        try:
            prediction = self.predictor.predict(
                candidate_iupac,
                return_attention=False,
                return_egnn_vector=False,
            )
            info.p_toxic = prediction.toxicity_score
        except Exception as e:
            logger.warning(f"ToxGuard prediction failed for '{candidate_iupac}': {e}")
            # Use a neutral score if prediction fails
            info.p_toxic = 0.5

        # ── Step 4: Compute component rewards ─────────────────────────

        # 4a. Detoxification reward: lower P(toxic) = higher reward
        info.detox_reward = 1.0 - info.p_toxic

        # 4b. Structural similarity reward
        info.similarity_reward = info.tanimoto

        # 4c. Drug-likeness reward
        info.qed_reward = info.qed

        # 4d. Synthetic accessibility reward
        info.sa_reward = info.sa_score

        # 4e. Property preservation reward (NEW)
        prop_comp = self.property_matcher.compare(seed_smiles, candidate_smiles)
        info.property_score = prop_comp.overall_score
        info.property_reward = prop_comp.overall_score

        # 4f. Conversion efficiency reward (NEW)
        conv_result = self.conversion_estimator.compute(seed_smiles, candidate_smiles)
        info.conversion_score = conv_result.conversion_score
        info.conversion_reward = conv_result.conversion_score

        # ── Step 5: Weighted combination ──────────────────────────────
        info.total_reward = (
            w_detox * info.detox_reward
            + w_sim * info.similarity_reward
            + w_qed * info.qed_reward
            + w_sa * info.sa_reward
            + w_prop * info.property_reward
            + w_conv * info.conversion_reward
            + info.validity_bonus
        )

        # ── Step 6: Tanimoto floor penalty ────────────────────────────
        # If structural similarity is too low, the molecule isn't a
        # meaningful "detoxification" — it's just a random safe molecule.
        # Penalize candidates below the min_tanimoto threshold.
        if info.tanimoto < self.config.min_tanimoto:
            # Linear penalty: 0 at threshold, -0.5 at tanimoto=0
            penalty = -0.5 * (1.0 - info.tanimoto / self.config.min_tanimoto)
            info.total_reward += penalty

        # ── Step 7: Relative toxicity penalty ─────────────────────────
        # Penalize only if candidate is MORE toxic than the seed.
        # Candidates above 0.5 but below seed are still valid detoxifications.
        if seed_p_toxic is not None and info.p_toxic > seed_p_toxic:
            tox_penalty = -0.5 * (info.p_toxic - seed_p_toxic)
            info.total_reward += tox_penalty

        return info.total_reward, info

    def compute_batch(
        self,
        candidate_iupacs: list,
        seed_smiles: str,
        seed_p_toxic: Optional[float] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Compute rewards for a batch of candidates.

        Args:
            candidate_iupacs: List of generated IUPAC names.
            seed_smiles: SMILES of the seed molecule.
            seed_p_toxic: P(toxic) of the seed.

        Returns:
            (rewards_tensor, infos_list)
        """
        rewards = []
        infos = []
        for iupac in candidate_iupacs:
            r, info = self.compute(iupac, seed_smiles, seed_p_toxic)
            rewards.append(r)
            infos.append(info)

        return torch.tensor(rewards, dtype=torch.float32), infos
