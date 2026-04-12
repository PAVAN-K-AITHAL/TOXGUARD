"""PPO hyperparameters and reward weights for Phase 4 RL training.

Configured for RTX 3060 (12GB VRAM) with IUPAC-GPT (7.1M params).
The model is small enough that policy + reference + reward all fit
comfortably in VRAM.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    """PPO training configuration.

    Attributes:
        # ── LoRA ──
        lora_rank: LoRA adapter rank for the policy network (new adapters,
            separate from Phase 1's rank-32 adapters).
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout for LoRA adapters.
        lora_target_modules: Which GPT-2 modules to apply LoRA to.

        # ── PPO ──
        clip_epsilon: PPO clipping parameter.
        kl_coeff: KL divergence coefficient (penalty for straying from
            reference model).
        kl_target: Target KL divergence. If exceeded, kl_coeff is
            dynamically increased.
        gamma: Discount factor for returns (usually 1.0 for single-step).
        gae_lambda: GAE lambda for advantage estimation.

        # ── Training ──
        learning_rate: LoRA adapter learning rate.
        num_epochs_per_batch: PPO epochs per batch of candidates.
        batch_size: Number of candidate molecules per seed.
        mini_batch_size: Mini-batch size within PPO update.
        max_grad_norm: Gradient clipping norm.
        total_train_steps: Total PPO training iterations.

        # ── Generation ──
        temperature: Sampling temperature for IUPAC-GPT generation.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter (0 = disabled).
        max_gen_length: Maximum generated IUPAC name length (tokens).
        prefix_fraction: Fraction of seed tokens to keep as prefix.

        # ── Reward ──
        reward_detox_weight: Weight for (1 - P(toxic)) reward.
        reward_similarity_weight: Weight for Tanimoto similarity.
        reward_qed_weight: Weight for QED drug-likeness.
        reward_sa_weight: Weight for synthetic accessibility.
        validity_bonus: Bonus for valid molecules.
        invalid_penalty: Penalty for invalid molecules.

        # ── Agent ──
        max_agent_rounds: Maximum agentic refinement rounds.
        target_toxicity: Target P(toxic) for early stopping.
        min_tanimoto: Minimum structural similarity to seed.
    """

    # ── LoRA ──────────────────────────────────────────────────────────
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "c_fc"]
    )

    # ── PPO ───────────────────────────────────────────────────────────
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.05
    kl_target: float = 6.0
    gamma: float = 1.0
    gae_lambda: float = 0.95

    # ── Training ─────────────────────────────────────────────────────
    learning_rate: float = 1e-5
    num_epochs_per_batch: int = 4
    batch_size: int = 16
    mini_batch_size: int = 4
    max_grad_norm: float = 1.0
    total_train_steps: int = 500
    warmup_steps: int = 50

    # ── Generation ───────────────────────────────────────────────────
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 0  # disabled by default — use top_p only
    max_gen_length: int = 128
    prefix_fraction: float = 0.5

    # ── Reward ───────────────────────────────────────────────────────
    reward_detox_weight: float = 0.10          # reduced: any reduction is good
    reward_similarity_weight: float = 0.20     # increased: structural similarity matters
    reward_qed_weight: float = 0.10
    reward_sa_weight: float = 0.05
    reward_property_weight: float = 0.30       # increased: preserve chemical properties
    reward_conversion_weight: float = 0.25     # conversion efficiency
    validity_bonus: float = 0.5
    invalid_penalty: float = -1.0

    # ── Agent ─────────────────────────────────────────────────────────
    max_agent_rounds: int = 5
    target_toxicity: float = 0.3
    min_tanimoto: float = 0.3

    # ── Device ───────────────────────────────────────────────────────
    device: str = "cuda"
    seed: int = 42

    def reward_weights_tuple(self):
        """Return reward weights as tuple."""
        return (
            self.reward_detox_weight,
            self.reward_similarity_weight,
            self.reward_qed_weight,
            self.reward_sa_weight,
            self.reward_property_weight,
            self.reward_conversion_weight,
        )
