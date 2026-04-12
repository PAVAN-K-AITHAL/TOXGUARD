"""PPO training loop for IUPAC-GPT molecule detoxification.

Proximal Policy Optimization (PPO) with:
    - Policy: IUPAC-GPT + LoRA (rank=8, trainable)
    - Reference: Frozen IUPAC-GPT (for KL divergence penalty)
    - Reward: Multi-objective (detox + similarity + QED + SA)
    - Clipped surrogate objective with advantage normalization
    - Adaptive KL coefficient

Training loop:
    For each toxic seed molecule:
        1. Generate N candidate IUPAC names from policy
        2. Resolve IUPAC → SMILES (OPSIN/PubChem)
        3. Validate with RDKit (Tanimoto, QED, SA)
        4. Score with ToxGuard Phase 1 (frozen reward model)
        5. Compute multi-objective reward
        6. Compute PPO loss + KL penalty
        7. Update policy LoRA parameters

Usage:
    trainer = PPOTrainer(
        generator=generator,
        reward_fn=reward_function,
        config=config,
    )
    trainer.train(
        seed_molecules=[
            {"iupac": "nitrobenzene", "smiles": "[O-][N+](=O)c1ccccc1", "p_toxic": 0.91},
            ...
        ],
        num_steps=500,
    )
"""

import logging
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

logger = logging.getLogger(__name__)

try:
    from .ppo_config import PPOConfig
    from .molecule_generator import MoleculeGenerator
    from .reward_function import RewardFunction, RewardInfo
except ImportError:
    from ppo_config import PPOConfig
    from molecule_generator import MoleculeGenerator
    from reward_function import RewardFunction, RewardInfo


class PPOTrainer:
    """PPO trainer for molecule detoxification.

    Implements the Proximal Policy Optimization algorithm adapted for
    autoregressive IUPAC name generation. Uses ToxGuard Phase 1 as the
    reward model and IUPAC-GPT as the policy.

    Key design decisions:
        - Single-step MDP: each generation is one "episode" with one reward
        - KL penalty (not clipping) for controlling policy divergence
        - Advantage = reward - baseline (moving average of rewards)
        - Adaptive KL coefficient to stay near target KL

    Args:
        generator: MoleculeGenerator with policy + reference models.
        reward_fn: RewardFunction for computing multi-objective rewards.
        config: PPOConfig with training hyperparameters.
        output_dir: Directory for saving checkpoints and logs.
    """

    def __init__(
        self,
        generator: MoleculeGenerator,
        reward_fn: RewardFunction,
        config: Optional[PPOConfig] = None,
        output_dir: str = "./outputs/rl_training",
    ):
        self.generator = generator
        self.reward_fn = reward_fn
        self.config = config or PPOConfig()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Optimizer for LoRA parameters only
        self.optimizer = AdamW(
            self.generator.get_trainable_params(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.total_train_steps,
            eta_min=self.config.learning_rate * 0.01,
        )

        # Adaptive KL coefficient
        self.kl_coeff = self.config.kl_coeff

        # Baseline (moving average of rewards)
        self.reward_baseline = 0.0
        self.baseline_decay = 0.95

        # Training stats
        self.step = 0
        self.training_log = []

    def train(
        self,
        seed_molecules: List[Dict],
        num_steps: Optional[int] = None,
        save_every: int = 50,
        log_every: int = 10,
    ):
        """Main PPO training loop.

        Args:
            seed_molecules: List of dicts, each with:
                - "iupac": seed IUPAC name
                - "smiles": seed SMILES
                - "p_toxic": seed P(toxic) from Phase 1
            num_steps: Total training steps (default: config.total_train_steps).
            save_every: Save checkpoint every N steps.
            log_every: Print log every N steps.
        """
        num_steps = num_steps or self.config.total_train_steps
        n_seeds = len(seed_molecules)

        logger.info(
            f"Starting PPO training: {num_steps} steps, "
            f"{n_seeds} seed molecules, batch_size={self.config.batch_size}"
        )

        for step in range(self.step, self.step + num_steps):
            self.step = step
            t0 = time.time()

            # Select seed molecule (round-robin)
            seed = seed_molecules[step % n_seeds]
            seed_iupac = seed["iupac"]
            seed_smiles = seed["smiles"]
            seed_p_toxic = seed.get("p_toxic", 0.5)

            # ── Step 1: Generate candidates from policy ───────────────
            candidates = self.generator.generate(
                seed_iupac=seed_iupac,
                n=self.config.batch_size,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                mode="prefix",
            )

            if not candidates:
                logger.warning(f"Step {step}: no candidates generated for '{seed_iupac}'")
                continue

            # ── Step 2: Compute rewards ───────────────────────────────
            rewards_list = []
            infos = []
            for cand in candidates:
                r, info = self.reward_fn.compute(
                    candidate_iupac=cand,
                    seed_smiles=seed_smiles,
                    seed_p_toxic=seed_p_toxic,
                )
                rewards_list.append(r)
                infos.append(info)

            rewards = torch.tensor(rewards_list, dtype=torch.float32).to(
                self.generator.device
            )

            # ── Step 3: Compute advantages ────────────────────────────
            # Simple baseline: moving average of rewards
            advantages = rewards - self.reward_baseline
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update baseline
            mean_reward = rewards.mean().item()
            self.reward_baseline = (
                self.baseline_decay * self.reward_baseline
                + (1 - self.baseline_decay) * mean_reward
            )

            # ── Step 4: PPO update ────────────────────────────────────
            ppo_loss, kl_div = self._ppo_update(
                candidates, advantages, rewards
            )

            # ── Step 5: Adaptive KL ──────────────────────────────────
            if kl_div > self.config.kl_target * 1.5:
                self.kl_coeff *= 1.5
            elif kl_div < self.config.kl_target / 1.5:
                self.kl_coeff *= 1.0 / 1.5
            self.kl_coeff = max(0.001, min(1.0, self.kl_coeff))

            # ── Step 6: Scheduler step ────────────────────────────────
            self.scheduler.step()

            # ── Logging ───────────────────────────────────────────────
            elapsed = time.time() - t0
            n_valid = sum(1 for i in infos if i.is_valid)
            n_less_toxic = sum(
                1 for i in infos
                if i.is_valid and i.p_toxic < seed_p_toxic
            )
            mean_tox = (
                sum(i.p_toxic for i in infos if i.is_valid) / max(n_valid, 1)
            )

            step_stats = {
                "step": step,
                "seed": seed_iupac,
                "n_candidates": len(candidates),
                "n_valid": n_valid,
                "n_less_toxic": n_less_toxic,
                "mean_reward": mean_reward,
                "mean_p_toxic": mean_tox,
                "ppo_loss": ppo_loss,
                "kl_div": kl_div,
                "kl_coeff": self.kl_coeff,
                "lr": self.scheduler.get_last_lr()[0],
                "time_s": elapsed,
            }
            self.training_log.append(step_stats)

            if step % log_every == 0:
                logger.info(
                    f"Step {step:4d} | seed={seed_iupac[:20]:20s} | "
                    f"valid={n_valid}/{len(candidates)} | "
                    f"less_toxic={n_less_toxic} | "
                    f"R={mean_reward:.3f} | "
                    f"P(tox)={mean_tox:.3f} | "
                    f"KL={kl_div:.4f} | "
                    f"loss={ppo_loss:.4f} | "
                    f"{elapsed:.1f}s"
                )

            # ── Checkpoints ───────────────────────────────────────────
            if step > 0 and step % save_every == 0:
                self._save_checkpoint(step)

        # Final save
        self._save_checkpoint(self.step)
        logger.info(f"PPO training complete after {num_steps} steps")

    def _ppo_update(
        self,
        candidates: List[str],
        advantages: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple:
        """Perform one PPO update step.

        For each candidate:
            1. Tokenize the generated IUPAC name
            2. Compute log-probs under policy (WITH gradients)
            3. Compute log-probs under reference (frozen, no gradients)
            4. Compute PPO loss = -(advantage * policy_logprob) + KL penalty
            5. Backpropagate and update LoRA parameters

        Returns:
            (loss_value, mean_kl_divergence)
        """
        self.generator.policy.train()
        total_loss = 0.0
        total_kl = 0.0
        n_updates = 0

        for epoch in range(self.config.num_epochs_per_batch):
            for i, cand in enumerate(candidates):
                if i >= len(advantages):
                    break
                if not cand.strip():
                    continue

                # Tokenize
                tokenized = self.generator.tokenizer(cand)
                token_ids = tokenized["input_ids"]
                if not token_ids or len(token_ids) < 2:
                    continue

                # Build full sequence with BOS
                full_ids = torch.tensor(
                    [self.generator._bos_id] + token_ids,
                    dtype=torch.long,
                ).unsqueeze(0).to(self.generator.device)

                attn_mask = torch.ones_like(full_ids)

                # ── Policy forward pass (WITH gradients for backprop) ──
                policy_outputs = self.generator.policy(
                    input_ids=full_ids,
                    attention_mask=attn_mask,
                )
                policy_logits = policy_outputs.logits[:, :-1, :]
                target_ids = full_ids[:, 1:]

                policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                policy_token_lps = policy_log_probs.gather(
                    2, target_ids.unsqueeze(-1)
                ).squeeze(-1)
                policy_seq_lp = policy_token_lps.sum(dim=-1)  # shape: (1,)

                # ── Reference forward pass (NO gradients — frozen) ─────
                with torch.no_grad():
                    ref_outputs = self.generator.ref(
                        input_ids=full_ids,
                        attention_mask=attn_mask,
                    )
                    ref_logits = ref_outputs.logits[:, :-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_lps = ref_log_probs.gather(
                        2, target_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    ref_seq_lp = ref_token_lps.sum(dim=-1)

                # ── KL divergence (policy vs reference) ────────────────
                kl = (policy_seq_lp - ref_seq_lp).mean()
                total_kl += kl.detach().item()

                # ── PPO loss ───────────────────────────────────────────
                # Advantage for this candidate
                adv = advantages[i].detach()

                # Loss = -advantage * policy_logprob + kl_coeff * KL
                # Minimizing this = maximizing reward-weighted logprob
                loss = -(adv * policy_seq_lp.mean()) + self.kl_coeff * kl

                # ── Gradient update ────────────────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.get_trainable_params(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_loss += loss.detach().item()
                n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)
        avg_kl = total_kl / max(n_updates, 1)

        return avg_loss, avg_kl

    def _save_checkpoint(self, step: int):
        """Save policy weights and training log."""
        ckpt_path = os.path.join(
            self.output_dir, f"policy_step_{step}.pt"
        )
        self.generator.save_policy(ckpt_path)

        # Save training log
        import json
        log_path = os.path.join(self.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

        logger.info(f"Checkpoint saved: step {step}")

    def get_training_summary(self) -> Dict:
        """Get summary statistics of the training run."""
        if not self.training_log:
            return {}

        rewards = [s["mean_reward"] for s in self.training_log]
        valid_rates = [
            s["n_valid"] / max(s["n_candidates"], 1)
            for s in self.training_log
        ]
        detox_rates = [
            s["n_less_toxic"] / max(s["n_candidates"], 1)
            for s in self.training_log
        ]

        return {
            "total_steps": len(self.training_log),
            "final_mean_reward": rewards[-1] if rewards else 0,
            "best_mean_reward": max(rewards) if rewards else 0,
            "mean_valid_rate": sum(valid_rates) / len(valid_rates),
            "mean_detox_rate": sum(detox_rates) / len(detox_rates),
            "final_kl_coeff": self.kl_coeff,
        }
