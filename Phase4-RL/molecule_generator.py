"""IUPAC-GPT policy network for PPO-based molecule generation.

Uses IUPAC-GPT (GPT-2, 7.1M params) as the autoregressive policy:
    - Policy model: IUPAC-GPT + new LoRA adapters (rank=8, trainable)
    - Reference model: Frozen copy of IUPAC-GPT (for KL penalty)

Generation modes:
    1. Prefix-conditioned: Keep first N tokens of seed IUPAC, generate rest
    2. Unconditional: Generate from BOS token (more diversity)

All generated IUPAC names are validated via OPSIN → RDKit pipeline.

Usage:
    generator = MoleculeGenerator.from_checkpoint(
        checkpoint_dir="iupacGPT/iupac-gpt/checkpoints/iupac",
        device="cuda",
    )
    candidates = generator.generate(seed_iupac="nitrobenzene", n=16)
"""

import copy
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from .ppo_config import PPOConfig
except ImportError:
    from ppo_config import PPOConfig


class MoleculeGenerator:
    """IUPAC-GPT policy network for autoregressive IUPAC name generation.

    Manages two copies of IUPAC-GPT:
        - policy_model: with new LoRA adapters (trainable by PPO)
        - ref_model: frozen copy (for KL divergence computation)

    The policy generates candidate IUPAC names by autoregressive sampling.
    Log-probabilities from both policy and reference are returned for
    PPO loss computation.

    Args:
        policy_model: IUPAC-GPT with trainable LoRA adapters.
        ref_model: Frozen IUPAC-GPT (same architecture, no LoRA or original LoRA).
        tokenizer: IUPAC SentencePiece tokenizer.
        config: PPOConfig instance.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        tokenizer,
        config: Optional[PPOConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        # Policy model (trainable LoRA)
        self.policy = policy_model.to(self.device)
        self.policy.train()

        # Reference model (frozen — for KL divergence)
        self.ref = ref_model.to(self.device)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer

        # BOS token ID
        self._bos_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
        # EOS / pad token
        self._eos_id = getattr(tokenizer, "eos_token_id", None) or self._bos_id
        self._pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        tokenizer_path: Optional[str] = None,
        config: Optional[PPOConfig] = None,
        device: str = "cuda",
    ) -> "MoleculeGenerator":
        """Load IUPAC-GPT and create policy + reference models.

        The policy gets NEW LoRA adapters (rank=8 by default) for PPO.
        The reference model is a clean frozen copy of the base model.

        Args:
            checkpoint_dir: Path to IUPAC-GPT checkpoint (config.json + weights).
            tokenizer_path: Path to iupac_spm.model tokenizer.
            config: PPOConfig instance.
            device: 'cuda' or 'cpu'.
        """
        config = config or PPOConfig()

        # Find Phase 1 path for imports
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        phase1_path = os.path.join(project_root, "Phase1-IUPACGPT")
        if phase1_path not in sys.path:
            sys.path.insert(0, phase1_path)

        from iupacGPT_finetune.tokenizer import get_tokenizer
        from iupacGPT_finetune.lora import LoRAConfig, apply_lora_to_model

        from transformers import GPT2Config, GPT2LMHeadModel

        # ── Load tokenizer ────────────────────────────────────────────
        spm_path = tokenizer_path or os.path.join(
            project_root, "iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model"
        )
        tokenizer = get_tokenizer(vocab_path=spm_path, iupacgpt_dir=checkpoint_dir)

        # ── Load base GPT-2 LM head model ─────────────────────────────
        gpt2_config = GPT2Config.from_pretrained(checkpoint_dir)
        base_model = GPT2LMHeadModel.from_pretrained(checkpoint_dir, config=gpt2_config)

        # ── Create reference model (frozen, no LoRA) ──────────────────
        ref_model = copy.deepcopy(base_model)
        ref_model.eval()

        # ── Create policy model (with new LoRA adapters) ──────────────
        policy_model = base_model  # reuse the loaded model
        lora_cfg = LoRAConfig(
            r=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            fan_in_fan_out=True,
        )
        # Apply LoRA to the transformer inside the LM head model
        policy_model.transformer, _ = apply_lora_to_model(
            policy_model.transformer, lora_cfg
        )

        # Freeze non-LoRA parameters
        for name, param in policy_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Count trainable params
        trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in policy_model.parameters())
        logger.info(
            f"Policy model: {trainable:,} trainable / {total:,} total params "
            f"({100*trainable/total:.2f}%)"
        )

        return cls(policy_model, ref_model, tokenizer, config, device)

    def _tokenize_prefix(self, iupac_name: str, fraction: float = 0.5) -> torch.Tensor:
        """Tokenize an IUPAC name and return the prefix tokens.

        Args:
            iupac_name: Full IUPAC name to use as seed.
            fraction: Fraction of tokens to keep as prefix (0.0 to 1.0).

        Returns:
            (1, prefix_len) tensor of token IDs with BOS prepended.
        """
        tokenized = self.tokenizer(iupac_name)
        all_ids = tokenized["input_ids"]
        prefix_len = max(1, int(len(all_ids) * fraction))
        prefix_ids = all_ids[:prefix_len]

        # Prepend BOS
        ids = [self._bos_id] + prefix_ids
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def generate(
        self,
        seed_iupac: str,
        n: int = 16,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_length: Optional[int] = None,
        prefix_fraction: Optional[float] = None,
        mode: str = "prefix",
    ) -> List[str]:
        """Generate candidate IUPAC names from the policy model.

        Args:
            seed_iupac: Seed IUPAC name for prefix conditioning.
            n: Number of candidates to generate.
            temperature: Sampling temperature (default from config).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            max_length: Maximum generated length in tokens.
            prefix_fraction: Fraction of seed tokens to keep.
            mode: "prefix" (conditioned on seed prefix) or "scratch"
                (generate from BOS token only).

        Returns:
            List of generated IUPAC name strings.
        """
        self.policy.eval()

        temp = temperature or self.config.temperature
        tp = top_p or self.config.top_p
        tk = top_k if top_k is not None else self.config.top_k
        ml = max_length or self.config.max_gen_length
        pf = prefix_fraction or self.config.prefix_fraction

        candidates = []

        for _ in range(n):
            # Build input
            if mode == "prefix":
                input_ids = self._tokenize_prefix(seed_iupac, fraction=pf)
            else:
                input_ids = torch.tensor(
                    [[self._bos_id]], dtype=torch.long
                ).to(self.device)

            generated = self._sample_sequence(
                input_ids, ml, temp, tp, tk
            )
            texts = self._decode(generated, seed_iupac=seed_iupac)
            candidates.extend(texts)

        self.policy.train()
        return candidates

    def _sample_sequence(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Autoregressive sampling from the policy model.

        Returns the full generated sequence including the prefix.
        """
        generated = input_ids.clone()

        for _ in range(max_length - generated.shape[1]):
            outputs = self.policy(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
            )
            next_token_logits = outputs.logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back
                next_token_logits = sorted_logits.scatter(
                    1, sorted_indices, sorted_logits
                )

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS
            if next_token.item() == self._eos_id:
                break

        return generated

    def _decode(self, token_ids: torch.Tensor, seed_iupac: str = "") -> List[str]:
        """Decode generated token IDs back to IUPAC name strings.

        Handles common issues with IUPAC-GPT generation:
        - Semicolons: model may concatenate multiple molecules ("A;B;C")
          -> split and treat each as a separate candidate
        - Seed prefix: in prefix mode, the seed is included in output
          -> strip if the generated text starts with the seed
        - Invalid characters: filter out non-IUPAC characters

        Returns a list of cleaned IUPAC name strings (may be empty).
        """
        ids = token_ids[0].cpu().tolist()

        # Remove BOS
        if ids and ids[0] == self._bos_id:
            ids = ids[1:]

        # Remove EOS and anything after
        if self._eos_id in ids:
            ids = ids[:ids.index(self._eos_id)]

        # Remove padding
        ids = [i for i in ids if i != self._pad_id]

        if not ids:
            return []

        try:
            text = self.tokenizer.decode(ids)
        except Exception as e:
            logger.debug(f"Decoding failed: {e}")
            return []

        # Clean up raw text
        text = text.strip()
        if not text:
            return []

        # Split on semicolons — model often generates "molA;molB;molC"
        fragments = [f.strip() for f in text.split(";") if f.strip()]

        # Also split on newlines
        expanded = []
        for frag in fragments:
            expanded.extend(line.strip() for line in frag.split("\n") if line.strip())

        # Clean and filter each fragment
        results = []
        seed_lower = seed_iupac.lower().strip() if seed_iupac else ""

        for frag in expanded:
            cleaned = self._clean_iupac_name(frag)
            if not cleaned:
                continue

            # Skip if it's just the seed repeated
            if seed_lower and cleaned.lower() == seed_lower:
                continue

            # Skip very short names (likely garbage)
            if len(cleaned) < 3:
                continue

            # Skip very long names (likely concatenated garbage)
            if len(cleaned) > 200:
                continue

            results.append(cleaned)

        return results

    @staticmethod
    def _clean_iupac_name(text: str) -> str:
        """Clean a raw generated text into a valid IUPAC name.

        - Remove non-printable characters
        - Keep only IUPAC-valid characters: letters, digits, hyphens,
          parentheses, brackets, commas, spaces, primes
        - Strip leading/trailing whitespace and hyphens
        """
        # Remove control characters
        text = "".join(c for c in text if c.isprintable())

        # Keep only valid IUPAC characters
        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "-,()[]' "
        )
        text = "".join(c for c in text if c in allowed)

        # Strip
        text = text.strip(" -,")

        return text

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities under policy and reference models.

        Used by PPO to compute the probability ratio π(a|s) / π_ref(a|s).

        Args:
            input_ids: (B, prefix_len) prefix tokens.
            attention_mask: (B, prefix_len) attention mask.
            generated_ids: (B, gen_len) the full generated sequence.

        Returns:
            (policy_logprobs, ref_logprobs) — both shape (B,) summed
            over generated tokens.
        """
        # Full sequence = input + generated continuation
        full_ids = generated_ids  # already includes prefix

        # Policy log probs
        self.policy.eval()
        with torch.no_grad():
            policy_outputs = self.policy(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
            )
        policy_logits = policy_outputs.logits[:, :-1, :]  # shift right
        target_ids = full_ids[:, 1:]  # shift left

        policy_logprobs = self._sequence_logprobs(policy_logits, target_ids)

        # Reference log probs
        with torch.no_grad():
            ref_outputs = self.ref(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
            )
        ref_logits = ref_outputs.logits[:, :-1, :]
        ref_logprobs = self._sequence_logprobs(ref_logits, target_ids)

        return policy_logprobs, ref_logprobs

    @staticmethod
    def _sequence_logprobs(
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sequence log probabilities.

        Args:
            logits: (B, L-1, V) predicted logits.
            target_ids: (B, L-1) actual token IDs.

        Returns:
            (B,) sum of log probs over sequence.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs.sum(dim=-1)

    def get_trainable_params(self):
        """Return trainable (LoRA) parameters for the optimizer."""
        return [p for p in self.policy.parameters() if p.requires_grad]

    def save_policy(self, save_path: str):
        """Save only the trainable LoRA weights."""
        trainable_state = {
            name: param.data
            for name, param in self.policy.named_parameters()
            if param.requires_grad
        }
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(trainable_state, save_path)
        logger.info(f"Policy LoRA weights saved to: {save_path}")

    def load_policy(self, load_path: str):
        """Load saved LoRA weights into the policy model."""
        state = torch.load(load_path, map_location=self.device)
        model_state = self.policy.state_dict()
        model_state.update(state)
        self.policy.load_state_dict(model_state)
        logger.info(f"Policy LoRA weights loaded from: {load_path}")
