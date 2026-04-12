"""Attention-based interpretability utilities for ToxGuard.

This module converts GPT-2 self-attention tensors into token-level attribution
scores and heatmap visualizations for IUPAC subword tokens.
"""

import os
import re
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

TOXICOPHORE_PATTERNS = ("nitro", "chloro", "epoxy")


def _display_token(token: str, is_bos: bool = False) -> str:
    """Convert raw SentencePiece token into a readable display token."""
    if is_bos:
        return "[BOS]"

    shown = (token or "").replace("\u2581", " ")
    shown = shown.replace("<pad>", "[PAD]")
    shown = shown.replace("<unk>", "[UNK]")
    shown = shown.replace("</s>", "[EOS]")
    shown = shown.strip()
    return shown if shown else "[SP]"


def compute_attention_token_scores(
    attentions: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    pooling_strategy: str,
    layer_aggregation: str = "mean",
) -> List[torch.Tensor]:
    """Compute token attribution scores from self-attention weights.

    Args:
        attentions: tuple/list of layer attention tensors, each (B, H, L, L).
        attention_mask: (B, L) mask of valid tokens.
        pooling_strategy: one of "last_token", "mean", "cls".
        layer_aggregation: "mean" (default) or "last" layer only.

    Returns:
        List of length B. Each element is a 1D tensor of normalized token
        attribution scores over valid tokens (sums to ~1).
    """
    if not attentions:
        return []

    if layer_aggregation not in {"mean", "last"}:
        raise ValueError("layer_aggregation must be either 'mean' or 'last'.")

    stacked = torch.stack([a.detach().float().cpu() for a in attentions], dim=0)
    # stacked: (N_layers, B, H, L, L)
    if layer_aggregation == "last":
        layer_attn = stacked[-1]  # (B, H, L, L)
    else:
        layer_attn = stacked.mean(dim=0)  # (B, H, L, L)

    # Average heads -> (B, L, L)
    attn = layer_attn.mean(dim=1)

    scores_per_sample: List[torch.Tensor] = []
    batch_size = attn.shape[0]
    for b in range(batch_size):
        valid_len = int(attention_mask[b].long().sum().item())
        valid_len = max(valid_len, 1)

        # Query->Key attention among valid tokens only.
        attn_valid = attn[b, :valid_len, :valid_len]

        if pooling_strategy == "cls":
            query_idx = torch.tensor([0], dtype=torch.long)
        elif pooling_strategy == "mean":
            query_idx = torch.arange(valid_len, dtype=torch.long)
        else:
            query_idx = torch.tensor([valid_len - 1], dtype=torch.long)

        token_scores = attn_valid[query_idx].mean(dim=0)
        token_scores = token_scores.clamp(min=0.0)
        token_scores = token_scores / token_scores.sum().clamp(min=1e-9)
        scores_per_sample.append(token_scores)

    return scores_per_sample


def build_token_attribution(
    tokenizer,
    input_ids: torch.Tensor,
    attention_scores: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int = 10,
) -> Dict[str, object]:
    """Create token-level attribution records for one sample."""
    valid_len = int(attention_mask.long().sum().item())
    valid_len = max(valid_len, 1)

    ids = input_ids[:valid_len].detach().cpu().tolist()
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)

    bos_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    display_tokens_full = [
        _display_token(tok, is_bos=(idx == 0 and ids[idx] == bos_id))
        for idx, tok in enumerate(raw_tokens)
    ]

    score_values_full = attention_scores[:valid_len].detach().cpu().numpy()

    # Remove special tokens for interpretability display.
    special_ids = {
        bos_id,
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    }
    keep_indices = [
        i for i, tok_id in enumerate(ids)
        if tok_id not in special_ids
    ]
    if not keep_indices:
        keep_indices = list(range(valid_len))

    display_tokens = [display_tokens_full[i] for i in keep_indices]
    score_values = score_values_full[keep_indices]
    score_values = score_values / max(float(score_values.sum()), 1e-9)
    records = []
    for idx, (tok, score) in enumerate(zip(display_tokens, score_values)):
        records.append({
            "index": idx,
            "token": tok,
            "score": float(score),
        })

    top_tokens = sorted(records, key=lambda r: r["score"], reverse=True)[:max(top_k, 1)]

    return {
        "tokens": display_tokens,
        "scores": [float(x) for x in score_values.tolist()],
        "token_attributions": records,
        "top_tokens": top_tokens,
    }


def detect_toxicophore_attention(
    tokens: Sequence[str],
    scores: Sequence[float],
    patterns: Iterable[str] = TOXICOPHORE_PATTERNS,
    max_span: int = 3,
) -> List[Dict[str, object]]:
    """Find toxicophore-like spans and aggregate their attention scores."""
    norm_tokens = [re.sub(r"[^a-z0-9]+", "", t.lower()) for t in tokens]

    hits: Dict[tuple, Dict[str, object]] = {}
    n = len(norm_tokens)
    for start in range(n):
        for span in range(1, max_span + 1):
            end = start + span
            if end > n:
                break
            normalized_span = "".join(norm_tokens[start:end])
            if not normalized_span:
                continue

            for pattern in patterns:
                if pattern in normalized_span:
                    key = (pattern, start, end)
                    fragment = "".join(tokens[start:end]).strip() or " ".join(tokens[start:end]).strip()
                    score = float(sum(scores[start:end]))
                    hits[key] = {
                        "pattern": pattern,
                        "fragment": fragment,
                        "start": start,
                        "end": end - 1,
                        "score": score,
                    }

    return sorted(hits.values(), key=lambda h: h["score"], reverse=True)


def save_attention_heatmap(
    tokens: Sequence[str],
    scores: Sequence[float],
    output_path: str,
    title: str,
):
    """Save a 1xL token-attention heatmap image to disk."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for attention heatmap export. "
            "Install with: pip install matplotlib"
        ) from exc

    values = np.asarray(scores, dtype=float).reshape(1, -1)
    width = max(8.0, 0.5 * len(tokens))

    fig, ax = plt.subplots(figsize=(width, 2.8))
    heat = ax.imshow(values, aspect="auto", cmap="magma")

    ax.set_yticks([])
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
    ax.set_title(title)

    cbar = fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("attention attribution", rotation=90)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path
