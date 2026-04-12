"""Tokenizer wrapper for ToxGuard, using IUPACGPT's SentencePiece model directly.

The original IUPACGPT code used T5Tokenizer as a wrapper around the IUPAC
SentencePiece model.  However, transformers >= v5 broke compatibility —
T5Tokenizer no longer loads the full SPM vocabulary, collapsing it to just
4 special tokens + 100 extra_ids = 104, which maps almost every IUPAC
subword to <unk>.

This rewrite uses SentencePiece directly via the sentencepiece library,
providing a HuggingFace-compatible interface (__call__ returns input_ids,
attention_mask; has pad_token_id, unk_token, etc.) without depending on
the broken T5Tokenizer path.

Compatibility:
    - Drop-in replacement for the old ToxGuardTokenizer
    - Same __call__ signature: tokenizer(text) -> {"input_ids": [...], ...}
    - Same special token IDs (pad=0, eos=1, unk=2)
    - Same _tokenize behavior (strips leading ▁ sentinel)
    - Works with transformers v4.x AND v5.x
"""

import os
import re
import logging
from typing import Dict, List, Optional, Union

import torch
import sentencepiece as spm

logger = logging.getLogger(__name__)


class ToxGuardTokenizer:
    """SentencePiece-based tokenizer for IUPAC names.

    Uses the IUPAC SentencePiece model directly (NOT through T5Tokenizer)
    to avoid the vocab-size bug in transformers >= 5.x.

    The tokenizer replaces spaces with underscores before tokenization
    (IUPAC names can contain spaces) and reverses on decode.

    Token ID conventions (matching the original IUPACGPT checkpoint):
        0 = <pad>
        1 = </s>  (EOS)
        2 = <unk>
    """

    def __init__(self, vocab_file: str):
        """
        Args:
            vocab_file: Path to the iupac_spm.model SentencePiece file.
        """
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"SPM model not found: {vocab_file}")

        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        self._vocab_size = self.sp_model.get_piece_size()

        # Standard special token IDs (from the SPM model itself)
        self._pad_token_id = self.sp_model.pad_id()      # 0
        self._eos_token_id = self.sp_model.eos_id()       # 1
        self._unk_token_id = self.sp_model.unk_id()       # 2
        self._bos_token_id = self._unk_token_id           # IUPACGPT uses UNK as BOS

        # Special token strings
        self.pad_token = self.sp_model.id_to_piece(self._pad_token_id) if self._pad_token_id >= 0 else "<pad>"
        self.eos_token = self.sp_model.id_to_piece(self._eos_token_id) if self._eos_token_id >= 0 else "</s>"
        self.unk_token = self.sp_model.id_to_piece(self._unk_token_id) if self._unk_token_id >= 0 else "<unk>"

        logger.info(
            f"Loaded IUPAC SentencePiece tokenizer: vocab_size={self._vocab_size}, "
            f"pad={self._pad_token_id}, eos={self._eos_token_id}, unk={self._unk_token_id}"
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def unk_token_id(self) -> int:
        return self._unk_token_id

    def __len__(self) -> int:
        return self._vocab_size

    # ─── Core tokenization ───────────────────────────────────────────

    def _prepare_text(self, text: str) -> str:
        """Replace spaces with underscores (IUPAC names can have spaces)."""
        return re.sub(" ", "_", text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword pieces.

        Mirrors the original IUPACGPT behavior: strips the leading
        sentencepiece '▁' token that SentencePiece adds.
        """
        prepared = self._prepare_text(text)
        pieces = self.sp_model.EncodeAsPieces(prepared)
        # SentencePiece adds a leading ▁ token — strip it (same as original)
        if pieces and pieces[0] == "▁":
            pieces = pieces[1:]
        return pieces

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text (IUPAC name).
            add_eos: Whether to append EOS token (default True, matching T5).

        Returns:
            List of integer token IDs.
        """
        prepared = self._prepare_text(text)
        ids = self.sp_model.EncodeAsIds(prepared)
        # Strip the leading ▁ token's ID (same as original _tokenize behavior)
        if ids and ids[0] == self.sp_model.piece_to_id("▁"):
            ids = ids[1:]
        if add_eos and self._eos_token_id >= 0:
            ids.append(self._eos_token_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        # Filter out special tokens
        filtered = [i for i in ids if i not in (self._pad_token_id, self._eos_token_id)]
        text = self.sp_model.DecodeIds(filtered)
        # Reverse the space -> underscore transformation
        text = re.sub("_", " ", text)
        return text

    def convert_tokens_to_ids(self, token: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token string(s) to ID(s)."""
        if isinstance(token, str):
            return self.sp_model.piece_to_id(token)
        return [self.sp_model.piece_to_id(t) for t in token]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert token ID(s) to string(s)."""
        if isinstance(ids, int):
            return self.sp_model.id_to_piece(ids)
        return [self.sp_model.id_to_piece(i) for i in ids]

    # ─── HuggingFace-compatible __call__ ─────────────────────────────

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Tokenize text in HuggingFace-compatible format.

        Args:
            text: Single string or list of strings.
            padding: Whether to pad (not implemented for simplicity).
            max_length: Maximum sequence length (truncate if exceeded).
            return_tensors: If "pt", return PyTorch tensors.

        Returns:
            Dict with "input_ids" and "attention_mask".
        """
        if isinstance(text, str):
            ids = self.encode(text, add_eos=True)
            if max_length:
                ids = ids[:max_length]
            mask = [1] * len(ids)

            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.tensor([mask], dtype=torch.long),
                }
            return {"input_ids": ids, "attention_mask": mask}

        # Batch mode
        all_ids = []
        for t in text:
            ids = self.encode(t, add_eos=True)
            if max_length:
                ids = ids[:max_length]
            all_ids.append(ids)

        if return_tensors == "pt":
            from torch.nn.utils.rnn import pad_sequence
            tensors = [torch.tensor(ids, dtype=torch.long) for ids in all_ids]
            padded = pad_sequence(tensors, batch_first=True, padding_value=self._pad_token_id)
            mask = (padded != self._pad_token_id).long()
            return {"input_ids": padded, "attention_mask": mask}

        masks = [[1] * len(ids) for ids in all_ids]
        return {"input_ids": all_ids, "attention_mask": masks}


def get_tokenizer(
    vocab_path: str = None,
    serialized_path: str = None,
    iupacgpt_dir: str = None,
) -> ToxGuardTokenizer:
    """Load the IUPAC SentencePiece tokenizer.

    Priority:
      1. vocab_path (iupac_spm.model file) — preferred, uses SPM directly
      2. iupacgpt_dir (auto-detect from IUPACGPT installation)

    Note: serialized_path (.pt files) is no longer used because the old
    T5Tokenizer-based serialized tokenizer is incompatible with
    transformers >= 5.x.  We always load from the SPM model directly.

    Returns:
        ToxGuardTokenizer instance
    """
    if vocab_path and os.path.exists(vocab_path):
        return ToxGuardTokenizer(vocab_file=vocab_path)

    if iupacgpt_dir:
        candidates = [
            os.path.join(iupacgpt_dir, "iupac_gpt", "iupac_spm.model"),
            os.path.join(iupacgpt_dir, "iupac_spm.model"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return ToxGuardTokenizer(vocab_file=path)

    raise FileNotFoundError(
        "Could not find IUPAC SentencePiece tokenizer. Provide one of:\n"
        "  - vocab_path: path to iupac_spm.model\n"
        "  - iupacgpt_dir: path to iupac-gpt/ directory"
    )
