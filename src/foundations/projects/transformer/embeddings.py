"""Reusable embeddings block for Transformer models.

Combines token embedding + positional encoding + dropout into a single
plug-and-play module, eliminating the repeated boilerplate that appears
in every Transformer variant.

Public symbol:

    TokenAndPositionalEmbedding:
        ``nn.Module`` that maps integer token ids ``(B, L)`` to contextualised
        float embeddings ``(B, L, d_model)``.  Positional encoding strategy
        is selectable at construction time via ``pos_kind``.

Depends on:
    ``SinusoidalPositionalEncoding`` and ``LearnedPositionalEmbedding``
    from ``tfs.nn.positional_encoding``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from foundations.projects.transformer.positional_encoding import SinusoidalPositionalEncoding, \
    LearnedPositionalEmbedding


class TokenAndPositionalEmbedding(nn.Module):
    """Token embedding + positional encoding + dropout in one module.

    This is the standard "embedding front-end" shared by virtually every
    Transformer: it converts a batch of integer token ids into the float
    representations consumed by the encoder/decoder stack.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Embedding / model dimension.
        max_len: Maximum sequence length accepted by the positional encoder.
            A ``ValueError`` is raised at forward time if the input exceeds
            this limit.
        pos_kind: Positional encoding strategy.
            ``"sinusoidal"`` — fixed sin/cos (Vaswani et al., 2017).
            ``"learned"``    — trainable embedding (BERT / GPT-2).
        dropout: Dropout probability applied to the summed embeddings.
            Defaults to 0.0 (no dropout).
        pad_id: Optional padding token index.  When provided, the
            corresponding embedding vector is kept at zero and excluded
            from gradient updates.

    Shapes:
        - Input:  ``(batch, seq_len)``  — ``torch.long`` token ids.
        - Output: ``(batch, seq_len, d_model)``  — float embeddings.

    Raises:
        ValueError: If the input sequence length exceeds ``max_len``.

    Example:
        >>> emb = TokenAndPositionalEmbedding(
        ...     vocab_size=30_000, d_model=512,
        ...     max_len=1024, pos_kind="sinusoidal",
        ... )
        >>> ids = torch.randint(0, 30_000, (2, 128))
        >>> emb(ids).shape
        torch.Size([2, 128, 512])
    """

    _POS_REGISTRY: dict[str, type[nn.Module]] = {
        "sinusoidal": SinusoidalPositionalEncoding,
        "learned": LearnedPositionalEmbedding,
    }

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        max_len: int,
        pos_kind: Literal["sinusoidal", "learned"],
        dropout: float = 0.0,
        pad_id: int | None = None,
    ) -> None:
        super().__init__()
        self.max_len = max_len

        self.token = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )

        pos_cls = self._POS_REGISTRY.get(pos_kind)
        if pos_cls is None:
            raise ValueError(
                f"Unknown pos_kind {pos_kind!r}. "
                f"Choose from {sorted(self._POS_REGISTRY)}."
            )
        # dropout=0.0 here because we apply our own dropout after the sum
        self.pos = pos_cls(d_model=d_model, max_len=max_len, dropout=0.0)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Map token ids to float embeddings with positional information.

        Args:
            token_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Float tensor of shape ``(batch, seq_len, d_model)``.

        Raises:
            ValueError: If ``seq_len > max_len``.
        """
        seq_len = token_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}."
            )

        x = self.token(token_ids)   # (B, L, d_model)
        x = self.pos(x)             # adds positional signal
        return self.dropout(x)