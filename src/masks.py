"""Mask utilities for Transformer models.

Two public functions are provided:

    make_padding_mask:
        Builds an **additive** mask that prevents attention to ``<pad>``
        tokens introduced by batching sequences of different lengths.
        Padded positions receive ``-inf``; real tokens receive ``0.0``.

    make_causal_mask:
        Builds an **additive** mask that prevents each position from
        attending to future (right-side) positions — the standard
        autoregressive constraint for decoder self-attention.

Both masks use the **additive convention**: they are added directly to
the raw attention logits before softmax.  This makes them composable —
a combined padding + causal mask is simply their element-wise sum::

    combined = make_padding_mask(ids, pad_id=0) + make_causal_mask(T)

Shapes are chosen for seamless broadcasting with multi-head attention
logits of shape ``(B, H, T, T)``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def make_padding_mask(token_ids: Tensor, pad_id: int = 0) -> Tensor:
    """Build an additive padding mask from token IDs.

    Positions whose token ID equals ``pad_id`` are filled with ``-inf``
    so that ``softmax(logits + mask)`` drives their attention weight to
    zero.  All other positions are ``0.0`` (no effect on logits).

    Args:
        token_ids: Integer tensor of shape ``(B, T)`` containing token
            indices.  Positions equal to ``pad_id`` are treated as
            padding.
        pad_id: The token index used for padding.  Defaults to ``0``.

    Returns:
        Float tensor of shape ``(B, 1, 1, T)``.  The two singleton
        dimensions correspond to *heads* and *query positions*,
        allowing the mask to broadcast over the full
        ``(B, H, T_q, T_k)`` logit tensor.

    Example::

        >>> ids = torch.tensor([[5, 3, 0, 0]])
        >>> make_padding_mask(ids, pad_id=0)
        tensor([[[[0., 0., -inf, -inf]]]])
    """
    # (B, T) → True where token is padding
    is_pad = token_ids.eq(pad_id)  # (B, T)

    # Convert to additive mask: pad → -inf, real → 0.0
    mask = torch.zeros_like(token_ids, dtype=torch.float32)
    mask.masked_fill_(is_pad, float("-inf"))

    # Reshape for broadcasting: (B, T) → (B, 1, 1, T)
    return mask.unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len: int) -> Tensor:
    """Build an additive causal (autoregressive) mask.

    Position ``(i, j)`` is ``0.0`` when ``j <= i`` (past and present)
    and ``-inf`` when ``j > i`` (future), so that after
    ``softmax(logits + mask)`` no position can attend to later tokens.

    Args:
        seq_len: Sequence length ``T``.

    Returns:
        Float tensor of shape ``(T, T)``.  Broadcasts naturally when
        added to logits of shape ``(B, H, T, T)``.

    Example::

        >>> make_causal_mask(3)
        tensor([[0., -inf, -inf],
                [0., 0., -inf],
                [0., 0., 0.]])
    """
    # Upper-triangular ones (strict: diagonal=1 excludes the main diagonal)
    future = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

    # Convert: 1 → -inf, 0 → 0.0
    return future.masked_fill(future.bool(), float("-inf"))
