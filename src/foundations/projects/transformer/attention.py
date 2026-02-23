"""Scaled dot-product attention for Transformer models.

Implements the core attention mechanism from Vaswani et al.
"Attention Is All You Need" (NeurIPS 2017, Section 3.2.1)::

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

The single public function :func:`scaled_dot_product_attention` accepts
an optional **additive** mask (``0.0`` for allowed positions, ``-inf``
for blocked positions) that is added to the raw logits before softmax.
This convention supports both causal and padding masks — and their
combination via simple addition.

Designed for the multi-head layout where Q, K, V have shape
``(B, H, T, D)`` (batch, heads, sequence length, head dimension).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention.

    .. math::

        \text{Attention}(Q, K, V) = \text{softmax}
        \\Bigl(\frac{Q K^\top}{\\sqrt{d_k}} + M\\Bigr) V

    where *M* is an optional additive mask (``0`` = attend, ``-inf`` = block).

    Shape legend:
        - **B** — Batch size (number of independent sequences).
        - **H** — Number of attention heads.
        - **T_q** — Query sequence length (number of query positions).
        - **T_k** — Key/value sequence length (number of positions to
          attend over). Equals ``T_q`` in self-attention; may differ in
          cross-attention.
        - **D** — Head dimension (``d_model // H``), i.e. the size of
          each query/key/value vector *per head*.

    Args:
        q: Query tensor of shape ``(B, H, T_q, D)``.
            Each query vector "asks a question" — it is compared against
            every key to decide where to focus.
        k: Key tensor of shape ``(B, H, T_k, D)``.
            Each key vector "advertises" what information its position
            holds. The dot product ``Q K^T`` measures relevance.
        v: Value tensor of shape ``(B, H, T_k, D)``.
            Each value vector carries the actual content that gets
            aggregated according to the attention weights.
        mask: Optional additive mask broadcastable to ``(B, H, T_q, T_k)``.
            Use ``0.0`` for positions that may be attended to and
            ``-inf`` for positions that must be ignored.  Common shapes:

            * ``(T_q, T_k)`` — causal mask shared across batch and heads.
            * ``(B, 1, 1, T_k)`` — padding mask shared across heads and
              query positions.

    Returns:
        A tuple ``(context, attn_weights)`` where:

        * **context** has shape ``(B, H, T_q, D)`` — the weighted sum
          of value vectors for each query position.
        * **attn_weights** has shape ``(B, H, T_q, T_k)`` — the
          softmax-normalised attention scores. Row *i* shows how much
          query position *i* attends to each key position.

    Raises:
        RuntimeError: If Q/K/V shapes are incompatible for matmul.

    Example::

        >>> B, H, T, D = 2, 4, 8, 16  # batch, heads, seq_len, head_dim
        >>> q = torch.randn(B, H, T, D)
        >>> k = torch.randn(B, H, T, D)
        >>> v = torch.randn(B, H, T, D)
        >>> ctx, w = scaled_dot_product_attention(q, k, v)
        >>> ctx.shape
        torch.Size([2, 4, 8, 16])
        >>> w.shape
        torch.Size([2, 4, 8, 8])
    """
    # Get the head dimension from the query shape (also equals k/v head dim)
    d_k = q.size(-1)

    # (B, H, T_q, D) @ (B, H, D, T_k) → (B, H, T_q, T_k)
    # This computes the raw attention scores by taking the dot product of each query vector with each key vector.
    # The result is a matrix of shape (B, H, T_q, T_k) where each entry [b, h, i, j] represents the attention score between the i-th query position and the j-th key position for head h in batch b.
    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply additive mask (broadcasts as needed)
    if mask is not None:
        logits = logits + mask

    # Normalise along the key dimension to get attention weights
    # This is the softmax step in the attention formula.
    # Each row of attn_weights corresponds to a query position and sums to 1,
    # indicating how much attention that query pays to each key position.
    attn_weights = F.softmax(logits, dim=-1)  # (B, H, T_q, T_k)

    # weighted sum of values: (B, H, T_q, T_k) @ (B, H, T_k, D) → (B, H, T_q, D)
    # This computes the final context vectors for each query position by taking a weighted sum of the value vectors,
    # where the weights are given by the attention scores.
    context = torch.matmul(attn_weights, v)

    return context, attn_weights
