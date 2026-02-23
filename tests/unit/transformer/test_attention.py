"""Tests for scaled dot-product attention.

Verifies the core attention mechanism described in Vaswani et al.
"Attention Is All You Need" (NeurIPS 2017, Section 3.2.1)::

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Tests cover:
    1. Output and weight shapes for multi-head layout ``(B, H, T, D)``
    2. Causal masking zeroes attention to future positions
    3. Padding masking zeroes attention to ``<pad>`` tokens
    4. Attention weight rows sum to 1 (valid probability distributions)
    5. Gradient flow through Q, K, V
"""

from __future__ import annotations

import pytest
import torch

from foundations.projects.transformer.attention import scaled_dot_product_attention
from masks import make_causal_mask, make_padding_mask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, H, T, D = 2, 4, 8, 16  # batch, heads, seq_len, head_dim


@pytest.fixture()
def qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random Q, K, V tensors with shape ``(B, H, T, D)``."""
    gen = torch.Generator().manual_seed(42)
    q = torch.randn(B, H, T, D, generator=gen)
    k = torch.randn(B, H, T, D, generator=gen)
    v = torch.randn(B, H, T, D, generator=gen)
    return q, k, v


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestScaledDotProductAttentionShapes:
    """Verify output shapes match the multi-head layout."""

    def test_context_shape(self, qkv: tuple) -> None:
        """Context tensor must have shape ``(B, H, T, D)``."""
        q, k, v = qkv
        context, _ = scaled_dot_product_attention(q, k, v)

        assert context.shape == (B, H, T, D)

    def test_attention_weights_shape(self, qkv: tuple) -> None:
        """Attention weights must have shape ``(B, H, T, T)``."""
        q, k, v = qkv
        _, attn_weights = scaled_dot_product_attention(q, k, v)

        assert attn_weights.shape == (B, H, T, T)


# ---------------------------------------------------------------------------
# Mask tests
# ---------------------------------------------------------------------------


class TestAttentionRespectsCausalMask:
    """Causal mask must zero out attention to future positions."""

    def test_future_weights_are_zero(self, qkv: tuple) -> None:
        """Attention weights at ``(i, j)`` with ``j > i`` must be ≈ 0.

        We apply a causal mask of ``-inf`` to logits before softmax,
        so the resulting weights should be exactly 0.0 (or within
        floating-point tolerance).
        """
        q, k, v = qkv
        causal_mask = make_causal_mask(seq_len=T)  # (T, T)
        _, attn_weights = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # Extract the strict upper triangle (future positions)
        future_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        future_weights = attn_weights[:, :, future_mask]  # (B*H, num_future)

        assert torch.all(future_weights < 1e-9), (
            f"Max future weight: {future_weights.max().item():.2e} — expected < 1e-9"
        )

    def test_past_and_present_weights_are_positive(self, qkv: tuple) -> None:
        """Non-masked positions should receive non-zero attention."""
        q, k, v = qkv
        causal_mask = make_causal_mask(seq_len=T)
        _, attn_weights = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # The diagonal (self-attention) should always be > 0
        for i in range(T):
            diag_weight = attn_weights[:, :, i, i]
            assert torch.all(diag_weight > 0), f"Diagonal weight at position {i} should be > 0"


class TestAttentionRespectsPaddingMask:
    """Padding mask must zero out attention to ``<pad>`` positions."""

    def test_padded_positions_get_zero_attention(self) -> None:
        """Attention weights assigned to pad positions must be ≈ 0.

        Setup: batch of 1, 4 heads, sequence ``[real, real, pad, pad]``.
        No query should attend to positions 2 or 3.
        """
        seq_len = 4
        gen = torch.Generator().manual_seed(7)
        q = torch.randn(1, H, seq_len, D, generator=gen)
        k = torch.randn(1, H, seq_len, D, generator=gen)
        v = torch.randn(1, H, seq_len, D, generator=gen)

        token_ids = torch.tensor([[5, 3, 0, 0]])  # positions 2, 3 padded
        pad_mask = make_padding_mask(token_ids, pad_id=0)  # (1, 1, 1, 4)

        _, attn_weights = scaled_dot_product_attention(q, k, v, mask=pad_mask)

        # All weights pointing to pad columns (indices 2, 3) must be ≈ 0
        pad_weights = attn_weights[:, :, :, 2:]  # (1, H, 4, 2)
        assert torch.all(pad_weights < 1e-9), (
            f"Max pad-column weight: {pad_weights.max().item():.2e} — expected < 1e-9"
        )

    def test_real_tokens_receive_all_attention(self) -> None:
        """When pad columns are zeroed, real tokens should absorb all weight.

        Attention weight rows must still sum to 1.0, concentrated on
        the non-padded positions.
        """
        seq_len = 4
        gen = torch.Generator().manual_seed(7)
        q = torch.randn(1, H, seq_len, D, generator=gen)
        k = torch.randn(1, H, seq_len, D, generator=gen)
        v = torch.randn(1, H, seq_len, D, generator=gen)

        token_ids = torch.tensor([[5, 3, 0, 0]])
        pad_mask = make_padding_mask(token_ids, pad_id=0)

        _, attn_weights = scaled_dot_product_attention(q, k, v, mask=pad_mask)

        # Weights on real columns (0, 1) should account for ~all mass
        real_weights = attn_weights[:, :, :, :2]  # (1, H, 4, 2)
        row_sums = real_weights.sum(dim=-1)

        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


# ---------------------------------------------------------------------------
# Probability & gradient sanity
# ---------------------------------------------------------------------------


class TestAttentionProperties:
    """General correctness properties of scaled dot-product attention."""

    def test_weights_sum_to_one(self, qkv: tuple) -> None:
        """Each attention-weight row must sum to 1 (softmax output)."""
        q, k, v = qkv
        _, attn_weights = scaled_dot_product_attention(q, k, v)

        row_sums = attn_weights.sum(dim=-1)  # (B, H, T)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_gradient_flows_through_qkv(self) -> None:
        """Gradients must propagate back to Q, K, and V."""
        q = torch.randn(B, H, T, D, requires_grad=True)
        k = torch.randn(B, H, T, D, requires_grad=True)
        v = torch.randn(B, H, T, D, requires_grad=True)

        context, _ = scaled_dot_product_attention(q, k, v)
        context.sum().backward()

        for name, param in [("Q", q), ("K", k), ("V", v)]:
            assert param.grad is not None, f"No gradient on {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient on {name}"
