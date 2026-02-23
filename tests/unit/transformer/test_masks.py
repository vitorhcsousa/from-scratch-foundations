"""Tests for transformer mask utilities.

Covers two mask types used in the Transformer architecture:

    Padding mask:
        Prevents attention to ``<pad>`` tokens introduced by batching
        sequences of different lengths.  Positions corresponding to
        ``pad_id`` are masked (filled with ``-inf`` or ``False``
        depending on convention).

    Causal mask:
        Prevents each position from attending to future (right-side)
        positions.  Required by the decoder during training so it
        cannot "cheat" by looking ahead.

The tests below lock down shapes and semantic correctness for
``make_padding_mask`` and ``make_causal_mask`` before the
implementations are written (TDD).
"""

from __future__ import annotations

import torch

from masks import make_causal_mask, make_padding_mask

# ---------------------------------------------------------------------------
# Padding mask
# ---------------------------------------------------------------------------


class TestMakePaddingMask:
    """Tests for ``make_padding_mask``."""

    def test_shape_broadcastable_over_heads(self) -> None:
        """Returned mask must broadcast to ``(B, H, T, T)`` attention logits.

        Expected shape: ``(B, 1, 1, T)`` — singleton dims for head and
        query-position so one mask broadcasts across all heads and all
        query positions.
        """
        # Batch of 2 sequences, length 6, pad_id = 0
        token_ids = torch.tensor(
            [
                [5, 12, 8, 3, 0, 0],  # last 2 positions are padding
                [7, 2, 0, 0, 0, 0],  # last 4 positions are padding
            ]
        )
        mask = make_padding_mask(token_ids, pad_id=0)

        assert mask.shape == (2, 1, 1, 6)

    def test_pad_positions_are_masked(self) -> None:
        """Pad positions must be ``-inf``; real tokens must be ``0.0``.

        Convention: the mask is *added* to attention logits before softmax,
        so ``-inf`` drives the softmax output to ≈ 0 for pad positions.
        """
        token_ids = torch.tensor([[4, 9, 0, 0]])  # positions 2, 3 are pad
        mask = make_padding_mask(token_ids, pad_id=0)

        # Real token positions → 0.0 (no effect when added to logits)
        assert mask[0, 0, 0, 0].item() == 0.0
        assert mask[0, 0, 0, 1].item() == 0.0

        # Pad positions → -inf
        assert mask[0, 0, 0, 2].item() == float("-inf")
        assert mask[0, 0, 0, 3].item() == float("-inf")

    def test_no_padding_yields_all_zeros(self) -> None:
        """When no tokens match ``pad_id`` the mask should be all zeros."""
        token_ids = torch.tensor([[1, 2, 3]])
        mask = make_padding_mask(token_ids, pad_id=0)

        assert torch.all(mask == 0.0)

    def test_all_padding_yields_all_neg_inf(self) -> None:
        """Edge case: every token is padding."""
        token_ids = torch.tensor([[0, 0, 0]])
        mask = make_padding_mask(token_ids, pad_id=0)

        assert torch.all(mask == float("-inf"))

    def test_custom_pad_id(self) -> None:
        """``pad_id`` other than 0 should work identically."""
        token_ids = torch.tensor([[1, 2, 99, 99]])
        mask = make_padding_mask(token_ids, pad_id=99)

        assert mask[0, 0, 0, 0].item() == 0.0
        assert mask[0, 0, 0, 1].item() == 0.0
        assert mask[0, 0, 0, 2].item() == float("-inf")
        assert mask[0, 0, 0, 3].item() == float("-inf")


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------


class TestMakeCausalMask:
    """Tests for ``make_causal_mask``."""

    def test_shape_is_square(self) -> None:
        """Causal mask shape must be ``(T, T)``."""
        mask = make_causal_mask(seq_len=8)

        assert mask.shape == (8, 8)

    def test_future_positions_are_blocked(self) -> None:
        """All positions ``(i, j)`` with ``j > i`` must be ``-inf``."""
        t = 6
        mask = make_causal_mask(seq_len=t)

        for i in range(t):
            for j in range(i + 1, t):
                assert mask[i, j].item() == float("-inf"), (
                    f"Position ({i}, {j}) should be -inf (future), got {mask[i, j].item()}"
                )

    def test_diagonal_and_past_are_allowed(self) -> None:
        """All positions ``(i, j)`` with ``j <= i`` must be ``0.0``."""
        t = 6
        mask = make_causal_mask(seq_len=t)

        for i in range(t):
            for j in range(i + 1):
                assert mask[i, j].item() == 0.0, (
                    f"Position ({i}, {j}) should be 0.0 (past/current), got {mask[i, j].item()}"
                )

    def test_seq_len_one_is_single_zero(self) -> None:
        """Edge case: length-1 sequence has a ``(1, 1)`` zero mask."""
        mask = make_causal_mask(seq_len=1)

        assert mask.shape == (1, 1)
        assert mask[0, 0].item() == 0.0

    def test_dtype_is_float(self) -> None:
        """Mask must be a float tensor so it can be added to logits."""
        mask = make_causal_mask(seq_len=4)

        assert mask.is_floating_point()
