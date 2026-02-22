"""Tests for the TokenAndPositionalEmbedding block.

Verifies the forward contract, guard rails, and determinism for both
positional encoding strategies.
"""

import pytest
import torch

from tfs.nn.embeddings import TokenAndPositionalEmbedding

# ── Shared fixtures ──────────────────────────────────────────────────────────

VOCAB, D_MODEL, MAX_LEN = 256, 64, 50
BATCH, SEQ_LEN = 2, 20


def _make_block(pos_kind: str = "sinusoidal", **kwargs) -> TokenAndPositionalEmbedding:
    """Convenience factory with sensible defaults."""
    defaults = dict(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        pos_kind=pos_kind,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return TokenAndPositionalEmbedding(**defaults)


def _sample_ids(batch: int = BATCH, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Random token-id tensor."""
    return torch.randint(0, VOCAB, (batch, seq_len))


# ── Shape ────────────────────────────────────────────────────────────────────


class TestEmbeddingsShape:
    """Output must be (B, L, d_model) for every pos_kind."""

    def test_shape_sinusoidal(self) -> None:
        """Sinusoidal variant produces the correct output shape."""
        block = _make_block("sinusoidal")
        out = block(_sample_ids())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_shape_learned(self) -> None:
        """Learned variant produces the correct output shape."""
        block = _make_block("learned")
        out = block(_sample_ids())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_output_dtype_is_float(self) -> None:
        """Output dtype must be float regardless of integer input."""
        block = _make_block()
        out = block(_sample_ids())
        assert out.dtype == torch.float32


# ── Guards ───────────────────────────────────────────────────────────────────


class TestEmbeddingsGuards:
    """Forward must reject invalid inputs early."""

    def test_raises_if_seq_len_exceeds_max_len(self) -> None:
        """Sequences longer than max_len must raise ValueError."""
        block = _make_block(max_len=10)
        ids = torch.randint(0, VOCAB, (1, 11))  # 11 > max_len=10
        with pytest.raises(ValueError, match="exceeds max_len"):
            block(ids)

    def test_exact_max_len_ok(self) -> None:
        """Sequence length == max_len must not raise."""
        block = _make_block(max_len=10)
        ids = torch.randint(0, VOCAB, (1, 10))
        out = block(ids)
        assert out.shape == (1, 10, D_MODEL)

    def test_invalid_pos_kind(self) -> None:
        """Unknown pos_kind must raise ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown pos_kind"):
            _make_block(pos_kind="rotary")


# ── Determinism ──────────────────────────────────────────────────────────────


class TestEmbeddingsDeterministic:
    """Same input → same output when randomness is removed."""

    def test_deterministic_sinusoidal_eval(self) -> None:
        """Sinusoidal block in eval mode (dropout=0) is deterministic."""
        block = _make_block("sinusoidal", dropout=0.0)
        block.eval()
        ids = _sample_ids()
        assert torch.allclose(block(ids), block(ids))

    def test_deterministic_learned_eval(self) -> None:
        """Learned block in eval mode (dropout=0) is deterministic."""
        block = _make_block("learned", dropout=0.0)
        block.eval()
        ids = _sample_ids()
        assert torch.allclose(block(ids), block(ids))


# ── Padding ──────────────────────────────────────────────────────────────────


class TestEmbeddingsPadding:
    """Padding token behaviour when pad_id is provided."""

    def test_pad_embedding_is_zero(self) -> None:
        """The embedding vector for pad_id must be all zeros."""
        pad_id = 0
        block = _make_block(pad_id=pad_id)
        assert torch.allclose(
            block.token.weight[pad_id],
            torch.zeros(D_MODEL),
        )