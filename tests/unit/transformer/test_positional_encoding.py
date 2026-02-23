import pytest
import torch
from src.foundations.projects.transformer.positional_encoding import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEncoding,
)


class TestBuildTable:
    """Tests for the raw sin/cos table builder."""

    def test_shape(self) -> None:
        """Output matches requested (seq_len, d_model)."""
        pe = SinusoidalPositionalEncoding.build_table(50, 128)
        assert pe.shape == (50, 128)

    def test_shape_small(self) -> None:
        """Smallest valid input (1 position, 2 dims) still works."""
        pe = SinusoidalPositionalEncoding.build_table(1, 2)
        assert pe.shape == (1, 2)

    def test_position0(self) -> None:
        """Row 0 must be [0, 1, 0, 1, …] (sin(0)=0, cos(0)=1)."""
        pe = SinusoidalPositionalEncoding.build_table(4, 16)
        row0 = pe[0]
        assert torch.allclose(row0[0::2], torch.zeros(8))
        assert torch.allclose(row0[1::2], torch.ones(8))

    def test_deterministic(self) -> None:
        """Two calls with the same args yield bit-identical tensors."""
        a = SinusoidalPositionalEncoding.build_table(32, 64)
        b = SinusoidalPositionalEncoding.build_table(32, 64)
        assert torch.allclose(a, b)

    def test_rejects_odd_d_model(self) -> None:
        """build_table must raise ValueError for odd d_model."""
        with pytest.raises(ValueError, match="d_model must be even"):
            SinusoidalPositionalEncoding.build_table(10, 15)


class TestSinusoidalModule:
    """Tests for the fixed sin/cos module wrapper."""

    def test_output_shape(self) -> None:
        """Output shape equals input shape."""
        layer = SinusoidalPositionalEncoding(d_model=64, dropout=0.0)
        x = torch.randn(2, 20, 64)
        assert layer(x).shape == (2, 20, 64)

    def test_zero_input_equals_pe_table(self) -> None:
        """Zero input → output is the raw PE table (dropout=0)."""
        layer = SinusoidalPositionalEncoding(d_model=16, dropout=0.0)
        out = layer(torch.zeros(1, 10, 16))
        expected = SinusoidalPositionalEncoding.build_table(10, 16)
        assert torch.allclose(out.squeeze(0), expected)

    def test_no_learnable_params(self) -> None:
        """Fixed encoding must add zero learnable parameters."""
        layer = SinusoidalPositionalEncoding(d_model=32)
        assert sum(p.numel() for p in layer.parameters()) == 0

    def test_buffer_registered(self) -> None:
        """The PE table must live in named_buffers as 'pe'."""
        layer = SinusoidalPositionalEncoding(d_model=32)
        names = [n for n, _ in layer.named_buffers()]
        assert "pe" in names

    def test_deterministic(self) -> None:
        """Eval mode (dropout off) → same input, same output."""
        layer = SinusoidalPositionalEncoding(d_model=32, dropout=0.5)
        layer.eval()
        x = torch.randn(1, 10, 32)
        assert torch.allclose(layer(x), layer(x))

    def test_raises_when_seq_len_exceeds_max_len(self) -> None:
        """Forward must raise ValueError when seq_len > max_len."""
        layer = SinusoidalPositionalEncoding(d_model=16, max_len=10, dropout=0.0)
        x = torch.randn(1, 20, 16)  # seq_len=20 > max_len=10
        with pytest.raises(ValueError, match="exceeds max_len"):
            layer(x)

    def test_dtype_preserved_fp16(self) -> None:
        """Output dtype must match input dtype (fp16 stays fp16)."""
        layer = SinusoidalPositionalEncoding(d_model=16, dropout=0.0)
        x = torch.randn(1, 5, 16, dtype=torch.float16)
        out = layer(x)
        assert out.dtype == torch.float16

    def test_dtype_preserved_bf16(self) -> None:
        """Output dtype must match input dtype (bf16 stays bf16)."""
        layer = SinusoidalPositionalEncoding(d_model=16, dropout=0.0)
        x = torch.randn(1, 5, 16, dtype=torch.bfloat16)
        out = layer(x)
        assert out.dtype == torch.bfloat16


class TestLearnedModule:
    """Tests for the learnable positional embedding module."""

    def test_output_shape(self) -> None:
        """Output shape equals input shape."""
        layer = LearnedPositionalEmbedding(d_model=64, dropout=0.0)
        x = torch.randn(2, 20, 64)
        assert layer(x).shape == (2, 20, 64)

    def test_zero_input_equals_embedding(self) -> None:
        """Zero input → output is the raw embedding lookup (dropout=0)."""
        layer = LearnedPositionalEmbedding(d_model=16, dropout=0.0)
        out = layer(torch.zeros(1, 10, 16))
        expected = layer.embedding(torch.arange(10))
        assert torch.allclose(out.squeeze(0), expected)

    def test_has_learnable_params(self) -> None:
        """Embedding table must contain max_len x d_model parameters."""
        layer = LearnedPositionalEmbedding(d_model=32, max_len=100)
        assert sum(p.numel() for p in layer.parameters()) == 100 * 32

    def test_params_require_grad(self) -> None:
        """Embedding parameters must be trainable (requires_grad=True)."""
        layer = LearnedPositionalEmbedding(d_model=32, max_len=100)
        assert any(p.requires_grad for p in layer.parameters())

    def test_embedding_table_size(self) -> None:
        """Internal embedding weight has shape (max_len, d_model)."""
        layer = LearnedPositionalEmbedding(d_model=64, max_len=200)
        assert layer.embedding.weight.shape == (200, 64)

    def test_deterministic_in_eval(self) -> None:
        """Eval mode (dropout off) → same input, same output."""
        layer = LearnedPositionalEmbedding(d_model=32, dropout=0.5)
        layer.eval()
        x = torch.randn(1, 10, 32)
        assert torch.allclose(layer(x), layer(x))

    def test_raises_when_seq_len_exceeds_max_len(self) -> None:
        """Forward must raise ValueError when seq_len > max_len."""
        layer = LearnedPositionalEmbedding(d_model=16, max_len=10, dropout=0.0)
        x = torch.randn(1, 20, 16)  # seq_len=20 > max_len=10
        with pytest.raises(ValueError, match="exceeds max_len"):
            layer(x)

    def test_dtype_preserved_fp16(self) -> None:
        """Output dtype must match input dtype (fp16 stays fp16)."""
        layer = LearnedPositionalEmbedding(d_model=16, max_len=50, dropout=0.0)
        x = torch.randn(1, 5, 16, dtype=torch.float16)
        out = layer(x)
        assert out.dtype == torch.float16
