import torch
from projects.transformer.positional_encoding import (
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
