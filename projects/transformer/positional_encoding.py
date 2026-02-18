"""Positional encoding utilities for Transformer models.

Two public symbols are provided:

    SinusoidalPositionalEncoding:
        ``nn.Module`` that adds the **fixed** sin/cos signal from Vaswani et al.
        "Attention Is All You Need" (NeurIPS 2017, Section 3.5).  Zero
        learnable parameters; can generalise to unseen sequence lengths.
        Exposes a ``build_table`` static method for direct table access.

    LearnedPositionalEmbedding:
        ``nn.Module`` that adds a **learnable** position embedding (BERT /
        GPT-2 / ViT style).  Each position gets an independent trainable
        vector, adding ``max_len × d_model`` parameters.

Both modules share the **same constructor signature and forward contract**,
so they can be swapped with zero code changes::

    self.pos_enc = SinusoidalPositionalEncoding(d_model=512)
    # or
    self.pos_enc = LearnedPositionalEmbedding(d_model=512)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos positional encoding (Vaswani et al., 2017).

    The encoding table is computed once at construction via ``build_table``
    and registered as a non-learnable buffer.  Adds **zero** parameters.

    Args:
        d_model: Embedding / model dimension. Must be even.
        max_len: Maximum sequence length. The buffer is pre-allocated at
            this size; shorter sequences index into the first rows.
            Defaults to 5000.
        dropout: Dropout probability applied after the addition.
            Defaults to 0.1.

    Shapes:
        - Input:  ``(batch, seq_len, d_model)``
        - Output: ``(batch, seq_len, d_model)``
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = self.build_table(max_len, d_model)      # (max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    @staticmethod
    def build_table(
        seq_len: int,
        d_model: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Build a fixed sinusoidal positional-encoding table.

        Encoding scheme (for position *pos* and dimension index *i*)::

            PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

        Args:
            seq_len: Number of token positions (rows).
            d_model: Model dimension (columns). **Must be even.**
            device: Target device. Defaults to current default device.
            dtype: Floating-point dtype. Defaults to ``torch.float32``.

        Returns:
            Tensor of shape ``(seq_len, d_model)``.
            Even columns contain ``sin``; odd columns contain ``cos``.

        Raises:
            ValueError: If ``d_model`` is not even.
        """
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        _dtype = dtype or torch.float32

        # pos → (seq_len, 1)
        pos = torch.arange(seq_len, device=device, dtype=_dtype).unsqueeze(1)

        # i → (1, d_model // 2)
        i = torch.arange(d_model // 2, device=device, dtype=_dtype).unsqueeze(0)

        # angles → (seq_len, d_model // 2)
        angles = pos / (10_000.0 ** (2.0 * i / d_model))

        pe = torch.zeros(seq_len, d_model, device=device, dtype=_dtype)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Add fixed positional encoding to the input embeddings.

        Args:
            x: Token embeddings ``(batch, seq_len, d_model)``.
                ``seq_len`` must be ≤ ``max_len``.

        Returns:
            ``x + PE[:seq_len]``, followed by dropout. Same shape as input.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learnable positional embedding (BERT / GPT-2 / ViT style).

    Each position in ``[0, max_len)`` is mapped to a trainable dense vector
    via ``nn.Embedding``.  Adds ``max_len × d_model`` learnable parameters.

    Args:
        d_model: Embedding / model dimension.
        max_len: Maximum sequence length. Defaults to 5000.
        dropout: Dropout probability applied after the addition.
            Defaults to 0.1.

    Shapes:
        - Input:  ``(batch, seq_len, d_model)``
        - Output: ``(batch, seq_len, d_model)``
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Add learned positional embeddings to the input.

        Args:
            x: Token embeddings ``(batch, seq_len, d_model)``.
                ``seq_len`` must be ≤ ``max_len``.

        Returns:
            ``x + Embedding[0 … seq_len-1]``, followed by dropout.
            Same shape as input.
        """
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.embedding(positions)
        return self.dropout(x)