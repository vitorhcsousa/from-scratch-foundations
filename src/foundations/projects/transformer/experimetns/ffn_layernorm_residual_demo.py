"""
ffn_layernorm_residual_demo.py
==============================
From-scratch NumPy implementation of the core transformer sublayer stack:
  - Feed-Forward Network (position-wise MLP)
  - Layer Normalization
  - Residual connection

Covers exactly the components built this week (Wed–Thu):
  FFN (Ch03 / Vaswani 2017): expand → activate → contract
  LayerNorm (Ba et al. 2016): normalize features per token → learned γ/β
  Residual (He et al. 2015): x + sublayer(x)

The full pre-LN transformer sub-block pattern is:
  h' = h + FFN(LayerNorm(h))

Run:
    python ffn_layernorm_residual_demo.py   → prints shape checks + OK
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(42)


# ── Layer Normalization ───────────────────────────────────────────────────────

class LayerNorm:
    """Layer Normalization (Ba et al. 2016).

    Normalizes each token's feature vector across the feature dimension,
    then applies learned scale (gamma) and shift (beta).

    Normalization axis: feature dim (axis=-1), NOT batch or sequence.
    Each token is normalized independently → no inter-token dependency.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.gamma = np.ones(d_model)    # learned scale; init to 1
        self.beta  = np.zeros(d_model)   # learned shift; init to 0
        self.eps   = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply LayerNorm to input x.

        Args:
            x: Input of shape (..., d_model); last dim is feature dim.

        Returns:
            Normalized output, same shape as x.
        """
        mu    = x.mean(axis=-1, keepdims=True)           # mean over features
        var   = x.var(axis=-1, keepdims=True)            # variance over features
        x_hat = (x - mu) / np.sqrt(var + self.eps)      # normalize
        return self.gamma * x_hat + self.beta             # learned affine


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class FeedForwardNetwork:
    """Position-wise Feed-Forward Network (Vaswani et al. 2017).

    Applied identically and independently to each token position.
    Architecture: expand d_model → d_ff (4×) → activate (ReLU) → contract.

    No information flows between positions inside this module.
    Cross-token interaction is the attention sublayer's responsibility.
    """

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        if d_ff is None:
            d_ff = 4 * d_model          # standard 4× expansion ratio

        # He initialization for ReLU layers
        scale_1 = np.sqrt(2.0 / d_model)
        scale_2 = np.sqrt(2.0 / d_ff)

        self.W1 = RNG.standard_normal((d_model, d_ff))   * scale_1
        self.b1 = np.zeros(d_ff)
        self.W2 = RNG.standard_normal((d_ff, d_model))   * scale_2
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply FFN to each token position independently.

        Args:
            x: Input of shape (B, T, d_model) or (T, d_model).

        Returns:
            Output of same shape as x.
        """
        h = np.maximum(0.0, x @ self.W1 + self.b1)   # expand + ReLU
        return h @ self.W2 + self.b2                   # contract


# ── Residual wrapper ──────────────────────────────────────────────────────────

def residual(x: np.ndarray, sublayer_output: np.ndarray) -> np.ndarray:
    """Add residual connection: y = x + sublayer(x).

    Args:
        x:               Original input to the sublayer.
        sublayer_output: Output of the sublayer applied to x.

    Returns:
        x + sublayer_output; same shape as both inputs.
    """
    assert x.shape == sublayer_output.shape, (
        f"Shape mismatch: x {x.shape} vs sublayer {sublayer_output.shape}"
    )
    return x + sublayer_output


# ── Pre-LN transformer sub-block ─────────────────────────────────────────────

class TransformerFFNBlock:
    """One transformer FFN sub-block with Pre-LN convention.

    Forward:
        h_out = h + FFN(LayerNorm(h))

    This is the second of the two sub-blocks in a full transformer layer
    (the first being the attention sub-block).
    """

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        self.norm = LayerNorm(d_model)
        self.ffn  = FeedForwardNetwork(d_model, d_ff)

    def __call__(self, h: np.ndarray) -> np.ndarray:
        """Apply FFN sub-block with Pre-LN and residual.

        Args:
            h: Input of shape (B, T, d_model).

        Returns:
            Output of shape (B, T, d_model).
        """
        return residual(h, self.ffn(self.norm(h)))


# ── demo and verification ─────────────────────────────────────────────────────

def main() -> None:
    B, T, d_model, d_ff = 2, 8, 64, 256

    # ── LayerNorm: verify normalization per token ─────────────────────────────
    ln = LayerNorm(d_model)
    x  = RNG.standard_normal((B, T, d_model)) * 10.0 + 5.0   # shifted & scaled

    x_normed = ln(x)
    assert x_normed.shape == (B, T, d_model), "LayerNorm: wrong output shape"

    # After normalization, mean ≈ 0 and std ≈ 1 per token (before γ/β)
    # With γ=1, β=0 (init), output should have these stats
    token_means = x_normed.mean(axis=-1)         # (B, T)
    token_stds  = x_normed.std(axis=-1)          # (B, T)
    assert np.allclose(token_means, 0.0, atol=1e-5), "LayerNorm: mean ≠ 0"
    assert np.allclose(token_stds,  1.0, atol=1e-4), "LayerNorm: std ≠ 1"
    print(f"LayerNorm — output shape: {x_normed.shape}  "
          f"mean≈0: {np.allclose(token_means, 0, atol=1e-5)}  "
          f"std≈1: {np.allclose(token_stds, 1, atol=1e-4)}")

    # ── FFN: verify position-wise, shape preserved ────────────────────────────
    ffn  = FeedForwardNetwork(d_model, d_ff)
    h    = RNG.standard_normal((B, T, d_model))
    h_ff = ffn(h)
    assert h_ff.shape == (B, T, d_model), "FFN: wrong output shape"

    # Position-wise check: applying FFN to a single token == applying to batch
    single_token = h[0, 0, :]                    # (d_model,)
    single_out   = ffn(single_token)
    assert np.allclose(single_out, h_ff[0, 0, :], atol=1e-10), (
        "FFN: position-wise property violated"
    )
    print(f"FFN — output shape: {h_ff.shape}  position-wise: OK")

    # ── Residual: verify identity init, shape match ───────────────────────────
    h_res = residual(h, h_ff)
    assert h_res.shape == h.shape, "Residual: wrong shape"

    # If FFN output is zero (simulated), residual should equal input
    h_zero_ffn = residual(h, np.zeros_like(h))
    assert np.allclose(h_zero_ffn, h), "Residual: identity not preserved when sublayer=0"
    print(f"Residual — output shape: {h_res.shape}  identity check: OK")

    # ── Full Pre-LN FFN block ─────────────────────────────────────────────────
    block  = TransformerFFNBlock(d_model, d_ff)
    h_in   = RNG.standard_normal((B, T, d_model))
    h_out  = block(h_in)
    assert h_out.shape == h_in.shape, "TransformerFFNBlock: wrong output shape"

    # Gradient sanity: output should differ from input (block is non-trivial)
    assert not np.allclose(h_out, h_in), "Block output == input; FFN is trivial"
    print(f"TransformerFFNBlock — in: {h_in.shape}  out: {h_out.shape}  non-trivial: OK")

    # ── Verify LayerNorm normalizes per token, not across tokens ──────────────
    # Multiply token 0 by 100; if norm were across tokens, all would change
    h_scaled = h_in.copy()
    h_scaled[:, 0, :] *= 100.0
    out_original = block(h_in)
    out_scaled   = block(h_scaled)
    # Other tokens' outputs should not be affected
    assert not np.allclose(out_original[:, 0, :], out_scaled[:, 0, :]), \
        "Token 0 should differ after scaling"
    assert np.allclose(out_original[:, 1:, :], out_scaled[:, 1:, :], atol=1e-10), \
        "Other tokens should be unaffected (LayerNorm is per-token)"
    print("LayerNorm per-token isolation check: OK")

    print("\nOK")


if __name__ == "__main__":
    main()
