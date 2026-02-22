# Learned Positional Embeddings

> **A trainable lookup table** where each position gets its own independent
> vector — more expressive than sinusoidal, but capped at `max_len`.

## Why a learned embedding

The sinusoidal scheme encodes position with a fixed formula that cannot
adapt to the data. A learned embedding lets the model discover whatever
positional patterns the training distribution rewards — relative spacing,
boundary effects, attention sinks — all without hand-crafted frequencies.

This is the approach used in BERT, GPT-2, and ViT.

## Math

Each position \(t\) maps to a row in a trainable matrix:

\[
\mathbf{p}_t = \mathbf{E}[t], \quad \mathbf{E} \in \mathbb{R}^{\text{max\_len} \times d_{\text{model}}}
\]

The position vector is then added to the token embedding:

\[
\mathbf{x}_t = \mathbf{e}(w_t) + \mathbf{p}_t
\]

## Trade-offs

| | Sinusoidal (fixed) | Learned |
|---|---|---|
| **Adapts to data** | No | Yes |
| **Extra parameters** | 0 | \(\text{max\_len} \times d_{\text{model}}\) |
| **Extrapolation** | Works for any position | Undefined past `max_len` |
| **Simplicity** | Formula, no training needed | Standard `nn.Embedding` |

## Design decisions

- **Fail fast on out-of-range positions.** If \(L > \text{max\_len}\) the
  module raises `ValueError` immediately. Silently wrapping or clamping
  indices would produce garbage embeddings that are hard to debug.
- **Same interface as `SinusoidalPositionalEncoding`.** Constructor takes
  `(d_model, max_len, dropout)`, forward takes `(B, L, D)` and returns
  `(B, L, D)`. This makes the two modules drop-in interchangeable — see
  [[positional-encoding-implementation]].
- **Long-context strategies are out of scope (for now).** Techniques like
  ALiBi, RoPE, or position interpolation solve the `max_len` ceiling but
  belong in dedicated modules, not here.

## Minimal usage

```python
from foundations.projects.transformer.positional_encoding import LearnedPositionalEmbedding

pos = LearnedPositionalEmbedding(d_model=512, max_len=1024, dropout=0.1)
x = torch.randn(2, 128, 512)   # (B, L, D)
out = pos(x)                     # (B, L, 512) — position added + dropout
```

## Links

- [[positional-encoding]] — conceptual overview (sinusoidal vs learned)
- [[positional-encoding-implementation]] — engineering contract both modules share
- [[embeddings-block]] — the combined token + position + dropout module