# Positional Encoding

> **Self-attention is permutation-invariant** — without an explicit position
> signal, the model cannot distinguish "the cat sat" from "sat the cat".

## Why positional encoding exists

The dot-product attention mechanism treats its inputs as a *set*: shuffling
the token order produces identical attention weights. Positional encoding
breaks this symmetry by injecting order information into every token
representation before the first attention layer.

The standard absolute approach adds a position vector to the token embedding:

\[
\mathbf{x}_t = \mathbf{e}(w_t) + \mathbf{p}_t
\]

| Symbol | Meaning |
|---|---|
| \(w_t\) | Token at position \(t\) |
| \(\mathbf{e}(w_t) \in \mathbb{R}^{d_{\text{model}}}\) | Token embedding |
| \(\mathbf{p}_t \in \mathbb{R}^{d_{\text{model}}}\) | Position vector |

## Two common absolute methods

| Method | Source | Learnable? | Extrapolates? |
|---|---|---|---|
| Sinusoidal (fixed) | Vaswani et al., 2017 | No | Yes — formula works for any position |
| Learned embedding | BERT / GPT-2 | Yes | No — capped at `max_len` |

- **Sinusoidal** encodes each position with sin/cos waves at geometrically
  spaced frequencies. Deterministic, parameter-free, and often generalises
  better to unseen lengths. See [[positional-encoding-implementation]].
- **Learned** uses a trainable `nn.Embedding` table. More expressive (each
  position gets an independent vector) but limited by the pre-allocated
  `max_len`. See [[learned-positional-embeddings]].

## Repo contract

All positional modules in this codebase follow the same interface:

| | |
|---|---|
| **Input** | `(B, L, d_model)` — float embeddings |
| **Output** | `(B, L, d_model)` — same shape after adding position |
| **Guard** | `L > max_len` → `ValueError` |

This makes them drop-in interchangeable inside
[[embeddings-block|TokenAndPositionalEmbedding]].

## Links

- [[positional-encoding-implementation]] — the two `nn.Module` classes and their engineering details
- [[learned-positional-embeddings]] — deeper look at the trainable variant
- [[embeddings-block]] — the combined token + position + dropout module
- [[testing-numerics]] — why tests use `allclose` instead of `==`