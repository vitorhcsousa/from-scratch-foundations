# Positional Encoding — Implementation

> **Two interchangeable `nn.Module` classes** with identical signatures — swap
> sinusoidal for learned (or vice versa) with zero code changes.

## Why two modules with one interface

Experiments often compare fixed vs learned position signals. If the two
implementations share the same `__init__` / `forward` contract, the rest of
the model (and every test) works unchanged — only the constructor argument
changes.

## Contract

Both `SinusoidalPositionalEncoding` and `LearnedPositionalEmbedding` obey:

| | |
|---|---|
| **Input** | `(B, L, d_model)` — float embeddings |
| **Output** | `(B, L, d_model)` — same shape, position signal added |
| **Guard** | \(L > \text{max\_len}\) → `ValueError` |

## Constructor

```python
SinusoidalPositionalEncoding(d_model: int, max_len: int = 5000, dropout: float = 0.1)
LearnedPositionalEmbedding(d_model: int, max_len: int = 5000, dropout: float = 0.1)
```

| Class | Learnable? | Extra params | Extrapolates? |
|---|---|---|---|
| `SinusoidalPositionalEncoding` | No | 0 | Yes |
| `LearnedPositionalEmbedding` | Yes | \(\text{max\_len} \times d_{\text{model}}\) | No |

## Design decisions

- **Fail-fast length guard.** Both modules enforce \(L \le \text{max\_len}\)
  and raise `ValueError` if violated. A clear error beats silent slicing or
  cryptic index-out-of-range crashes.
- **Dtype / device cast before addition.** The PE slice is cast to
  `x.dtype` and `x.device` before adding, which avoids silent fp16/bf16 →
  fp32 upcasting, keeps performance predictable, and reduces test
  brittleness.
- **`build_table` lives as a `@staticmethod`.** The sinusoidal formula is
  pure computation with no learnable state, so it sits inside
  `SinusoidalPositionalEncoding` as a static method — callable without
  instantiating the module, but scoped to the class that owns the logic.
- **Buffer vs parameter.** The sinusoidal table is registered via
  `register_buffer` (travels with `.to(device)` / `.half()`, appears in
  `state_dict`, but is never updated by the optimiser). The learned table
  uses `nn.Embedding`, whose weights *are* parameters.

## Minimal test checklist

- Shape preserved: `(B, L, D)` in → `(B, L, D)` out
- Raises on \(L > \text{max\_len}\)
- Dtype preserved after addition
- Deterministic in `.eval()` mode (same input → same output)
- Sinusoidal: zero learnable params
- Learned: `requires_grad=True` on embedding weights
- See [[testing-numerics]] for why tests use `allclose`

## Links

- [[positional-encoding]] — conceptual overview (why position matters)
- [[learned-positional-embeddings]] — deeper look at the trainable variant
- [[embeddings-block]] — the combined token + position + dropout module
- `src/foundations/projects/transformer/positional_encoding.py` — implementation
- `tests/test_positional_encoding.py` — test suite