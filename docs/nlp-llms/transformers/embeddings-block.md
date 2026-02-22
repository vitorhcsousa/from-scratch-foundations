# Embeddings Block

> **One module** that replaces the repeated "token embed → pos encode → dropout"
> boilerplate found in every Transformer variant.

## Why this abstraction exists

Every Transformer (encoder, decoder, encoder-decoder) starts the same way:

1. Look up token embeddings → `nn.Embedding(vocab_size, d_model)`
2. Add positional information → sinusoidal or learned
3. Apply dropout

Without a shared block you copy-paste these three lines into every model
and inevitably drift (different dropout placement, forgetting the `padding_idx`,
swapping pos strategies mid-experiment without updating both encoder *and*
decoder, etc.).

`TokenAndPositionalEmbedding` encapsulates the three steps behind a single
`forward(token_ids) → embeddings` call, so the rest of the model never
thinks about embedding logistics.

## Contract

| | |
|---|---|
| **Input** | `(B, L)` — `torch.long` token ids |
| **Output** | `(B, L, d_model)` — `torch.float` embeddings |
| **Guard** | \(L > \text{max\_len}\) → `ValueError` |

## Constructor

```python
TokenAndPositionalEmbedding(
    vocab_size: int,
    d_model: int,
    *,
    max_len: int,
    pos_kind: Literal["sinusoidal", "learned"],
    dropout: float = 0.0,
    pad_id: int | None = None,
)
```

## `pos_kind` strategies

| Value | Class used | Learnable? | Extra params |
|---|---|---|---|
| `"sinusoidal"` | `SinusoidalPositionalEncoding` | No | 0 |
| `"learned"` | `LearnedPositionalEmbedding` | Yes | \(\text{max\_len} \times d_{\text{model}}\) |

Both positional modules share the same `__init__` / `forward` signature, so
the embedding block can instantiate either one with identical arguments — see
[[positional-encoding-implementation]].

## Design decisions

- **Dropout lives here, not inside the positional module.** The positional
  modules are kept pure (they *add* the signal and nothing else). Dropout
  is applied once *after* the sum `token + position`, matching the original
  Transformer paper.
- **`pad_id` flows to `nn.Embedding(padding_idx=...)`**, which zeros the
  padding vector and excludes it from gradient updates automatically.
- **`_POS_REGISTRY` dict** maps string keys to classes, making it trivial
  to add a third strategy (e.g. `"rotary"`) later without touching the
  constructor logic.

## Minimal usage

```python
from foundations.projects.transformer.embeddings import TokenAndPositionalEmbedding

emb = TokenAndPositionalEmbedding(
    vocab_size=30_000,
    d_model=512,
    max_len=1024,
    pos_kind="sinusoidal",
    dropout=0.1,
    pad_id=0,
)

ids = torch.randint(0, 30_000, (2, 128))   # (B, L)
x = emb(ids)                                # (B, L, 512)
```

## Links

- [[positional-encoding]] — conceptual overview (why position matters)
- [[positional-encoding-implementation]] — the two positional modules this block depends on
- [[learned-positional-embeddings]] — deeper look at the trainable variant
- [[testing-numerics]] — why tests use `allclose` instead of `==`
- `src/foundations/projects/transformer/embeddings.py` — implementation
- `tests/test_embeddings.py` — 10 tests (shape, guards, determinism, padding)