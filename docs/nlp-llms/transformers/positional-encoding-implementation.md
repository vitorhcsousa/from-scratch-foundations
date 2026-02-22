# Positional Encoding — Implementation Notes

This documents the exact engineering contract used in the codebase.

## Contract
- Input: `x` with shape `(B, L, D)`
- Output: tensor with shape `(B, L, D)`

## Fail-fast length guard
Enforce:

$$
L \le \text{max\_len}
$$

Raise `ValueError` if violated (clear error message beats silent slicing / cryptic index errors).

## Dtype rule (mixed precision)
Before adding PE to `x`, cast the PE slice to `x.dtype` and `x.device`:

- avoids silent fp16/bf16 → fp32 upcasting
- keeps performance predictable
- reduces test brittleness

## Dropout + testing
To make tests stable:
- use `dropout=0.0`, or
- call `.eval()` during assertions

## Minimal test checklist
- shape preserved
- raises on `L > max_len`
- dtype preserved
- determinism in `.eval()` (same seed)
- learned module has trainable params (`requires_grad=True`)