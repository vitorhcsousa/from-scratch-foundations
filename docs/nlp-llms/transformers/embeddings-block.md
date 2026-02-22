# Embeddings Block (Token + Position + Dropout)

## Goal
A single reusable module that converts token ids into model inputs.

- input: token ids `(B, L)`
- output: embeddings `(B, L, D)`

## Why it exists
Centralizes:
- `L <= max_len` guard
- positional choice (sinusoidal vs learned)
- dropout application
- predictable dtype/device behavior

## Interface
`forward(token_ids: LongTensor[B, L]) -> FloatTensor[B, L, D]`

## Tests to keep forever
- correct output shape
- raises on `L > max_len`
- deterministic in `.eval()` with dropout=0