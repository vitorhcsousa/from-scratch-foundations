# Learned Positional Embeddings

A learned positional embedding is a trainable lookup table mapping positions to vectors:

$$
\mathbf{p}_t = \mathbf{E}[t], \quad \mathbf{E} \in \mathbb{R}^{\text{max\_len} \times d_{\text{model}}}
$$

## Pros
- can adapt to the training distribution
- simple, fast, widely used

## Cons / constraints
- requires choosing `max_len`
- extrapolation past training length is non-trivial (out-of-range positions)

## Engineering rule
Fail fast when `seq_len > max_len` unless you intentionally implement a long-context strategy.