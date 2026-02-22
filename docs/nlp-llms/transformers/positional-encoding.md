# Positional Encoding

Self-attention alone does not encode token order. Positional encoding injects order information into token representations.

A common absolute approach adds a position vector to the token embedding:

$$
\mathbf{x}_t = \mathbf{e}(w_t) + \mathbf{p}_t
$$

Where:
- $w_t$ is the token at position $t$
- $\mathbf{e}(w_t) \in \mathbb{R}^{d_{\text{model}}}$ is its token embedding
- $\mathbf{p}_t \in \mathbb{R}^{d_{\text{model}}}$ encodes position

## Two common absolute methods
### Sinusoidal (fixed)
- Deterministic, parameter-free
- Often extrapolates better to longer lengths
- Encodes position with sin/cos waves at multiple frequencies

### Learned (trainable)
- Trainable embedding table
- Can fit training distribution well
- Limited by `max_len` unless explicitly designed otherwise

## Repo contract used here
All positional modules follow:
- Input: $x \in \mathbb{R}^{B \times L \times D}$
- Output: same shape after adding position

See: [Implementation Notes](positional-encoding-implementation.md).