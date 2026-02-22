# Testing Numerics (`allclose`)

## Why exact equality is brittle
Floating-point computations can differ slightly due to:
- rounding/representation limits
- different computation order (non-associativity)
- different kernels/hardware

So exact equality (`==`) often fails for correct results.

## What `allclose` checks
Elementwise:

$$
|a-b| \le \text{atol} + \text{rtol}\cdot |b|
$$

- `atol`: absolute tolerance (important near zero)
- `rtol`: relative tolerance (scales with magnitude)

## Rule of thumb
- Use exact equality for integers, shapes, masks, indices.
- Use `allclose` for `sin/cos`, matmuls, reductions, GPU ops.