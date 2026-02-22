# Testing Numerics (`allclose`)

> **Exact equality is the wrong tool for floats** — use `torch.allclose` with
> explicit tolerances so tests stay green across hardware and dtypes.

## Why exact equality breaks

Floating-point arithmetic is not associative. Two mathematically identical
computations can produce slightly different bits depending on evaluation
order, fused-multiply-add availability, and GPU kernel selection. Testing
with `==` rejects correct results over noise in the last few ULPs.

## What `allclose` checks

Elementwise, the assertion passes when:

\[
|a - b| \le \text{atol} + \text{rtol} \cdot |b|
\]

| Parameter | Role | Default (`torch`) |
|---|---|---|
| `atol` | Absolute tolerance — dominates near zero | `1e-8` |
| `rtol` | Relative tolerance — scales with magnitude | `1e-5` |

## When to use which

| Assertion | Use for |
|---|---|
| Exact `==` | Integers, shapes, masks, indices, string/enum comparisons |
| `allclose` | Anything that touches floating-point math: sin/cos, matmuls, reductions, softmax, GPU ops |

## Design decisions

- **Always pass `atol` / `rtol` explicitly in new tests.** Relying on
  defaults works today but breaks silently when someone switches from
  fp32 to bf16 (which needs looser tolerances).
- **Keep a project-wide constant if tolerances recur.** One `ATOL`, `RTOL`
  pair in a `conftest.py` fixture avoids magic numbers scattered across
  dozens of test files.
- **Disable randomness before asserting.** Set `dropout=0.0` or call
  `.eval()` so the only variation left is floating-point noise, not
  stochastic masking — see [[positional-encoding-implementation#minimal-test-checklist]].

## Minimal usage

```python
import torch

a = torch.sin(torch.tensor(3.14159))
b = torch.tensor(0.0)

# default tolerances — fine for fp32
assert torch.allclose(a, b, atol=1e-5)

# tighter check when you know values are exact
pe1 = build_table(32, 64)
pe2 = build_table(32, 64)
assert torch.allclose(pe1, pe2)          # deterministic → zero diff
```

## Links

- [[positional-encoding-implementation]] — test checklist that relies on `allclose`
- [[embeddings-block]] — integration tests using `allclose` for shape + value checks