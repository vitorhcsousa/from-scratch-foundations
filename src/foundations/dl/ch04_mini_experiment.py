"""
ch04_mini_experiment.py
=======================
From-scratch 3-layer deep ReLU network (NumPy only).
Demonstrates Prince Ch.4: deep nets beat shallow via composition.

Task: learn f(x1, x2) = sin(x1) * cos(x2) — requires non-linear
      composition across multiple layers; a single-layer net struggles.

Architecture: 2 -> 8 -> 8 -> 1  (two hidden layers, linear output)
Training: vanilla gradient descent, 200 steps
Assertion: final loss < initial loss * 0.5

Usage:
    python ch04_mini_experiment.py
"""
from __future__ import annotations
import numpy as np

RNG = np.random.default_rng(42)

# ── data ─────────────────────────────────────────────────────────────────────

def make_data(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.uniform(-np.pi, np.pi, (n, 2))
    y = (np.sin(X[:, 0]) * np.cos(X[:, 1])).reshape(-1, 1)
    return X, y

# ── parameter init (He for ReLU layers) ──────────────────────────────────────

def init_params(dims: list[int]) -> list[dict[str, np.ndarray]]:
    params = []
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        w = RNG.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / fan_in)
        b = np.zeros((1, dims[i + 1]))
        params.append({"W": w, "b": b})
    return params

# ── activations ──────────────────────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

# ── forward pass ─────────────────────────────────────────────────────────────

def forward(
    X: np.ndarray,
    params: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Run forward pass; cache pre-activations and activations for backprop.

    Prince Ch.4: each layer h^(l) = phi(W^(l) h^(l-1) + b^(l)).
    Cache is required by backprop — never recompute forward values during backward.
    """
    cache: list[dict[str, np.ndarray]] = []
    h = X
    for i, p in enumerate(params):
        z = h @ p["W"] + p["b"]                 # pre-activation z^(l)
        is_last = (i == len(params) - 1)
        h_next = z if is_last else relu(z)       # linear output layer
        cache.append({"h_in": h, "z": z, "h_out": h_next})
        h = h_next
    return h, cache

# ── backward pass ─────────────────────────────────────────────────────────────

def backward(
    y_hat: np.ndarray,
    y: np.ndarray,
    params: list[dict[str, np.ndarray]],
    cache: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Backprop: compute dL/dW and dL/db for each layer.

    Prince Ch.4 derivation:
        delta^(L) = (y_hat - y) * 2/n          # MSE gradient at output
        delta^(l) = (delta^(l+1) @ W^(l+1).T) * phi'(z^(l))
        dL/dW^(l) = h^(l-1).T @ delta^(l)
    """
    n = y.shape[0]
    grads = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])} for p in params]

    delta = (2.0 / n) * (y_hat - y)            # output layer error signal

    for l in range(len(params) - 1, -1, -1):
        h_in = cache[l]["h_in"]
        z    = cache[l]["z"]

        grads[l]["W"] = h_in.T @ delta
        grads[l]["b"] = delta.sum(axis=0, keepdims=True)

        if l > 0:                               # propagate to previous layer
            delta = (delta @ params[l]["W"].T) * relu_grad(cache[l - 1]["z"])

    return grads

# ── gradient descent step ─────────────────────────────────────────────────────

def update(
    params: list[dict[str, np.ndarray]],
    grads: list[dict[str, np.ndarray]],
    lr: float = 0.01,
) -> list[dict[str, np.ndarray]]:
    return [{"W": p["W"] - lr * g["W"], "b": p["b"] - lr * g["b"]}
            for p, g in zip(params, grads)]

def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y_hat - y) ** 2))

# ── training loop ─────────────────────────────────────────────────────────────

def main() -> None:
    X, y = make_data(300)
    dims = [2, 8, 8, 1]
    params = init_params(dims)

    y_hat_init, _ = forward(X, params)
    loss_init = mse(y_hat_init, y)

    for step in range(200):
        y_hat, cache = forward(X, params)
        grads = backward(y_hat, y, params, cache)
        params = update(params, grads, lr=0.01)

    y_hat_final, _ = forward(X, params)
    loss_final = mse(y_hat_final, y)

    print(f"Loss initial : {loss_init:.6f}")
    print(f"Loss final   : {loss_final:.6f}")
    print(f"Reduction    : {loss_init / loss_final:.1f}x")
    assert loss_final < loss_init * 0.5, (
        f"Loss did not halve: {loss_init:.4f} -> {loss_final:.4f}"
    )
    print("OK")


if __name__ == "__main__":
    main()
