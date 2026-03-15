"""
ch03_shallow_nn_demo.py
=======================
A minimal, from-scratch shallow neural network (one hidden layer) using NumPy only.

Demonstrates:
  - Forward pass: input → hidden (ReLU) → output (linear)
  - Loss: Mean Squared Error
  - Backward pass: manual gradient computation
  - One gradient descent update step

Architecture: D_i=2 → D_h=4 → D_o=1
Task:         Approximate XOR (non-linearly separable → tests that the hidden
              layer actually learns a useful representation)

Run:
    python ch03_shallow_nn_demo.py   → prints loss before/after one step + OK
"""

from __future__ import annotations

import numpy as np

# ── reproducibility ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── data: XOR ─────────────────────────────────────────────────────────────────
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)  # (4, 2)
y = np.array([[0], [1], [1], [0]], dtype=float)               # (4, 1)

# ── architecture ──────────────────────────────────────────────────────────────
D_i, D_h, D_o = 2, 4, 1


# ── activation ────────────────────────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    """ReLU: max(0, z), element-wise."""
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """Subgradient of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(float)


# ── loss ──────────────────────────────────────────────────────────────────────

def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean Squared Error over a batch."""
    return float(np.mean((y_hat - y) ** 2))


# ── initialisation ────────────────────────────────────────────────────────────

def init_params(d_i: int, d_h: int, d_o: int) -> dict[str, np.ndarray]:
    """He initialisation for W1 (ReLU); small random for W2."""
    return {
        "W1": RNG.standard_normal((d_h, d_i)) * np.sqrt(2.0 / d_i),
        "b1": np.zeros((d_h, 1)),
        "W2": RNG.standard_normal((d_o, d_h)) * 0.01,
        "b2": np.zeros((d_o, 1)),
    }


# ── forward pass ──────────────────────────────────────────────────────────────

def forward(
    X: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute predictions and cache intermediate values for backprop.

    Args:
        X: Input matrix (n_samples, D_i).
        params: Weight dict with W1, b1, W2, b2.

    Returns:
        y_hat: Predictions (n_samples, D_o).
        cache: Intermediate values needed for backward pass.
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    X_T = X.T                   # (D_i, n)
    Z1 = W1 @ X_T + b1         # (D_h, n) — pre-activation
    H  = relu(Z1)               # (D_h, n) — hidden activations
    Z2 = W2 @ H  + b2          # (D_o, n) — output pre-activation
    y_hat = Z2.T                # (n, D_o) — predictions

    cache = {"X": X, "Z1": Z1, "H": H, "Z2": Z2}
    return y_hat, cache


# ── backward pass ─────────────────────────────────────────────────────────────

def backward(
    y_hat: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute gradients of MSE loss w.r.t. all parameters.

    Chain rule, manually derived:
        dL/dZ2 = (2/n)(y_hat - y)
        dL/dW2 = dL/dZ2 · H^T
        dL/dH  = W2^T · dL/dZ2
        dL/dZ1 = dL/dH * relu_grad(Z1)
        dL/dW1 = dL/dZ1 · X^T

    Args:
        y_hat: Predictions (n, D_o).
        y:     Targets    (n, D_o).
        params: Weight dict.
        cache:  Values from forward pass.

    Returns:
        grads: Dict mapping param names to their gradients.
    """
    n = y.shape[0]
    X, Z1, H = cache["X"], cache["Z1"], cache["H"]
    W2 = params["W2"]

    # Output layer gradients
    dZ2 = (2.0 / n) * (y_hat - y).T     # (D_o, n)
    dW2 = dZ2 @ H.T                      # (D_o, D_h)
    db2 = dZ2.sum(axis=1, keepdims=True) # (D_o, 1)

    # Hidden layer gradients
    dH  = W2.T @ dZ2                     # (D_h, n)
    dZ1 = dH * relu_grad(Z1)             # (D_h, n)
    dW1 = dZ1 @ X                        # (D_h, D_i)
    db1 = dZ1.sum(axis=1, keepdims=True) # (D_h, 1)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ── gradient descent step ─────────────────────────────────────────────────────

def update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    lr: float = 0.05,
) -> dict[str, np.ndarray]:
    """Apply one vanilla gradient descent update."""
    return {k: params[k] - lr * grads[k] for k in params}


# ── main demo ─────────────────────────────────────────────────────────────────

def main() -> None:
    params = init_params(D_i, D_h, D_o)

    # Before update
    y_hat_before, cache = forward(X, params)
    loss_before = mse(y_hat_before, y)

    # One backward + update
    grads = backward(y_hat_before, y, params, cache)
    params = update(params, grads, lr=0.05)

    # After update
    y_hat_after, _ = forward(X, params)
    loss_after = mse(y_hat_after, y)

    print(f"Loss before update : {loss_before:.6f}")
    print(f"Loss after  update : {loss_after:.6f}")
    assert loss_after < loss_before, "Loss should decrease after one gradient step"
    print("OK")


if __name__ == "__main__":
    main()
