"""
Prince Ch 2 — Supervised Learning Mini Experiment
===================================================
End-to-end supervised learning pipeline implementing Prince's 4-step recipe:
  1. Choose model  →  shallow neural network (1 hidden layer)
  2. Choose loss   →  MSE (squared error, assumes Gaussian noise)
  3. Fit model     →  mini-batch gradient descent
  4. Evaluate      →  train / validation / test loss

Demonstrates:
  - Train/val/test splits (information flow rule)
  - Overfitting vs underfitting (model capacity via hidden_dim)
  - Learning curves (loss vs epoch)
  - ERM: minimising empirical risk as proxy for true risk
  - Bias-variance visual: compare low-capacity vs high-capacity models

Dataset: 1D sinusoidal regression  y = sin(2πx) + ε,  ε ~ N(0, σ²)
         Prince uses 1D regression to visualise everything clearly.

Usage:
    python ch02_mini_experiment.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ═══════════════════════════════════════════════════════════════
# Step 0: Make Dataset
# ═══════════════════════════════════════════════════════════════

def make_dataset(
    n_total: int = 200,
    noise_std: float = 0.2,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate 1D sinusoidal regression data with train/val/test splits.

    True function: f(x) = sin(2πx)
    Observed: y = f(x) + ε,  ε ~ N(0, noise_std²)

    Split ratio: 60% train / 20% val / 20% test
    (Prince Ch2: "data that influences any decision cannot give unbiased estimate")

    Returns dict with x_train, y_train, x_val, y_val, x_test, y_test.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0.0, 1.0, size=n_total)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + rng.normal(0.0, noise_std, size=n_total)

    # Shuffle and split 60/20/20
    indices = rng.permutation(n_total)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "x_train": x[train_idx, np.newaxis],  # (n_train, 1)
        "y_train": y[train_idx, np.newaxis],
        "x_val": x[val_idx, np.newaxis],
        "y_val": y[val_idx, np.newaxis],
        "x_test": x[test_idx, np.newaxis],
        "y_test": y[test_idx, np.newaxis],
    }


# ═══════════════════════════════════════════════════════════════
# Step 1: Choose Model — Shallow Neural Network
# ═══════════════════════════════════════════════════════════════

class ShallowNet:
    """Single hidden layer neural network: y = W₂ · ReLU(W₁x + b₁) + b₂.

    Architecture:
        Input (1) → Hidden (hidden_dim, ReLU) → Output (1)

    This is Prince's "shallow neural network" from Ch2/Ch3.
    Capacity controlled by hidden_dim:
        - Small hidden_dim → low capacity → underfitting risk
        - Large hidden_dim → high capacity → overfitting risk
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 20, seed: int = 0):
        rng = np.random.default_rng(seed)

        # He initialisation (good default for ReLU)
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, 1)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

        # Cache for backprop
        self._cache: dict[str, np.ndarray] = {}

    @property
    def num_params(self) -> int:
        """Total trainable parameters (Prince's P)."""
        return (
            self.W1.size + self.b1.size + self.W2.size + self.b2.size
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x → ŷ.  Shape: (batch, 1) → (batch, 1)."""
        z1 = x @ self.W1 + self.b1  # (batch, hidden)
        a1 = np.maximum(z1, 0.0)  # ReLU
        y_hat = a1 @ self.W2 + self.b2  # (batch, 1)

        # Cache for backward
        self._cache = {"x": x, "z1": z1, "a1": a1}
        return y_hat

    def backward(self, grad_y: np.ndarray) -> dict[str, np.ndarray]:
        """Backward pass: ∂L/∂ŷ → gradients for all params.

        Backpropagation = chain rule applied recursively.
        Cost ≈ 2× forward pass (Prince Ch2).
        """
        x = self._cache["x"]
        z1 = self._cache["z1"]
        a1 = self._cache["a1"]
        batch = x.shape[0]

        # ∂L/∂W₂ and ∂L/∂b₂
        grad_W2 = (a1.T @ grad_y) / batch
        grad_b2 = grad_y.mean(axis=0)

        # ∂L/∂a₁
        grad_a1 = grad_y @ self.W2.T

        # ReLU gradient: pass through where z1 > 0
        grad_z1 = grad_a1 * (z1 > 0).astype(float)

        # ∂L/∂W₁ and ∂L/∂b₁
        grad_W1 = (x.T @ grad_z1) / batch
        grad_b1 = grad_z1.mean(axis=0)

        return {
            "W1": grad_W1,
            "b1": grad_b1,
            "W2": grad_W2,
            "b2": grad_b2,
        }

    def update(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """Gradient descent step: φ ← φ − α∇L (Prince Ch2)."""
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]


# ═══════════════════════════════════════════════════════════════
# Step 2: Choose Loss — Mean Squared Error
# ═══════════════════════════════════════════════════════════════

def mse_loss(y_hat: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute MSE loss and its gradient.

    L = (1/I) Σᵢ (ŷᵢ - yᵢ)²

    Equivalent to negative log-likelihood under Gaussian noise assumption
    (Prince Ch2: "right loss = NLL under assumed output distribution").

    Returns:
        loss: scalar MSE
        grad: ∂L/∂ŷ of shape (batch, 1)
    """
    residual = y_hat - y
    loss = float(np.mean(residual**2))
    grad = 2.0 * residual / y.shape[0]
    return loss, grad


# ═══════════════════════════════════════════════════════════════
# Step 3: Fit Model — Mini-Batch Gradient Descent
# ═══════════════════════════════════════════════════════════════

def train(
    model: ShallowNet,
    data: dict[str, np.ndarray],
    lr: float = 0.1,
    epochs: int = 500,
    batch_size: int = 32,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train using mini-batch gradient descent.

    Prince Ch2 recipe:
        φ* = argmin_φ (1/I) Σᵢ ℓ(f(xᵢ; φ), yᵢ)

    Mini-batch SGD: unbiased estimator of the full gradient.
    E_B[∇ℓ_batch] = ∇L  (Prince Ch2).

    Returns history dict with train_loss and val_loss per epoch.
    """
    rng = np.random.default_rng(seed)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    n_train = x_train.shape[0]

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Shuffle training data each epoch
        perm = rng.permutation(n_train)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        # Mini-batch loop
        for start in range(0, n_train, batch_size):
            x_batch = x_shuffled[start : start + batch_size]
            y_batch = y_shuffled[start : start + batch_size]

            # Forward
            y_hat = model.forward(x_batch)
            _, grad = mse_loss(y_hat, y_batch)

            # Backward + update
            grads = model.backward(grad)
            model.update(grads, lr)

        # Epoch-level evaluation (on full splits)
        train_loss, _ = mse_loss(model.forward(x_train), y_train)
        val_loss, _ = mse_loss(model.forward(x_val), y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch + 1:4d} │ train_loss: {train_loss:.6f} │ val_loss: {val_loss:.6f}")

    return history


# ═══════════════════════════════════════════════════════════════
# Step 4: Evaluate
# ═══════════════════════════════════════════════════════════════

def evaluate(
    model: ShallowNet,
    data: dict[str, np.ndarray],
) -> dict[str, float]:
    """Evaluate model on all splits.

    Prince Ch2: test set must be touched only ONCE for unbiased estimate.
    The generalization gap = test_loss - train_loss.
    """
    results = {}
    for split in ["train", "val", "test"]:
        x = data[f"x_{split}"]
        y = data[f"y_{split}"]
        y_hat = model.forward(x)
        loss, _ = mse_loss(y_hat, y)
        results[f"{split}_loss"] = loss
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main(seed: int = 0) -> None:
    """Run the full Prince Ch2 pipeline."""

    print("=" * 64)
    print("  Prince Ch 2 — Supervised Learning Mini Experiment")
    print("=" * 64)
    print()

    # --- Dataset ---
    data = make_dataset(n_total=200, noise_std=0.2, seed=seed)
    print(f"  Dataset: y = sin(2πx) + ε,  ε ~ N(0, 0.04)")
    print(f"  Splits:  train={data['x_train'].shape[0]}, "
          f"val={data['x_val'].shape[0]}, "
          f"test={data['x_test'].shape[0]}")
    print()

    # --- Experiment: compare two model capacities ---
    configs = [
        {"name": "Low capacity (H=4)", "hidden_dim": 4},
        {"name": "Good capacity (H=20)", "hidden_dim": 20},
    ]

    for cfg in configs:
        print(f"  ── {cfg['name']} {'─' * (50 - len(cfg['name']))}")
        model = ShallowNet(
            input_dim=1, hidden_dim=cfg["hidden_dim"], seed=seed
        )
        print(f"  Parameters: {model.num_params}  (P = {cfg['hidden_dim']}×1 + {cfg['hidden_dim']} + 1×{cfg['hidden_dim']} + 1)")

        # Train
        print(f"  Training (500 epochs, lr=0.1, batch=32):")
        history = train(model, data, lr=0.1, epochs=500, batch_size=32, seed=seed)

        # Evaluate
        results = evaluate(model, data)
        print()
        print(f"  Final losses:")
        print(f"    Train: {results['train_loss']:.6f}")
        print(f"    Val:   {results['val_loss']:.6f}")
        print(f"    Test:  {results['test_loss']:.6f}")
        gen_gap = results["test_loss"] - results["train_loss"]
        print(f"    Generalization gap (test - train): {gen_gap:+.6f}")
        print()

        # Diagnosis (Prince Ch2 diagnostic table)
        train_l = results["train_loss"]
        val_l = results["val_loss"]
        if train_l > 0.1 and val_l > 0.1:
            diagnosis = "UNDERFIT (both high → increase capacity)"
        elif train_l < 0.05 and val_l > train_l * 2:
            diagnosis = "OVERFIT (train low, val high → regularise or more data)"
        else:
            diagnosis = "GOOD FIT (both reasonably low)"
        print(f"  Diagnosis: {diagnosis}")
        print()

    # --- Irreducible error (noise floor) ---
    noise_var = 0.2**2
    print(f"  ── Irreducible error {'─' * 40}")
    print(f"  σ² = {noise_var:.4f}  (noise variance in data)")
    print(f"  No model can achieve MSE < σ² = {noise_var:.4f}")
    print(f"  (Prince Ch2: bias-variance decomposition → irreducible term)")
    print()

    print("=" * 64)
    print("OK")
    print("=" * 64)


if __name__ == "__main__":
    main(seed=0)
