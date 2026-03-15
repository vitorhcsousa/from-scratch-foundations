"""
ch03_linear_systems_demo.py
===========================
From-scratch implementation of Gaussian elimination and LU decomposition
using NumPy only (no scipy.linalg).

Demonstrates:
  - Forward elimination with partial pivoting → PA = LU
  - Forward substitution (solve Ly = b)
  - Back substitution (solve Ux = y)
  - Full solve: Ax = b via LU
  - Solvability detection: unique / infinite / no solution

Architecture:
  - lu_decompose(A)           → P, L, U
  - forward_sub(L, b)         → y
  - back_sub(U, y)            → x
  - solve(A, b)               → x (or raises for inconsistent system)
  - classify_system(A, b)     → "unique" | "infinite" | "none"

Run:
    python ch03_linear_systems_demo.py   → prints all checks + OK
"""

from __future__ import annotations

import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

ATOL = 1e-10  # tolerance for near-zero pivot detection


def _augment(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])


# ── LU decomposition with partial pivoting ────────────────────────────────────

def lu_decompose(
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose A into P, L, U such that PA = LU.

    Uses partial pivoting (swap rows to put largest pivot on top)
    for numerical stability.

    Args:
        A: Square matrix (n, n).

    Returns:
        P: Permutation matrix (n, n).
        L: Unit lower-triangular matrix (n, n).
        U: Upper-triangular matrix (n, n).

    Raises:
        ValueError: If A is singular (pivot is zero after row swaps).
    """
    A = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n)
    P = np.eye(n)

    for col in range(n):
        # Partial pivoting: find row with largest absolute value in this column
        pivot_row = col + int(np.argmax(np.abs(A[col:, col])))
        if abs(A[pivot_row, col]) < ATOL:
            raise ValueError(f"Singular matrix: zero pivot at column {col}")

        # Swap rows in A and P; update L for already-processed columns
        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
            P[[col, pivot_row]] = P[[pivot_row, col]]
            if col > 0:
                L[[col, pivot_row], :col] = L[[pivot_row, col], :col]

        # Eliminate below pivot
        for row in range(col + 1, n):
            multiplier = A[row, col] / A[col, col]
            L[row, col] = multiplier
            A[row, col:] -= multiplier * A[col, col:]

    U = A
    return P, L, U


# ── forward substitution: solve Ly = b ───────────────────────────────────────

def forward_sub(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve lower-triangular system Ly = b (L has ones on diagonal).

    Args:
        L: Unit lower-triangular (n, n).
        b: Right-hand side (n,).

    Returns:
        y: Solution vector (n,).
    """
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


# ── back substitution: solve Ux = y ──────────────────────────────────────────

def back_sub(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve upper-triangular system Ux = y.

    Args:
        U: Upper-triangular (n, n); diagonal entries are pivots.
        y: Right-hand side (n,).

    Returns:
        x: Solution vector (n,).
    """
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


# ── full solve ────────────────────────────────────────────────────────────────

def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b via LU decomposition with partial pivoting.

    Args:
        A: Square coefficient matrix (n, n).
        b: Right-hand side (n,).

    Returns:
        x: Solution vector (n,).

    Raises:
        ValueError: If A is singular.
    """
    P, L, U = lu_decompose(A)
    pb = P @ b.astype(float)       # apply same row permutation to b
    y = forward_sub(L, pb)         # solve Ly = Pb
    x = back_sub(U, y)             # solve Ux = y
    return x


# ── solvability classifier ────────────────────────────────────────────────────

def classify_system(A: np.ndarray, b: np.ndarray) -> str:
    """Classify linear system Ax = b as unique / infinite / none.

    Uses rank comparison between A and augmented [A|b].

    Returns:
        "unique"   — rank(A) == rank([A|b]) == n
        "infinite" — rank(A) == rank([A|b]) < n
        "none"     — rank([A|b]) > rank(A)
    """
    aug = _augment(A, b)
    rank_A   = int(np.linalg.matrix_rank(A,   tol=ATOL))
    rank_aug = int(np.linalg.matrix_rank(aug, tol=ATOL))
    n = A.shape[1]

    if rank_aug > rank_A:
        return "none"
    if rank_A == n:
        return "unique"
    return "infinite"


# ── demo ─────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(42)

    # ── Example 1: unique solution ────────────────────────────────────────────
    A1 = np.array([[2.0, 1.0, -1.0],
                   [-3.0, -1.0, 2.0],
                   [-2.0, 1.0, 2.0]])
    b1 = np.array([8.0, -11.0, -3.0])
    x1_expected = np.array([2.0, 3.0, -1.0])

    x1 = solve(A1, b1)
    residual1 = np.linalg.norm(A1 @ x1 - b1)
    assert residual1 < 1e-9, f"Residual too large: {residual1}"
    assert np.allclose(x1, x1_expected, atol=1e-9)
    print(f"Example 1 — unique solution:  x = {x1}  residual = {residual1:.2e}")

    # ── Example 2: verify PA = LU ─────────────────────────────────────────────
    A2 = rng.standard_normal((5, 5))
    P2, L2, U2 = lu_decompose(A2)
    reconstruction_error = np.linalg.norm(P2 @ A2 - L2 @ U2)
    assert reconstruction_error < 1e-10, f"PA ≠ LU: error = {reconstruction_error}"
    print(f"Example 2 — LU reconstruction error:  {reconstruction_error:.2e}")

    # ── Example 3: solvability classification ─────────────────────────────────
    # Infinite solutions: rank-deficient but consistent
    A3 = np.array([[1.0, 2.0, 3.0],
                   [2.0, 4.0, 6.0]])   # row 2 = 2 × row 1
    b3_inf = np.array([6.0, 12.0])     # consistent
    b3_none = np.array([6.0, 7.0])     # inconsistent

    assert classify_system(A3, b3_inf)  == "infinite"
    assert classify_system(A3, b3_none) == "none"
    assert classify_system(A1, b1)      == "unique"
    print("Example 3 — solvability classification: all correct")

    # ── Example 4: multiple RHS (LU reuse) ───────────────────────────────────
    A4 = rng.standard_normal((4, 4))
    P4, L4, U4 = lu_decompose(A4)
    for _ in range(5):
        b4 = rng.standard_normal(4)
        pb4 = P4 @ b4
        y4 = forward_sub(L4, pb4)
        x4 = back_sub(U4, y4)
        assert np.linalg.norm(A4 @ x4 - b4) < 1e-9
    print("Example 4 — LU reuse for 5 different RHS: all residuals < 1e-9")

    print("\nOK")


if __name__ == "__main__":
    main()
