"""
Danka Ch 1 — Linear Algebra Sanity Demo
=========================================
Verifies core linear algebra identities from Danka's Mathematics of ML, Ch 1.

Demos:
  1. Dot product and norms (L1, L2, L∞) + cosine similarity
  2. Matrix–vector multiplication (column interpretation)
  3. Distributive identity: A(x + y) = Ax + Ay (linearity proof)
  4. Bonus: inner product generates norm, norm generates metric

Each demo prints results and runs identity checks with np.allclose.
All checks must pass → prints 'OK' at the end.

Usage:
    python ch01_sanity_demo.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

PASS = "✓"
FAIL = "✗"


# ── Demo 1: Dot Product and Norms ──────────────────────────────
def demo_dot_and_norm() -> list[bool]:
    """Demonstrate dot product, norms, and cosine similarity."""
    checks: list[bool] = []

    x = np.array([3.0, 4.0])
    y = np.array([1.0, 2.0])

    print("=" * 60)
    print("DEMO 1: Dot Product and Norms")
    print("=" * 60)
    print(f"  x = {x}")
    print(f"  y = {y}")
    print()

    # --- Dot product ---
    dot_manual = np.sum(x * y)  # component-wise multiply then sum
    dot_numpy = np.dot(x, y)
    print(f"  Dot product (manual):  Σ xᵢyᵢ = {dot_manual}")
    print(f"  Dot product (np.dot):  x·y    = {dot_numpy}")
    ok = np.allclose(dot_manual, dot_numpy)
    checks.append(ok)
    print(f"  {PASS if ok else FAIL} Manual == np.dot")
    print()

    # --- Norms of x = (3, 4) ---
    l1 = np.linalg.norm(x, ord=1)  # |3| + |4| = 7
    l2 = np.linalg.norm(x, ord=2)  # √(9 + 16) = 5
    linf = np.linalg.norm(x, ord=np.inf)  # max(3, 4) = 4
    print(f"  L1 norm (Manhattan):   ||x||₁  = {l1}  (expected 7.0)")
    print(f"  L2 norm (Euclidean):   ||x||₂  = {l2}  (expected 5.0)")
    print(f"  L∞ norm (Chebyshev):   ||x||∞  = {linf}  (expected 4.0)")

    ok_l1 = np.allclose(l1, 7.0)
    ok_l2 = np.allclose(l2, 5.0)
    ok_linf = np.allclose(linf, 4.0)
    checks.extend([ok_l1, ok_l2, ok_linf])
    print(f"  {PASS if ok_l1 else FAIL} L1 == 7")
    print(f"  {PASS if ok_l2 else FAIL} L2 == 5  (Pythagoras: √(3² + 4²))")
    print(f"  {PASS if ok_linf else FAIL} L∞ == 4")
    print()

    # --- Homogeneity: ||cx|| = |c| ||x|| ---
    c = -2.5
    homogeneity_lhs = np.linalg.norm(c * x)
    homogeneity_rhs = abs(c) * np.linalg.norm(x)
    ok = np.allclose(homogeneity_lhs, homogeneity_rhs)
    checks.append(ok)
    print(f"  Homogeneity: ||{c}·x|| = {homogeneity_lhs:.4f}")
    print(f"               |{c}|·||x|| = {homogeneity_rhs:.4f}")
    print(f"  {PASS if ok else FAIL} ||cx|| == |c|·||x||")
    print()

    # --- Cosine similarity ---
    cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    angle_rad = np.arccos(np.clip(cos_sim, -1, 1))
    angle_deg = np.degrees(angle_rad)
    print(f"  Cosine similarity:  cos(θ) = x·y / (||x||·||y||) = {cos_sim:.4f}")
    print(f"  Angle between x, y:  θ = {angle_deg:.2f}°")

    # Orthogonal vectors should have cos(θ) = 0
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    cos_ortho = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ok = np.allclose(cos_ortho, 0.0)
    checks.append(ok)
    print(f"  Orthogonal check: cos(e₁, e₂) = {cos_ortho}  (expected 0)")
    print(f"  {PASS if ok else FAIL} Orthogonal vectors → cos(θ) = 0")
    print()

    return checks


# ── Demo 2: Matrix–Vector Multiplication ───────────────────────
def demo_matrix_vector_mul() -> list[bool]:
    """Demonstrate matrix-vector product and column interpretation."""
    checks: list[bool] = []

    A = np.array([[2.0, -1.0], [1.0, 3.0]])
    x = np.array([3.0, 2.0])

    print("=" * 60)
    print("DEMO 2: Matrix–Vector Multiplication")
    print("=" * 60)
    print(f"  A = {A[0]}")
    print(f"      {A[1]}")
    print(f"  x = {x}")
    print()

    # --- Standard multiplication ---
    result = A @ x
    print(f"  Ax = A @ x = {result}")
    print()

    # --- Row interpretation: row i dotted with x ---
    row_result = np.array([np.dot(A[0], x), np.dot(A[1], x)])
    ok = np.allclose(result, row_result)
    checks.append(ok)
    print(f"  Row interpretation:")
    print(f"    row₀ · x = {A[0]} · {x} = {np.dot(A[0], x)}")
    print(f"    row₁ · x = {A[1]} · {x} = {np.dot(A[1], x)}")
    print(f"  {PASS if ok else FAIL} Ax == [row_i · x for each row]")
    print()

    # --- Column interpretation: weighted sum of columns ---
    col_result = x[0] * A[:, 0] + x[1] * A[:, 1]
    ok = np.allclose(result, col_result)
    checks.append(ok)
    print(f"  Column interpretation (Danka's key insight):")
    print(f"    Ax = x₀ · col₀ + x₁ · col₁")
    print(f"       = {x[0]} · {A[:, 0]} + {x[1]} · {A[:, 1]}")
    print(f"       = {x[0] * A[:, 0]} + {x[1] * A[:, 1]}")
    print(f"       = {col_result}")
    print(f"  {PASS if ok else FAIL} Ax == Σ xⱼ · colⱼ(A)")
    print()

    # --- Columns = images of basis vectors ---
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    print(f"  Columns as images of basis vectors:")
    print(f"    A @ e₁ = {A @ e1}  (= column 0 of A: {A[:, 0]})")
    print(f"    A @ e₂ = {A @ e2}  (= column 1 of A: {A[:, 1]})")
    ok_e1 = np.allclose(A @ e1, A[:, 0])
    ok_e2 = np.allclose(A @ e2, A[:, 1])
    checks.extend([ok_e1, ok_e2])
    print(f"  {PASS if ok_e1 else FAIL} A @ e₁ == col₀(A)")
    print(f"  {PASS if ok_e2 else FAIL} A @ e₂ == col₁(A)")
    print()

    # --- Shape semantics ---
    print(f"  Shape: A is {A.shape} (2 outputs, 2 inputs)")
    print(f"         x is {x.shape} (2 inputs)")
    print(f"         Ax is {result.shape} (2 outputs)")
    print(f"  → A maps ℝ² → ℝ²")
    print()

    return checks


# ── Demo 3: Distributive Identity (Linearity) ─────────────────
def demo_distributive_identity() -> list[bool]:
    """Verify the linearity axiom: A(ax + by) = aAx + bAy."""
    checks: list[bool] = []

    rng = np.random.default_rng(42)
    m, n = 3, 4  # A maps ℝ⁴ → ℝ³
    A = rng.standard_normal((m, n))
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    a, b = 2.5, -1.3

    print("=" * 60)
    print("DEMO 3: Linearity — A(ax + by) = aAx + bAy")
    print("=" * 60)
    print(f"  A shape: {A.shape} (maps ℝ⁴ → ℝ³)")
    print(f"  x = {x}")
    print(f"  y = {y}")
    print(f"  a = {a}, b = {b}")
    print()

    # --- Additivity: A(x + y) = Ax + Ay ---
    lhs_add = A @ (x + y)
    rhs_add = A @ x + A @ y
    ok = np.allclose(lhs_add, rhs_add)
    checks.append(ok)
    print(f"  Additivity:")
    print(f"    A(x + y) = {lhs_add}")
    print(f"    Ax + Ay  = {rhs_add}")
    print(f"  {PASS if ok else FAIL} A(x + y) == Ax + Ay")
    print()

    # --- Homogeneity: A(ax) = a(Ax) ---
    lhs_hom = A @ (a * x)
    rhs_hom = a * (A @ x)
    ok = np.allclose(lhs_hom, rhs_hom)
    checks.append(ok)
    print(f"  Homogeneity:")
    print(f"    A(ax) = {lhs_hom}")
    print(f"    a(Ax) = {rhs_hom}")
    print(f"  {PASS if ok else FAIL} A(ax) == a(Ax)")
    print()

    # --- Full linearity: A(ax + by) = aAx + bAy ---
    lhs_full = A @ (a * x + b * y)
    rhs_full = a * (A @ x) + b * (A @ y)
    ok = np.allclose(lhs_full, rhs_full)
    checks.append(ok)
    print(f"  Full linearity:")
    print(f"    A(ax + by)   = {lhs_full}")
    print(f"    aAx + bAy    = {rhs_full}")
    print(f"  {PASS if ok else FAIL} A(ax + by) == aAx + bAy")
    print()

    # --- Origin preservation: A(0) = 0 ---
    zero_in = np.zeros(n)
    zero_out = A @ zero_in
    ok = np.allclose(zero_out, np.zeros(m))
    checks.append(ok)
    print(f"  Origin preservation:")
    print(f"    A(0) = {zero_out}")
    print(f"  {PASS if ok else FAIL} A(0) == 0  (linear maps fix the origin)")
    print()

    # --- Composition: (AB)x = A(Bx) ---
    B = rng.standard_normal((n, 5))  # B maps ℝ⁵ → ℝ⁴
    z = rng.standard_normal(5)
    lhs_comp = (A @ B) @ z
    rhs_comp = A @ (B @ z)
    ok = np.allclose(lhs_comp, rhs_comp)
    checks.append(ok)
    print(f"  Composition (matrix multiplication = function composition):")
    print(f"    (AB)z = {lhs_comp}")
    print(f"    A(Bz) = {rhs_comp}")
    print(f"  {PASS if ok else FAIL} (AB)z == A(Bz)")
    print()

    return checks


# ── Demo 4: Inner Product → Norm → Metric Hierarchy ───────────
def demo_hierarchy() -> list[bool]:
    """Verify the hierarchy: inner product generates norm generates metric."""
    checks: list[bool] = []

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, -1.0, 2.0])

    print("=" * 60)
    print("DEMO 4: Inner Product → Norm → Metric Hierarchy")
    print("=" * 60)
    print(f"  x = {x}")
    print(f"  y = {y}")
    print()

    # --- Inner product generates L2 norm: ||x|| = √⟨x, x⟩ ---
    ip_xx = np.dot(x, x)
    norm_from_ip = np.sqrt(ip_xx)
    norm_direct = np.linalg.norm(x)
    ok = np.allclose(norm_from_ip, norm_direct)
    checks.append(ok)
    print(f"  Inner product → norm:")
    print(f"    ⟨x, x⟩      = {ip_xx}")
    print(f"    √⟨x, x⟩     = {norm_from_ip:.4f}")
    print(f"    ||x||₂       = {norm_direct:.4f}")
    print(f"  {PASS if ok else FAIL} ||x||₂ == √⟨x, x⟩")
    print()

    # --- Norm generates metric: d(x, y) = ||x - y|| ---
    diff = x - y
    metric = np.linalg.norm(diff)
    print(f"  Norm → metric:")
    print(f"    x - y        = {diff}")
    print(f"    d(x, y)      = ||x - y|| = {metric:.4f}")
    print()

    # --- Metric axioms ---
    # Symmetry: d(x, y) = d(y, x)
    ok = np.allclose(np.linalg.norm(x - y), np.linalg.norm(y - x))
    checks.append(ok)
    print(f"  Metric axiom — symmetry:")
    print(f"    d(x, y) = {np.linalg.norm(x - y):.4f}")
    print(f"    d(y, x) = {np.linalg.norm(y - x):.4f}")
    print(f"  {PASS if ok else FAIL} d(x, y) == d(y, x)")

    # Triangle inequality: d(x, z) ≤ d(x, y) + d(y, z)
    z = np.array([0.0, 0.0, 0.0])
    d_xz = np.linalg.norm(x - z)
    d_xy = np.linalg.norm(x - y)
    d_yz = np.linalg.norm(y - z)
    ok = d_xz <= d_xy + d_yz + 1e-10
    checks.append(ok)
    print(f"  Metric axiom — triangle inequality:")
    print(f"    d(x, z)          = {d_xz:.4f}")
    print(f"    d(x, y) + d(y, z) = {d_xy:.4f} + {d_yz:.4f} = {d_xy + d_yz:.4f}")
    print(f"  {PASS if ok else FAIL} d(x, z) ≤ d(x, y) + d(y, z)")
    print()

    # --- Polarization identity: ⟨x, y⟩ = ½(||x+y||² - ||x||² - ||y||²) ---
    ip_direct = np.dot(x, y)
    ip_polarization = 0.5 * (
        np.linalg.norm(x + y) ** 2
        - np.linalg.norm(x) ** 2
        - np.linalg.norm(y) ** 2
    )
    ok = np.allclose(ip_direct, ip_polarization)
    checks.append(ok)
    print(f"  Polarization identity (recover inner product from norm):")
    print(f"    ⟨x, y⟩ (direct)       = {ip_direct:.4f}")
    print(f"    ½(||x+y||² - ||x||² - ||y||²) = {ip_polarization:.4f}")
    print(f"  {PASS if ok else FAIL} Polarization identity holds")
    print()

    return checks


# ── Main ───────────────────────────────────────────────────────
def main() -> None:
    """Run all demos and report."""
    all_checks: list[bool] = []

    all_checks.extend(demo_dot_and_norm())
    all_checks.extend(demo_matrix_vector_mul())
    all_checks.extend(demo_distributive_identity())
    all_checks.extend(demo_hierarchy())

    # --- Final report ---
    passed = sum(all_checks)
    total = len(all_checks)
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("OK")
    else:
        failed_indices = [i for i, c in enumerate(all_checks) if not c]
        print(f"FAILED checks: {failed_indices}")
        raise AssertionError(f"{total - passed} checks failed")


if __name__ == "__main__":
    main()
