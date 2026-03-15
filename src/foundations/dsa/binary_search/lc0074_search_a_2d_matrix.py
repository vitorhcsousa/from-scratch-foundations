"""
LC 0074 — Search a 2D Matrix
Difficulty: Medium
Pattern: Binary Search — virtual flattened sorted array
Date: 2026-03-13
"""

from __future__ import annotations


def search_matrix(matrix: list[list[int]], target: int) -> bool:
    """Search for target in an m×n matrix where:
      - each row is sorted ascending,
      - first integer of each row > last integer of previous row.

    The two properties together mean the matrix can be treated as a
    single sorted array of m*n elements. Binary search on the virtual
    flat index, then map back to (row, col) via divmod.

    Virtual flat index i → row = i // n, col = i % n.

    Args:
        matrix: m×n integer matrix satisfying the sorted properties.
        target: Value to search for.

    Returns:
        True if target exists in matrix, False otherwise.

    Time:  O(log(m*n)) = O(log m + log n).
    Space: O(1).
    """
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1

    while left <= right:
        mid = left + (right - left) // 2
        row, col = divmod(mid, n)          # map flat index → 2D coords
        value = matrix[row][col]

        if value == target:
            return True
        elif value < target:
            left = mid + 1
        else:
            right = mid - 1

    return False


# ── tests ─────────────────────────────────────────────────────────────────────

def test_target_found_middle() -> None:
    matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    assert search_matrix(matrix, 3) is True


def test_target_not_found() -> None:
    matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    assert search_matrix(matrix, 13) is False


def test_single_element_found() -> None:
    assert search_matrix([[5]], 5) is True


def test_single_element_not_found() -> None:
    assert search_matrix([[5]], 3) is False


def test_target_at_last_cell() -> None:
    matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    assert search_matrix(matrix, 60) is True


def test_target_at_first_cell() -> None:
    matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    assert search_matrix(matrix, 1) is True


if __name__ == "__main__":
    tests = [
        test_target_found_middle,
        test_target_not_found,
        test_single_element_found,
        test_single_element_not_found,
        test_target_at_last_cell,
        test_target_at_first_cell,
    ]
    for t in tests:
        t()
    print("OK")
