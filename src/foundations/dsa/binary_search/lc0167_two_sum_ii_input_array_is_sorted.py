"""
LC 0167 — Two Sum II (Input Array Is Sorted)
Difficulty: Medium
Pattern: Two Pointers — Inward Traversal
Date: 2026-03-09
"""

from __future__ import annotations


def two_sum(numbers: list[int], target: int) -> list[int]:
    """Return 1-indexed positions of two numbers that sum to target.

    Key insight: array is sorted → use inward two pointers.
    - If sum < target: left pointer moves right (need larger value)
    - If sum > target: right pointer moves left (need smaller value)
    - If sum == target: found

    The sorted property guarantees we can always make a meaningful
    decision about which pointer to move — this is the "predictable
    dynamics" that makes O(n) possible.

    Args:
        numbers: Sorted (ascending) array of integers.
        target: Target sum.

    Returns:
        1-indexed [left+1, right+1] of the two elements.

    Time:  O(n) — each pointer moves at most n steps.
    Space: O(1) — no auxiliary structure.
    """
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]

        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed output
        elif current_sum < target:
            left += 1   # need a larger value
        else:
            right -= 1  # need a smaller value

    # Problem guarantees exactly one solution; we never reach here.
    raise ValueError("No solution found")  # pragma: no cover


# ── tests ─────────────────────────────────────────────────────────────────────

def test_basic_example() -> None:
    assert two_sum([2, 7, 11, 15], 9) == [1, 2]


def test_adjacent_elements() -> None:
    assert two_sum([2, 3, 4], 6) == [1, 3]


def test_negative_numbers() -> None:
    assert two_sum([-1, 0], -1) == [1, 2]


if __name__ == "__main__":
    tests = [test_basic_example, test_adjacent_elements, test_negative_numbers]
    for t in tests:
        t()
    print("OK")
