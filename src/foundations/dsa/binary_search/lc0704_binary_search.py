"""
LC 0704 — Binary Search
Difficulty: Easy
Pattern: Binary Search
Date: 2026-03-09
"""

from __future__ import annotations


def search(nums: list[int], target: int) -> int:
    """Return index of target in sorted nums, or -1 if not found.

    Invariant: target ∈ nums[left..right] if it exists anywhere.
    We maintain this by shrinking the search space while never
    excluding a position that could still hold target.

    Args:
        nums: Sorted array of distinct integers.
        target: Value to search for.

    Returns:
        Index of target in nums, or -1 if absent.

    Time:  O(log n) — search space halves each iteration.
    Space: O(1)     — two pointers, no auxiliary structure.
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2  # avoids int overflow vs (l+r)//2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1   # target must be to the right
        else:
            right = mid - 1  # target must be to the left

    return -1


# ── tests ─────────────────────────────────────────────────────────────────────

def test_target_present_middle() -> None:
    assert search([-1, 0, 3, 5, 9, 12], 9) == 4


def test_target_absent() -> None:
    assert search([-1, 0, 3, 5, 9, 12], 2) == -1


def test_single_element_found() -> None:
    assert search([5], 5) == 0


def test_single_element_not_found() -> None:
    assert search([5], 3) == -1


def test_target_at_left_boundary() -> None:
    assert search([1, 3, 5, 7], 1) == 0


def test_target_at_right_boundary() -> None:
    assert search([1, 3, 5, 7], 7) == 3


if __name__ == "__main__":
    tests = [
        test_target_present_middle,
        test_target_absent,
        test_single_element_found,
        test_single_element_not_found,
        test_target_at_left_boundary,
        test_target_at_right_boundary,
    ]
    for t in tests:
        t()
    print("OK")
