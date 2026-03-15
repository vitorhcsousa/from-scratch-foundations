"""
LC 0033 — Search in Rotated Sorted Array
Difficulty: Medium
Pattern: Binary Search — Rotated
Date: 2026-03-11
"""

from __future__ import annotations


def search(nums: list[int], target: int) -> int:
    """Search for target in a rotated sorted array.

    Key insight: for any mid, one of [left, mid] or [mid, right]
    is always fully sorted. Determine which, then check if target
    lies within it. No need to find the pivot explicitly.

    Invariant: target ∈ nums[left..right] if it exists anywhere.

    Args:
        nums: Integer array, originally sorted ascending, then
              rotated at some unknown pivot. All values distinct.
        target: Value to find.

    Returns:
        Index of target, or -1 if not found.

    Time:  O(log n) — each step eliminates one half.
    Space: O(1).
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Determine which half is sorted
        if nums[left] <= nums[mid]:           # left half [left, mid] is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1              # target in sorted left half
            else:
                left = mid + 1               # target must be in right half
        else:                                # right half [mid, right] is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1               # target in sorted right half
            else:
                right = mid - 1              # target must be in left half

    return -1


# ── tests ─────────────────────────────────────────────────────────────────────

def test_target_in_left_sorted_half() -> None:
    assert search([4, 5, 6, 7, 0, 1, 2], 5) == 1


def test_target_in_right_sorted_half() -> None:
    assert search([4, 5, 6, 7, 0, 1, 2], 1) == 5


def test_target_not_found() -> None:
    assert search([4, 5, 6, 7, 0, 1, 2], 3) == -1


def test_single_element_found() -> None:
    assert search([1], 1) == 0


def test_single_element_not_found() -> None:
    assert search([1], 0) == -1


def test_not_rotated() -> None:
    assert search([1, 2, 3, 4, 5], 3) == 2


def test_target_at_pivot() -> None:
    # pivot (minimum) is at index 4; target = 0
    assert search([4, 5, 6, 7, 0, 1, 2], 0) == 4


if __name__ == "__main__":
    tests = [
        test_target_in_left_sorted_half,
        test_target_in_right_sorted_half,
        test_target_not_found,
        test_single_element_found,
        test_single_element_not_found,
        test_not_rotated,
        test_target_at_pivot,
    ]
    for t in tests:
        t()
    print("OK")
