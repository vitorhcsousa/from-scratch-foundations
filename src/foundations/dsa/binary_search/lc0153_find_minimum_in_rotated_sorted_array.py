"""
LC 0153 — Find Minimum in Rotated Sorted Array
Difficulty: Medium
Pattern: Binary Search — Rotated (pivot-finding variant)
Date: 2026-03-12
"""

from __future__ import annotations


def find_min(nums: list[int]) -> int:
    """Find the minimum value in a rotated sorted array.

    Key insight: the minimum is always at the pivot — the point where
    the array wraps around. Compare nums[mid] with nums[right]:
      - nums[mid] > nums[right]: pivot (min) is in right half
      - nums[mid] <= nums[right]: mid could be the min; search left

    Invariant: minimum ∈ nums[left..right] at every step.

    Note: 'right = mid' (not mid - 1) because mid is a candidate.

    Args:
        nums: Rotated sorted array of distinct integers.

    Returns:
        Minimum value in nums.

    Time:  O(log n).
    Space: O(1).
    """
    left, right = 0, len(nums) - 1

    while left < right:                      # note: <, not <=
        mid = left + (right - left) // 2

        if nums[mid] > nums[right]:
            left = mid + 1                   # pivot (min) is right of mid
        else:
            right = mid                      # mid could be min; keep it

    return nums[left]


# ── tests ─────────────────────────────────────────────────────────────────────

def test_standard_rotation() -> None:
    assert find_min([3, 4, 5, 1, 2]) == 1


def test_rotation_at_start() -> None:
    # No rotation — already sorted
    assert find_min([1, 2, 3, 4, 5]) == 1


def test_rotation_at_end() -> None:
    assert find_min([2, 3, 4, 5, 1]) == 1


def test_single_element() -> None:
    assert find_min([1]) == 1


def test_two_elements_rotated() -> None:
    assert find_min([2, 1]) == 1


def test_larger_rotation() -> None:
    assert find_min([4, 5, 6, 7, 0, 1, 2]) == 0


if __name__ == "__main__":
    tests = [
        test_standard_rotation,
        test_rotation_at_start,
        test_rotation_at_end,
        test_single_element,
        test_two_elements_rotated,
        test_larger_rotation,
    ]
    for t in tests:
        t()
    print("OK")
