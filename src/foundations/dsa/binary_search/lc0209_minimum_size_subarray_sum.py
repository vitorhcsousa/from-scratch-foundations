"""
LC 0209 — Minimum Size Subarray Sum
Difficulty: Medium
Pattern: Sliding Window — Dynamic (minimise length)
Date: 2026-03-13
"""

from __future__ import annotations


def min_sub_array_len(target: int, nums: list[int]) -> int:
    """Find length of shortest contiguous subarray with sum >= target.

    Pattern: dynamic sliding window (minimise length).
    Invariant: shrink while window sum >= target; update answer at each
    valid shrink — every valid smaller window is a candidate.

    Why sliding window works: all nums are positive, so expanding
    always increases the sum (monotonic). Shrinking from left always
    decreases it. The constraint has predictable dynamics.

    Args:
        target: Minimum required subarray sum.
        nums:   Array of positive integers.

    Returns:
        Length of shortest subarray with sum >= target, or 0 if none.

    Time:  O(n) — left and right each advance at most n times.
    Space: O(1).
    """
    left = 0
    window_sum = 0
    best = float("inf")

    for right in range(len(nums)):
        # EXPAND
        window_sum += nums[right]

        # SHRINK while valid → update answer inside loop (min-length pattern)
        while window_sum >= target:
            best = min(best, right - left + 1)
            window_sum -= nums[left]
            left += 1

    return 0 if best == float("inf") else int(best)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_example_1() -> None:
    assert min_sub_array_len(7, [2, 3, 1, 2, 4, 3]) == 2  # [4,3]


def test_example_2() -> None:
    assert min_sub_array_len(4, [1, 4, 4]) == 1  # [4]


def test_no_valid_subarray() -> None:
    assert min_sub_array_len(11, [1, 1, 1, 1, 1, 1, 1, 1]) == 0


def test_entire_array_needed() -> None:
    assert min_sub_array_len(15, [1, 2, 3, 4, 5]) == 5


def test_single_element_meets_target() -> None:
    assert min_sub_array_len(3, [3]) == 1


if __name__ == "__main__":
    tests = [
        test_example_1,
        test_example_2,
        test_no_valid_subarray,
        test_entire_array_needed,
        test_single_element_meets_target,
    ]
    for t in tests:
        t()
    print("OK")
