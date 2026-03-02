"""
LC 0209 — Minimum Size Subarray Sum
=====================================
https://leetcode.com/problems/minimum-size-subarray-sum/

Pattern:  Sliding Window — Dynamic (Min-Length), Variant B
Invariant: window_sum >= target (window is valid)
Shrink:   while window_sum >= target → update answer, remove arr[left], left++
Update:   INSIDE shrink loop — every valid smaller window is a candidate

Time:  O(n)  — each element added/removed at most once
Space: O(1)  — only sum and pointers

Prereq: All elements are positive → sum is monotonically increasing with window size.
        Sliding window would NOT work with negative numbers.
"""


def min_sub_array_len(target: int, nums: list[int]) -> int:
    """Find the minimal length subarray with sum >= target."""
    left = 0
    window_sum = 0
    best = float("inf")

    for right in range(len(nums)):
        # EXPAND: add nums[right] to window
        window_sum += nums[right]

        # SHRINK while valid → update inside loop (min-length pattern)
        while window_sum >= target:
            best = min(best, right - left + 1)
            window_sum -= nums[left]
            left += 1

    return best if best != float("inf") else 0


# ── Quick validation ────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        (7, [2, 3, 1, 2, 4, 3], 2),
        (4, [1, 4, 4], 1),
        (11, [1, 1, 1, 1, 1, 1, 1, 1], 0),
        (15, [1, 2, 3, 4, 5], 5),
        (6, [10, 2, 3], 1),
    ]
    for target, nums, expected in cases:
        result = min_sub_array_len(target, nums)
        status = "✓" if result == expected else "✗"
        print(f"  {status} target={target}, nums={nums} → {result} (expected {expected})")
    print("OK")
