"""
LC 1004 — Max Consecutive Ones III
====================================
https://leetcode.com/problems/max-consecutive-ones-iii/

Pattern:  Sliding Window — Dynamic (Max-Length), Variant C
Invariant: zeros_in_window <= k (at most k zeros allowed)
Shrink:   while zeros > k → if nums[left] == 0 then zeros--, left++
Update:   after shrink (outside loop) — we want the LARGEST valid window

Reframe: "flip at most k zeros" = "longest subarray with at most k zeros"

Time:  O(n)  — each element visited at most twice
Space: O(1)  — only zero count and pointers
"""


def longest_ones(nums: list[int], k: int) -> int:
    """Find the longest subarray of 1s achievable by flipping at most k zeros."""
    left = 0
    zeros = 0
    best = 0

    for right in range(len(nums)):
        # EXPAND
        if nums[right] == 0:
            zeros += 1

        # SHRINK while invariant broken
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1

        # UPDATE: max-length → after restoring validity
        best = max(best, right - left + 1)

    return best


# ── Quick validation ────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        ([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2, 6),
        ([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], 3, 10),
        ([1, 1, 1, 1], 0, 4),
        ([0, 0, 0, 0], 0, 0),
        ([0, 0, 0, 0], 4, 4),
    ]
    for nums, k, expected in cases:
        result = longest_ones(nums, k)
        status = "✓" if result == expected else "✗"
        print(f"  {status} k={k}, nums={nums} → {result} (expected {expected})")
    print("OK")
