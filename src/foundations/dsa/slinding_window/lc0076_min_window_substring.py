"""
LC 0076 — Minimum Window Substring
====================================
https://leetcode.com/problems/minimum-window-substring/

Pattern:  Sliding Window — Dynamic (Min-Length), Variant B
Invariant: missing == 0 (window contains all chars of t with required frequencies)
Shrink:   while missing == 0 → update answer, remove s[left], left++
Update:   INSIDE shrink loop — every valid smaller window is a candidate

Key trick: track `missing` (distinct chars still unsatisfied) instead of comparing
           full counters. Transition: need[c] goes 1→0 → missing--, 0→1 → missing++.

Time:  O(|s| + |t|)  — each char added/removed at most once, plus building counter
Space: O(|t|)         — counter for t
"""

from collections import Counter


def min_window(s: str, t: str) -> str:
    """Find the minimum window in s that contains all characters of t."""
    if not t or not s:
        return ""

    need = Counter(t)
    missing = len(need)  # distinct chars still needed
    left = 0
    best = (float("inf"), 0, 0)  # (length, left, right)

    for right in range(len(s)):
        # EXPAND: process s[right]
        char = s[right]
        if char in need:
            need[char] -= 1
            if need[char] == 0:
                missing -= 1

        # SHRINK while valid → update inside loop (min-length pattern)
        while missing == 0:
            window_len = right - left + 1
            if window_len < best[0]:
                best = (window_len, left, right)

            # Remove s[left] from window
            left_char = s[left]
            if left_char in need:
                need[left_char] += 1
                if need[left_char] > 0:
                    missing += 1
            left += 1

    return "" if best[0] == float("inf") else s[best[1] : best[2] + 1]


# ── Quick validation ────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        ("ADOBECODEBANC", "ABC", "BANC"),
        ("a", "a", "a"),
        ("a", "aa", ""),
        ("aa", "aa", "aa"),
        ("cabwefgewcwaefgcf", "cae", "cwae"),
    ]
    for s, t, expected in cases:
        result = min_window(s, t)
        status = "✓" if result == expected else "✗"
        print(f"  {status} s={s!r}, t={t!r} → {result!r} (expected {expected!r})")
    print("OK")
