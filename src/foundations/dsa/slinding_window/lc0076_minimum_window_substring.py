"""
LC 0076 — Minimum Window Substring
====================================
https://leetcode.com/problems/minimum-window-substring/

Pattern:   Sliding Window (variable-width, shrinkable)
Invariant: Window [left, right] always tracks character frequencies;
           'formed' counts how many of t's unique chars are satisfied.
Shrink:    When formed == required, shrink left while maintaining validity.
Time:  O(|s| + |t|)
Space: O(|s| + |t|)  — for Counter dicts + output string
"""

from collections import Counter


def min_window(s: str, t: str) -> str:
    """Find minimum window in s that contains all characters of t."""
    if not s or not t:
        return ""

    need = Counter(t)             # chars we need and their counts
    required = len(need)          # number of unique chars to satisfy
    formed = 0                    # how many unique chars currently satisfied
    window: dict[str, int] = {}   # current window char counts

    best_len = float("inf")
    best_left = 0

    left = 0
    for right, char in enumerate(s):
        # Expand: add s[right] to window
        window[char] = window.get(char, 0) + 1

        # Check if this char is now satisfied
        if char in need and window[char] == need[char]:
            formed += 1

        # Shrink: while window is valid, try to minimise
        while formed == required:
            # Update best
            window_len = right - left + 1
            if window_len < best_len:
                best_len = window_len
                best_left = left

            # Remove s[left] from window
            leaving = s[left]
            window[leaving] -= 1
            if leaving in need and window[leaving] < need[leaving]:
                formed -= 1
            left += 1

    return "" if best_len == float("inf") else s[best_left:best_left + best_len]


if __name__ == "__main__":
    cases = [
        (("ADOBECODEBANC", "ABC"), "BANC"),
        (("a", "a"), "a"),
        (("a", "aa"), ""),
        (("aa", "aa"), "aa"),
        (("cabwefgewcwaefgcf", "cae"), "cwae"),
    ]
    for (s, t), expected in cases:
        result = min_window(s, t)
        status = "\u2713" if result == expected else "\u2717"
        print(f"  {status} min_window({s!r}, {t!r}) = {result!r}  (expected {expected!r})")
    print("OK")
