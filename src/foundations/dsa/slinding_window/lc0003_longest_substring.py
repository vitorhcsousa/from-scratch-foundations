"""
LC 0003 — Longest Substring Without Repeating Characters
=========================================================
https://leetcode.com/problems/longest-substring-without-repeating-characters/

Pattern:  Sliding Window — Dynamic (Max-Length), Variant C
Invariant: All characters in window s[left..right] are unique
Shrink:   while s[right] already in seen → remove s[left], left++
Update:   after shrink (outside loop) — we want the LARGEST valid window

Time:  O(n)  — each char added/removed from set at most once
Space: O(min(n, |Σ|))  — set holds at most alphabet-size chars
"""


def length_of_longest_substring(s: str) -> int:
    """Find the length of the longest substring without repeating characters."""
    left = 0
    seen: set[str] = set()
    best = 0

    for right in range(len(s)):
        # SHRINK: remove from left until duplicate is gone
        while s[right] in seen:
            seen.discard(s[left])
            left += 1

        # EXPAND: add current char
        seen.add(s[right])

        # UPDATE: max-length → update after restoring validity
        best = max(best, right - left + 1)

    return best


def length_of_longest_substring_v2(s: str) -> int:
    """Optimised: hash map jump — avoids inner while loop."""
    last_seen: dict[str, int] = {}
    left = 0
    best = 0

    for right in range(len(s)):
        char = s[right]
        if char in last_seen and last_seen[char] >= left:
            left = last_seen[char] + 1
        last_seen[char] = right
        best = max(best, right - left + 1)

    return best


# ── Quick validation ────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("", 0),
        ("a", 1),
        ("abcdef", 6),
        ("dvdf", 3),
    ]
    for s, expected in cases:
        r1 = length_of_longest_substring(s)
        r2 = length_of_longest_substring_v2(s)
        status = "✓" if r1 == expected == r2 else "✗"
        print(f"  {status} s={s!r:16s} → {r1} (expected {expected})")
    print("OK")
