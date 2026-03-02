"""
LC 0567 — Permutation in String
=================================
https://leetcode.com/problems/permutation-in-string/

Pattern:  Fixed Sliding Window, Variant A/D (existence check)
Invariant: window_size == len(s1) AND window counter == need counter
Shrink:   exactly one element per step (fixed window — remove s2[right - n])
Update:   check counter match after each slide

Key: "permutation as substring" = "anagram exists" = "same char frequencies"

Time:  O(|s2|)  — one pass with O(26) counter comparison per step
Space: O(26)    — two counters with at most 26 keys
"""

from collections import Counter


def check_inclusion(s1: str, s2: str) -> bool:
    """Check if any permutation of s1 exists as a substring of s2."""
    n = len(s1)
    if n > len(s2):
        return False

    need = Counter(s1)
    window = Counter(s2[:n])

    if window == need:
        return True

    for right in range(n, len(s2)):
        # EXPAND: add new right char
        window[s2[right]] += 1

        # SHRINK: remove char exiting the fixed window
        left_char = s2[right - n]
        window[left_char] -= 1
        if window[left_char] == 0:
            del window[left_char]

        # CHECK: counter match
        if window == need:
            return True

    return False


# ── Quick validation ────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        ("ab", "eidbaooo", True),
        ("ab", "eidboaoo", False),
        ("adc", "dcda", True),
        ("a", "a", True),
        ("ab", "a", False),
        ("abc", "bbbca", True),
    ]
    for s1, s2, expected in cases:
        result = check_inclusion(s1, s2)
        status = "✓" if result == expected else "✗"
        print(f"  {status} s1={s1!r}, s2={s2!r} → {result} (expected {expected})")
    print("OK")
