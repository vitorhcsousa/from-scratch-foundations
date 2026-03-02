"""
LC 0424 — Longest Repeating Character Replacement
===================================================
https://leetcode.com/problems/longest-repeating-character-replacement/

Pattern:  Sliding Window — Dynamic (Max-Length), Variant C
Invariant: (right - left + 1) - max_freq <= k
           (chars to replace = window size minus most frequent char count)
Shrink:   while replacements needed > k → freq[s[left]]--, left++
Update:   after shrink (outside loop) — we want the LARGEST valid window

Key trick: max_freq doesn't need to decrease on shrink. Stale max_freq causes
           extra shrinks but never wrong answers, because the answer only
           improves when max_freq increases.

Time:  O(n)    — each char added/removed at most once
Space: O(26)   — frequency counter for uppercase English letters
"""


def character_replacement(s: str, k: int) -> int:
    """Find the longest substring achievable by replacing at most k characters."""
    left = 0
    freq: dict[str, int] = {}
    max_freq = 0
    best = 0

    for right in range(len(s)):
        # EXPAND
        freq[s[right]] = freq.get(s[right], 0) + 1
        max_freq = max(max_freq, freq[s[right]])

        # SHRINK while invariant broken (too many replacements needed)
        while (right - left + 1) - max_freq > k:
            freq[s[left]] -= 1
            left += 1

        # UPDATE: max-length → after restoring validity
        best = max(best, right - left + 1)

    return best


# ── Quick validation ────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        ("ABAB", 2, 4),
        ("AABABBA", 1, 4),
        ("AAAA", 0, 4),
        ("ABCD", 2, 3),
        ("A", 0, 1),
    ]
    for s, k, expected in cases:
        result = character_replacement(s, k)
        status = "✓" if result == expected else "✗"
        print(f"  {status} s={s!r}, k={k} → {result} (expected {expected})")
    print("OK")
