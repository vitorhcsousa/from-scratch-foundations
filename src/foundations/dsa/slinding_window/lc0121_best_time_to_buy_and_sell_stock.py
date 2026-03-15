"""
LC 0121 — Best Time to Buy and Sell Stock
===========================================
https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

Pattern:   Sliding Window / One-pass minimum tracking
Invariant: min_price always holds the lowest price seen so far (days 0..i-1).
           At each day i, the best profit selling on day i is prices[i] - min_price.
Time:  O(n)
Space: O(1)
"""


def max_profit(prices: list[int]) -> int:
    """Find max profit from one buy + one sell (buy before sell)."""
    if len(prices) < 2:
        return 0

    min_price = prices[0]
    best = 0

    for price in prices[1:]:
        # Could we profit by selling today?
        profit = price - min_price
        if profit > best:
            best = profit
        # Update cheapest buying opportunity
        if price < min_price:
            min_price = price

    return best


if __name__ == "__main__":
    cases = [
        ([7, 1, 5, 3, 6, 4], 5),        # buy@1 sell@6
        ([7, 6, 4, 3, 1], 0),            # strictly decreasing
        ([1, 2], 1),                       # minimal profit
        ([2, 4, 1], 2),                    # buy@2 sell@4, not buy@1
        ([3, 3, 5, 0, 0, 3, 1, 4], 4),   # buy@0 sell@4 (last segment)
    ]
    for prices, expected in cases:
        result = max_profit(prices)
        status = "\u2713" if result == expected else "\u2717"
        print(f"  {status} max_profit({prices}) = {result}  (expected {expected})")
    print("OK")
