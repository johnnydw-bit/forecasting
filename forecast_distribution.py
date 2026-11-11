"""Turn per-deal probabilities into a quarterly forecast *distribution*.

Each deal resolves to all-or-nothing, so the sum of value x probability is only the
mean of a lumpy distribution. This module reports the full distribution instead:
  1. exact outcome distribution for independent deals (Poisson-binomial over values)
  2. correlated Monte Carlo (one shared "quarter factor") with slippage and downsizing
  3. P(hit target), P10/P50/P90, and the "swing deals" that drive the uncertainty
Run:  python forecast_distribution.py
"""
import random
from statistics import NormalDist

N = NormalDist()


def exact_distribution(deals, bucket=1000):
    """deals: list of (name, value, p). Returns {total_value: probability}, assuming independence."""
    dist = {0: 1.0}
    for _, v, p in deals:
        b = round(v / bucket)
        nxt = {}
        for k, pr in dist.items():
            nxt[k] = nxt.get(k, 0) + pr * (1 - p)
            nxt[k + b] = nxt.get(k + b, 0) + pr * p
        dist = nxt
    return {k * bucket: pr for k, pr in sorted(dist.items())}


def quantile(dist, q):
    c = 0.0
    for v, pr in dist.items():
        c += pr
        if c >= q:
            return v
    return max(dist)


def p_at_least(dist, target):
    return sum(pr for v, pr in dist.items() if v >= target)


def simulate(deals, rho=0.25, runs=20000, downsize=(0.75, 1.0, 1.0), seed=7):
    """deals: list of (name, value, p_win, p_timing).
    rho = how strongly deals move together (shared economy, quarter-end budget, same team).
    A deal is booked this quarter with probability p_win * p_timing. Won deals can shrink
    (triangular downsize: low, high, mode). Slipped deals go to next quarter rather than counting as lost."""
    rnd = random.Random(seed)
    totals, slipped_totals = [], []
    for _ in range(runs):
        m = rnd.gauss(0, 1)
        tot = slip = 0.0
        for _, v, pw, pt in deals:
            u = N.cdf(rho ** 0.5 * m + (1 - rho) ** 0.5 * rnd.gauss(0, 1))  # correlated uniform
            if u < pw * pt:
                tot += v * rnd.triangular(*downsize)
            elif u < pw:
                slip += v
        totals.append(tot)
        slipped_totals.append(slip)
    totals.sort()
    pct = lambda q: totals[int(q * (runs - 1))]
    return totals, pct, sum(slipped_totals) / runs


def swing_deals(deals):
    """Each deal's share of forecast variance, v^2 * p * (1 - p): the deals the number really depends on."""
    var = [(n, v * v * p * (1 - p)) for n, v, p in deals]
    tot = sum(x for _, x in var)
    return sorted(((n, x / tot) for n, x in var), key=lambda t: -t[1])


if __name__ == "__main__":
    # name, value, p_win, p_timing (from the EWCP model)
    pipeline = [
        ("Acme", 120_000, 0.63, 0.80), ("Globex", 400_000, 0.55, 0.60),
        ("Initech", 60_000, 0.80, 0.90), ("Umbrella", 250_000, 0.35, 0.70),
        ("Hooli", 45_000, 0.70, 0.95), ("Stark", 90_000, 0.50, 0.85),
        ("Wayne", 30_000, 0.85, 0.95), ("Wonka", 75_000, 0.40, 0.80),
        ("Tyrell", 150_000, 0.25, 0.60), ("Cyberdyne", 35_000, 0.60, 0.90),
    ]
    target = 450_000
    in_q = [(n, v, pw * pt) for n, v, pw, pt in pipeline]
    ev = sum(v * p for _, v, p in in_q)
    print(f"Sum of value x probability (the usual 'forecast'): ${ev:,.0f}  target ${target:,.0f}\n")

    d = exact_distribution(in_q)
    print("Independent deals (exact):")
    print(f"  P10 ${quantile(d, .1):,.0f}   P50 ${quantile(d, .5):,.0f}   P90 ${quantile(d, .9):,.0f}")
    print(f"  P(hit target) {p_at_least(d, target):.0%}   P(exactly the EV) "
          f"{d.get(round(ev, -3), 0):.1%}\n")

    totals, pct, slip = simulate(pipeline)
    hit = sum(t >= target for t in totals) / len(totals)
    print("Correlated (rho=0.25) + slippage + downsizing (Monte Carlo):")
    print(f"  P10 ${pct(.1):,.0f}   P50 ${pct(.5):,.0f}   P90 ${pct(.9):,.0f}   P(hit target) {hit:.0%}")
    print(f"  Expected value slipping into next quarter: ${slip:,.0f}\n")

    print("Swing deals (share of forecast variance):")
    for n, s in swing_deals(in_q)[:4]:
        print(f"  {n:<10} {s:5.0%}")
