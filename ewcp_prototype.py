"""Evidence-Weighted Close Probability (EWCP) - reference prototype.

Points are log-odds units (10 points = 1.0 log-odds; ~7 points doubles the odds),
so a deal's score converts directly into a calibrated probability.
Run:  python ewcp_prototype.py
"""
import math
from dataclasses import dataclass, field
from datetime import date

POINTS_PER_LOGIT = 10.0
EVIDENCE_QUALITY = {"claimed": 0.4, "evidenced": 0.8, "buyer_verified": 1.0}


@dataclass
class Milestone:
    key: str
    owner: str              # "seller" or "buyer"
    points: float           # learned (or prior) weight in points; negative = red flag
    half_life_days: float = 0  # 0 = no decay
    gate_cap: float | None = None  # max P(win) while this milestone is NOT complete


@dataclass
class Completion:
    key: str
    on: date
    quality: str = "evidenced"


@dataclass
class Deal:
    name: str
    value: float
    base_rate: float        # historical win rate for this segment
    completions: list[Completion] = field(default_factory=list)


def score(deal: Deal, playbook: dict[str, Milestone], today: date):
    done = {c.key: c for c in deal.completions}
    total_pts, lines = 0.0, []
    for c in deal.completions:
        m = playbook[c.key]
        q = EVIDENCE_QUALITY[c.quality] if m.points > 0 else 1.0
        age = (today - c.on).days
        decay = 0.5 ** (age / m.half_life_days) if m.half_life_days else 1.0
        pts = m.points * q * decay
        total_pts += pts
        lines.append((c.key, round(pts, 1), c.quality, age))
    logit = math.log(deal.base_rate / (1 - deal.base_rate)) + total_pts / POINTS_PER_LOGIT
    p = 1 / (1 + math.exp(-logit))
    caps = [m.gate_cap for k, m in playbook.items() if m.gate_cap is not None and k not in done]
    p_capped = min([p] + caps)
    # next best action: incomplete positive milestone with the largest probability lift
    nba = []
    for k, m in playbook.items():
        if k in done or m.points <= 0:
            continue
        p_new = 1 / (1 + math.exp(-(logit + m.points / POINTS_PER_LOGIT)))
        cap_new = min([p_new] + [c.gate_cap for kk, c in playbook.items()
                                  if c.gate_cap is not None and kk not in done and kk != k])
        nba.append((k, round(cap_new - p_capped, 3)))
    nba.sort(key=lambda x: -x[1])
    return {"points": round(total_pts, 1), "p_raw": round(p, 3), "p_win": round(p_capped, 3),
            "expected_value": round(p_capped * deal.value), "lines": lines, "next_best": nba[:3]}


PLAYBOOK = {m.key: m for m in [
    Milestone("pain_confirmed", "buyer", 4, half_life_days=120),
    Milestone("champion_verified", "buyer", 5, half_life_days=60),
    Milestone("economic_buyer_met", "seller", 6, gate_cap=0.35),
    Milestone("decision_criteria_agreed", "buyer", 4, half_life_days=40),
    Milestone("technical_validation_passed", "buyer", 7),
    Milestone("mutual_action_plan_signed", "buyer", 3, half_life_days=30),
    Milestone("paper_process_started", "buyer", 5, gate_cap=0.75),
    Milestone("close_date_pushed", "seller", -4),
    Milestone("champion_left", "buyer", -8),
]}

if __name__ == "__main__":
    today = date(2026, 11, 9)
    d = Deal("Acme renewal+expansion", 120_000, base_rate=0.20, completions=[
        Completion("pain_confirmed", date(2026, 10, 1), "buyer_verified"),
        Completion("champion_verified", date(2026, 10, 20), "buyer_verified"),
        Completion("economic_buyer_met", date(2026, 10, 25), "evidenced"),
        Completion("decision_criteria_agreed", date(2026, 9, 30), "evidenced"),
        Completion("technical_validation_passed", date(2026, 11, 2), "buyer_verified"),
        Completion("mutual_action_plan_signed", date(2026, 11, 5), "buyer_verified"),
        Completion("close_date_pushed", date(2026, 11, 1)),
    ])
    r = score(d, PLAYBOOK, today)
    for line in r.pop("lines"):
        print("  %-30s %+5.1f pts  (%s, %d days old)" % line)
    for k, v in r.items():
        print(f"{k}: {v}")
