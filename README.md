# Points-Based Sales Forecasting: Research & Proposed Design

*Research date: November 2026. Many sources are vendor blogs, so treat feature claims as directional and check them with each vendor.*

---

## Part 1 — Do such systems already exist?

Yes, but in pieces. Nobody sells a single standard "points for weighted seller and buyer activities → probability to close" product. What exists falls into five families, and each one covers only part of the idea.

### 1. Stage-weighted pipeline (the CRM default)
*Salesforce, HubSpot, Dynamics 365, Oracle Sales, Pipedrive.*
- Each stage carries a fixed probability (e.g. Discovery 10%, Proposal 50%). Expected value = deal value × stage %.
- Better setups add **exit criteria**: the deal can't move to the next stage until named actions are done.
- **Weaknesses:** every deal in a stage gets the same probability. The stage % is often a guess, not measured. Reps move stages by hand. There's no difference between "the rep says" and "the buyer did".

### 2. Qualification scorecards (methodology-driven points)
*MEDDIC/MEDDPICC, Miller Heiman Blue Sheet, BANT, Value Selling, Force Management Command of the Message, Kurlan "Opportunity Scorecard", Gartner/CEB opportunity scorecards.*
- These are the closest match to your idea. Each element (Metrics, Economic Buyer, Decision Criteria, Decision Process, Paper Process, Implicate Pain, Champion, Competition) gets a score, often 0–3. The total, or a rule set, gives a qualification grade.
- **Weaknesses:** weights are set by hand and equal by default. The score is usually **not calibrated to a probability**. Answers go stale. Reps grade themselves, so the scores drift upward.

### 3. Mutual Action Plans and buyer-side milestones
*Salesforce, Outreach, Highspot, Aviso, Clari Align, Dock, Accord, GetAccept, DealHub.*
- A plan shared by buyer and seller, listing milestones, owners and dates. Good practice ties each milestone to buyer outcomes and approvals.
- Newer tools track **buyer engagement with the plan** and feed it into deal health. Example: a buyer who completed 3 of 5 milestones in two weeks is healthier than one who hasn't touched the plan.
- **Weaknesses:** completed milestones are rarely turned into a probability, and they usually sit apart from the forecast.

### 4. ML opportunity scoring (built into CRMs)
*Salesforce Einstein Opportunity Scoring, Dynamics 365 Predictive Opportunity Scoring, HubSpot deal scoring.*
- A model trained on past wins and losses gives each open deal a 1–99 score, with "top positive and negative factors".
- **Weaknesses:** it needs a lot of history. Einstein reportedly needs about 200 closed-won deals in 24 months. It sees only what is logged in the CRM, so a deal with no logged activity looks the same as a stalled one. It doesn't know your sales process, so it can't tell the rep *which* step to do next.

### 5. Revenue intelligence and AI deal inspection
*Clari, Gong (Forecast, AI Deal Reviewer), BoostUp (reportedly renamed Terret), Aviso, People.ai, Ebsta/Fullcast, Spotlight, Coffee.*
- They collect email, calendar and call data automatically. Some auto-score MEDDPICC from call transcripts and quote the supporting evidence. They also flag risks such as a missing economic buyer, single-threaded deals and pushed close dates. On top of that they offer AI forecast roll-ups, scenario planning and variance analysis.
- **Weaknesses:** the scores are often **not probabilities**. Gong's score is reportedly a percentile rank: 80 means "healthier than 80% of deals", not "80% likely to close". The models are often black boxes and cost enterprise prices. They also measure *activity volume* better than *buyer commitment*.

### Feature summary

| Capability | Stage % | Scorecards | MAPs | CRM ML | Rev-Intel |
|---|:-:|:-:|:-:|:-:|:-:|
| Uses your defined sales process | ● | ● | ● | ○ | ◐ |
| Weighted activities / points | ○ | ● | ○ | ◐ | ◐ |
| Buyer-side (customer) activities | ○ | ◐ | ● | ○ | ◐ |
| Weights learned from outcomes | ◐ | ○ | ○ | ● | ● |
| Output is a calibrated probability | ◐ | ○ | ○ | ◐ | ◐ |
| Explainable per deal | ● | ● | ● | ◐ | ◐ |
| Evidence quality (claimed vs verified) | ○ | ○ | ◐ | ○ | ◐ |
| Time decay / staleness | ○ | ○ | ◐ | ◐ | ● |
| Separates *win* from *close this period* | ○ | ○ | ◐ | ○ | ◐ |
| Recommends next best action | ○ | ◐ | ◐ | ○ | ◐ |

● strong ◐ partial ○ absent

**The gap:** no common system combines an **explicit, process-defined points model** (explainable like a scorecard, with buyer actions like a MAP) with **weights learned from outcomes and calibrated to a real probability** (like ML), plus evidence quality, decay and timing.

---

## Part 2 — Proposed system: Evidence-Weighted Close Probability (EWCP)

### Core idea
**Points are log-odds.** Each milestone's points are its measured contribution to the log-odds of winning (10 points = 1.0 log-odds, so about **+7 points doubles the odds**). This brings three benefits:
- The score converts straight into a probability. No "what does 62 points mean?"
- Points can be set by hand at first, then **learned from outcomes** with the same meaning.
- Each point on a deal can be traced to one milestone, so the score stays as explainable as a scorecard.

```
logit(P_win) = logit(segment base rate)
             + Σ  points_i × evidence_quality_i × decay_i      (completed milestones)
             − Σ  red_flag_points_j                            (risk events)
P_win        = min( sigmoid(logit) , gate caps of unmet must-haves )
P_close_in_period = P_win × P_timing(remaining milestones, days left, velocity)
Forecast     = Σ value × P_close_in_period   (+ Monte Carlo for a P10/P50/P90 range)
```

### Components

1. **Process playbook (configurable).** Milestones per segment or motion. Each one records an owner (**seller** or **buyer**), a prior point weight, an optional decay half-life, an optional **gate cap** and dependencies. Buyer actions (e.g. "buyer ran the technical validation", "buyer introduced procurement") usually earn more than seller actions ("sent proposal").

2. **Evidence quality multiplier.** One milestone scores differently depending on proof:
   - *Claimed* by the rep: ×0.4
   - *Evidenced*, e.g. a meeting on the calendar, a document or a call transcript: ×0.8
   - *Buyer-verified*, e.g. ticked off by the buyer in the shared MAP or a signed document: ×1.0

   This makes inflated self-grading worth less and pushes reps toward a real shared plan with the buyer.

3. **Time decay.** Soft milestones lose points over time; `decay = 0.5^(age / half_life)`. For example, a "decision criteria agreed" made 40 days ago in a 40-day half-life loses half its value. Hard events, such as a passed security review, don't decay.

4. **Red flags (negative points).** Close date pushed, champion left, competitor preferred, the buyer misses MAP dates, the deal goes quiet longer than the benchmark, single-threaded contact.

5. **Gate caps.** Must-haves limit the probability no matter how many other points a deal earns. Examples: no economic buyer contact → P_win ≤ 35%; paper process not started → P_win ≤ 75%. This stops "lots of activity" from outscoring "missing the one thing that matters".

6. **Separate *if* from *when*.** P_win measures *if* the deal closes. P_timing measures *when*. P_timing comes from how many critical-path milestones are left, compared with days to period end and historical cycle time per milestone, which is a survival/hazard model. A strong deal with two months of procurement left shouldn't be in this quarter's forecast.

7. **Learning loop (cold start → data-driven).**
   - *Day 0:* expert priors. Sales leaders set the weights in the same log-odds units.
   - *As deals close:* fit a **Bayesian or regularised logistic regression**, with milestone completion × quality × decay as features and the expert priors as the prior. Weights move toward the data only as fast as the evidence supports. This works with tens of closed deals, unlike the ~200+ that pure ML tools need.
   - *Quarterly:* re-fit per segment, and flag milestones whose learned weight is near 0 as "process theatre".

8. **Next best action.** For each open milestone, compute the probability lift if it were completed, taking caps into account. Show the top three to the rep. This turns the forecast into coaching.

9. **Forecast roll-up.** Expected value plus a Monte Carlo range (P10/P50/P90). Optionally show the rep's own call next to the model's, and track the gap between them.

### Worked example (from `ewcp_prototype.py`)
Segment base rate is 20% (logit −1.39). Deal value is $120k.

| Milestone | Base pts | Quality | Age / decay | Earned |
|---|--:|---|---|--:|
| Pain confirmed | 4 | buyer-verified | 39d, HL 120 | +3.2 |
| Champion verified | 5 | buyer-verified | 20d, HL 60 | +4.0 |
| Economic buyer met | 6 | evidenced | — | +4.8 |
| Decision criteria agreed | 4 | evidenced | 40d, HL 40 | +1.6 |
| Technical validation passed | 7 | buyer-verified | — | +7.0 |
| Mutual action plan signed | 3 | buyer-verified | 4d, HL 30 | +2.7 |
| Close date pushed (red flag) | −4 | — | — | −4.0 |
| **Total** | | | | **19.3** |

logit = −1.39 + 1.93 = 0.54 → **P_win = 63%**, so the expected value is **≈ $76k**. The paper-process cap (75%) isn't binding yet. The **next best action** is "paper process started", worth about +11 percentage points.

### Why it is better than what exists

| Problem with existing systems | EWCP answer |
|---|---|
| Stage % treats all deals in a stage the same | The probability is specific to each deal |
| Scorecard points have no probability meaning | Points are log-odds, so they are calibrated by design |
| Reps grade themselves upward | Evidence-quality multiplier; buyer verification needed for full points |
| Stale qualification | Decay half-lives |
| ML needs big data and is a black box | Expert priors + Bayesian updating; every point can be traced |
| "Will win" mixed up with "will win this quarter" | Separate P_win and P_timing |
| Scores don't say what to do | Next best action ranked by probability lift |

---

## Part 2b — From probabilities to a quarterly number (the 0%/100% problem)

Each deal is either won or lost, so the total of value × probability is only the **average** of many possible quarters. The real quarter will rarely land on it. With a few large deals, the possible results are spread wide and bunched around particular deal combinations, and the average may not be a possible result at all. The fix is to **forecast a distribution and manage against it**, not to report one number.

1. **Forecast the full distribution.** Work out the exact spread of possible totals (a Poisson-binomial distribution over deal values), or simulate it. Report P10/P50/P90 and the **probability of hitting the target**, not a single figure.
2. **Model deals that move together.** Deals share the economy, budget cycles, a product release and the same managers, so they aren't independent. A shared "quarter factor" (`rho`) is fitted from how far past quarters missed. Without it the range looks narrower, and more certain, than it is.
3. **Three outcomes, not two.** Each deal is *won this quarter*, *slips to next quarter* or *lost*, using P_win × P_timing from EWCP. Won deals can also shrink in value. Slipped deals feed next quarter's forecast instead of disappearing, which makes misses look less like 0% surprises.
4. **Swing deals.** Each deal's share of the uncertainty is `value² × p × (1−p)`. A few deals usually drive most of it. Forecast those deal by deal, as named scenarios ("Globex in / out"), and focus leadership help and next-best actions on them.
5. **Split lumpy and steady business.** Many small, frequent deals (renewals, run-rate, transactional) average out, so their expected value is reliable. Forecast them with time-series or conversion-rate methods, and keep the deal-by-deal distribution for large deals.
6. **Turn the distribution into decisions.** Report "P(hit plan) = 42%", along with the extra pipeline, or the specific deal advances, needed to raise it to a chosen confidence (e.g. 80%). Commit/best-case calls then come from percentiles, not rep instinct: *Commit ≈ P75–P80, Best case ≈ P25*.
7. **Re-forecast every week.** The range narrows as deals close or are lost. Show how it changed from last week, as a waterfall of which deals moved the number.
8. **Blend with a top-down check.** Compare the bottom-up distribution with past "pipeline at week N → final bookings" ratios. A large gap between the two is a warning sign in itself.
9. **Score the range, not the point.** Judge forecasts with pinball loss/CRPS, and check that results land below P10 about 10% of the time and above P90 about 10% of the time. Also check that the stated P(hit target) is accurate over many quarters.

**Example (`forecast_distribution.py`, 10 deals, target $450k):**

| View | Result |
|---|---|
| Sum of value × probability | $455k, which looks like "on target" |
| Chance of landing within ±$500 of that average | 0.7% |
| Independent deals: P10 / P50 / P90 | $170k / $420k / $785k, **P(hit) 47%** |
| Correlated + slippage + downsizing: P10 / P50 / P90 | $84k / $362k / $832k, **P(hit) 42%** |
| Expected value slipping to next quarter | $165k |
| Swing deals | Globex 61%, Umbrella 20% of the uncertainty |

The single-number forecast says the target is met. The distribution says it is close to a coin flip and depends mainly on two deals.

---

## Part 3 — Evaluation plan

1. **Backtest** on 12–24 months of closed deals. Rebuild each deal's milestone history as it stood at weekly snapshots, then score each snapshot.
2. **Baselines:** CRM stage %, the rep's own commit call, an equal-weight MEDDPICC score, and (if available) Einstein or vendor scores.
3. **Metrics:**
   - *Discrimination:* AUC and log-loss for win/loss.
   - *Calibration:* Brier score and reliability plots. Does "60%" win about 60% of the time?
   - *Forecast accuracy:* period-end error (MAPE / bias) at weeks −8, −4 and −1.
   - *Timing:* accuracy of the close-in-period prediction.
4. **Ablations:** remove evidence quality, decay, gate caps and buyer milestones one at a time to check that each one earns its keep.
5. **Pilot:** one team over one quarter, comparing the model against rep calls. Watch for gaming, e.g. a jump in "claimed" entries, and check how often reps follow the next best action.
6. **Success criteria (suggested):** Brier score at least 15% better than stage %, quarterly forecast error under ±10% at week −4, and reps using the next best action on at least half of their open deals.

---

## Sources
- [Rework — Weighted pipeline](https://resources.rework.com/libraries/pipeline-management/weighted-pipeline)
- [Forrester — Stop asking salespeople to estimate probabilities](https://www.forrester.com/blogs/stop-asking-salespeople-to-estimate-probabilities-in-their-forecast-process/)
- [Kurlan & Associates — Opportunity scorecard](https://www.kurlanassociates.com/?p=8343)
- [Oracle — Overview of Opportunities](https://docs.oracle.com/en/cloud/saas/sales/fastg/overview-of-opportunities.html)
- [Fullcast/Ebsta — What is a Deal Score?](https://support.fullcast.com/revenue-intelligence/docs/what-is-ebsta-s-deal-score)
- [Prospeo — Deal scoring 2026 guide](https://prospeo.io/s/deal-scoring)
- [Spotlight — Best MEDDICC tools 2026](https://www.spotlight.ai/post/best-meddicc-sales-tools-and-software-for-2026)
- [Coffee — Best MEDDPICC AI tools 2026](https://coffee.ai/articles/best-meddpicc-ai-tools-2026)
- [Docket — Clari review](https://docket.io/resources/research/clari-review)
- [Salesforce — Mutual action plan](https://www.salesforce.com/blog/mutual-action-plan/)
- [Octave — GTM engineer's guide to MAPs](https://www.octavehq.com/post/gtm-engineers-guide-to-mutual-action-plans)
- [Fitgap — Align MAPs with live buyer behaviour](https://us.fitgap.com/stack-guides/align-mutual-action-plans-with-live-buyer-behavior)
- [Aviso — Mutual action plan](https://www.aviso.com/blog/mutual-action-plan)
- [Highspot — Mutual action plan](https://www.highspot.com/blog/mutual-action-plan/)
- [Coffee — Salesforce deal scoring: when Einstein falls short](https://coffee.ai/articles/salesforce-deal-scoring-einstein)
- [Peeklogic — Salesforce Einstein opportunity scoring](https://www.peeklogic.com/blog/salesforce-einstein-opportunity-scoring/)
