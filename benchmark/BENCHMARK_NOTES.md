# Routing benchmark — does it actually save money?

Run against build.nvidia.com on a 4 vCPU CPU instance. Script: `bench.py`.

## Setup

- **12 queries**, three each across four difficulty levels (`lookup`, `aggregate`, `reason`,
  `multi-hop`), each hand-labelled with the tier it *should* route to.
- **weak** = `nvidia/nemotron-3.5-lightning-30b-a3b` (30B MoE, 3B active)
- **strong** = `nvidia/nemotron-3-ultra-550b-a55b` (550B MoE, 55B active)
- `max_tokens=96`. Token usage captured per request and priced at published rates.
- Tier attribution by diffing `/v1/stats` counters around each request — this is what makes
  routing **accuracy** measurable, not just cost.

Prices ($ per million tokens): Lightning **0.08 in / 0.20 out**, Ultra **0.50 in / 2.20 out**.
Ultra costs **6.3x** more on input, **11x** on output.

## Results

Normalised per **successful** request:

| config | $ / request | vs always-strong | tier split | accuracy | p50 latency |
|---|---|---|---|---|---|
| always STRONG (550B) | $0.000227 | — | — | — | 2004 ms |
| always WEAK (30B) | $0.000022 | **−90%** | — | — | 2214 ms |
| **routed, judge = Lightning 30B** | **$0.000178** | **−22%** | 6 weak / 5 strong | **92%** | 5283 ms |
| routed, judge = Mini 4B | $0.000255 | **+13%** | 0 weak / 12 strong | 50% | 6225 ms |

## Findings

**1. Routing saved 22%, not the 74% NVIDIA publishes.** Both numbers are real; the difference
is workload shape — see lever 3.

**2. The three levers that decide whether routing pays.**

- **Price gap between tiers.** 11x on output here. Narrow the gap and the arithmetic collapses.
- **Fraction routed to the cheap tier.** 6 of 11. A property of *your traffic*, not the router.
- **The classifier tax.** The judge is an LLM call on every request. Classifier prompt was
  ~438 in / 106 out; our questions were ~25 in / 96 out. **The classifier prompt was ~17x
  larger than the question.** It consumed **32% of routed cost** — without it, savings would
  have been **46%** rather than 22%.

**3. The rule.** Routing pays when request cost is large relative to classifier cost. At these
prices with ~50% weak-routing, break-even is around **50 output tokens**. Below that routing
*loses* money. A 2,000-output-token agent turn makes the classifier ~2.5% of cost, which is
where published figures like 74% come from.

**Short cheap requests are the worst case for routing. Long agent turns are the best case.**

**4. A cheap judge is not a cheap win.** Mini 4B cut routing overhead 17x (2113 ms → 122 ms)
and routed **0 of 12** to the cheap tier — everything to the 550B, plus the classifier on top.
**13% more expensive than not routing**, at 50% accuracy. A router that cannot recognise an
easy question is strictly worse than no router.

**5. Routing costs latency.** Routed p50 (5283 ms) is worse than always-strong (2004 ms);
every request pays a classifier call first. The win is cost, not speed.

## A measurement bug worth repeating

The first version divided total cost by *all* queries rather than *successful* ones. With 3
failures in the strong baseline contributing $0, the baseline was understated and a genuine
22% saving looked like 4%. Normalise by what actually completed.

## Caveats

- 12 queries, 1–3 failures per config. Indicative, not authoritative.
- Mini 4B pricing is estimated, not published.
- Accuracy is measured against labels we assigned by hand.
- Single run. Given API latency ranging 2–67 s on identical work, latency figures are noisy.

**Re-run against your own traffic before drawing conclusions.**
