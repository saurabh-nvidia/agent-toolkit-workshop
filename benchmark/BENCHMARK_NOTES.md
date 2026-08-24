# Switchyard routing benchmark — 20 Aug 2026

Run on `workshop-cpu` (m8i.xlarge, 4 vCPU) against build.nvidia.com.
Script: `bench.py`. Raw log: `bench_full.log`. Raw JSON: `/tmp/bench_results.json` on the box.

## Setup

- **20 queries**, 5 each across four difficulty tiers, from single-table lookups to multi-hop correlation,
  hand-labelled with the tier they *should* route to (L1/L2 → weak, L3/L4 → strong).
- **weak** = `nvidia/nemotron-3.5-lightning-30b-a3b` (30B MoE, 3B active)
- **strong** = `nvidia/nemotron-3-ultra-550b-a55b` (550B MoE, 55B active)
- `max_tokens=64` — we are measuring routing behaviour, not answer quality. End-to-end
  latencies are therefore floor values; routing overhead is unaffected (classifier call is
  fixed-size).
- Tier attribution per request by diffing `/v1/stats` tier counters before and after.

## Results

| config | p50 ms | p95 ms | overhead p50 | weak/strong | accuracy |
|---|---|---|---|---|---|
| baseline: always STRONG (550B) | 2119 | 23695 | 0 | – | – |
| baseline: always WEAK (30B) | 1244 | 3994 | 0 | – | – |
| routed: judge = Lightning 30B | 3359 | 26762 | **2113** | 6/11 | 80%* |
| routed: judge = Nemotron Mini 4B | 2125 | 27994 | **122** | **0/18** | 50%* |

\* accuracy counts failed requests as misses — see below.

## Findings

**1. The Lightning judge routes well — better than the headline number.**
Per-query: L1 5/5 correct, L3 5/5, L4 5/5, L2 1/5 (three failures + one over-escalation).
That is **16/17 registered decisions correct (94%)**, and the single real misroute sent an
L2 query to the strong tier — the safe direction. The 80% figure is depressed because three
requests failed and were counted as misses.

**2. A cheap classifier destroys the whole point.**
Mini 4B cut overhead 17× (2113 ms → 122 ms) but routed **0 weak / 18 strong** — everything to
the expensive model. That is not routing, it is a passthrough with extra latency. Its 50%
"accuracy" comes entirely from strong-tier queries being right by accident.
**The classifier must be capable enough to recognise a simple query.**

**3. Routing costs latency; the win is cost, not speed.**
Routed p50 (3359 ms) is *slower* than always-STRONG (2119 ms), because every request pays a
classifier call first. Always-WEAK is fastest (1244 ms). Anyone selling routing as a latency
win has not measured it. The benefit is that 5–6 of 20 queries never touch the 550B.

**4. Tail latency is dreadful across the board.** p95 of 23–28 s on every config including the
no-routing baselines, so it is the hosted API's tail, not Switchyard's.

## Caveats before quoting any of this externally

- 3 of 20 requests failed under the Lightning judge, 2 under Mini 4B (all returned in
  ~250–1000 ms, i.e. fast errors). **Rerun with retries** before quoting accuracy.
- Single run, no repeats — given the p95 spread, the p50s are indicative only.
- `max_tokens=64` means end-to-end numbers are not representative of real answers.

## How to use this in the workshop

Teach the **method**, not the product. NVIDIA's published claims are cost-side only (~1/3 the
cost, partners reporting 27–58% reductions); this benchmark shows the mechanism those numbers
depend on — a classifier good enough to actually select the weak tier — and gives you a
repeatable way to test whether routing pays for *their* query mix against *their* tiers.

That framing is more credible with an audience that will measure it themselves anyway, and it
maps onto their existing L1–L4 KPI framework rather than replacing it.
