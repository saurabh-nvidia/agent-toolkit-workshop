# NVIDIA Agent Toolkit — hands-on workshop

Adding NVIDIA's open-source agent tooling to an agent you have already built.

**The premise: none of this replaces your stack.** The notebook builds a LangGraph
text-to-SQL agent, then adds three capabilities on top — without changing the agent,
the framework, or how you serve your model.

| Lever | What it adds | What you change |
|---|---|---|
| **NeMo Relay** — visibility | full execution trajectory, latency attribution | 2 lines |
| **NeMo Relay** — control | policy enforced *at the runtime*, not in your tool code | 1 function + 1 registration |
| **NeMo Switchyard** — routing | cheap model for simple questions, big model for hard ones | 1 URL |

Every section shows **before** and **after**, so the difference is demonstrated rather
than asserted.

---

## Before the session

Read **[ATTENDEE_PREWORK.md](ATTENDEE_PREWORK.md)** — about 10 minutes. You need a Brev
account and a free API key from build.nvidia.com.

## During the session

You will be given a Launchable link. It asks for your API key, builds the environment, and
opens Jupyter. Open `agent_workshop.ipynb` and run the cells in order.

No installs, no terminal, no configuration. **No GPU.**

The first cell waits for the setup script to finish and tells you what it is doing — a
fresh machine takes 2–5 minutes to become ready.

---

## What is in here

```
notebook/     the workshop notebook — edit this directly
benchmark/    routing-overhead benchmark, results and method
optional/     running the same notebook against your own GPU
docs/         a minimal Relay proof script, and the original generator (historical)
setup.sh      the Launchable setup script (runs at first boot)
```

`notebook/agent_workshop.ipynb` is the source of truth — edit it in Jupyter like any other
notebook.

`docs/build_nb_ORIGINAL.py` generated the first version and is kept for provenance only.
**Do not run it** — it would overwrite the notebook and discard any edits since.

---

## A note on the numbers

Section 7 prints a routing-overhead figure from two requests. **That number is noise** —
across four identical runs it came back as 1820 ms, 2174 ms, 3378 ms and 5796 ms. The
notebook says so, and shows the proper measurement instead: 20 queries across four
configurations, in [benchmark/BENCHMARK_NOTES.md](benchmark/BENCHMARK_NOTES.md).

The headline findings there are worth knowing before you adopt routing:

- A cheap classifier destroys the benefit entirely — a 4B judge routed **0 of 18** requests
  to the cheap tier
- Routing makes you **slower**, not faster; the win is cost
- With a capable judge, routing accuracy was 16 of 17 completed decisions

Run `benchmark/bench.py` against your own labelled queries to find out whether it pays for
your workload. It takes about 35 minutes.

---

## Versions this was verified against

| Component | Version |
|---|---|
| nemo-relay | 0.7.3 |
| nemo-switchyard | 0.2.0 |
| langgraph | 1.2.10 |
| langchain | 1.3.14 |
| Python | 3.12 |

Switchyard is **pre-alpha** — its own documentation says "not for production use". Treat
section 7 as an evaluation, not a deployment recommendation.
