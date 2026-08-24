"""Routing benchmark WITH cost.

Captures token usage per request, attributes it to the tier that served the
request, and prices it at published per-million-token rates. Answers the only
question that matters: does routing actually save money, and how much?
"""
import json, os, subprocess, time, urllib.request, statistics as st

KEY    = os.environ["NVIDIA_API_KEY"]
API    = "https://integrate.api.nvidia.com/v1"
PROXY  = "http://localhost:4000/v1"
WEAK   = "nvidia/nemotron-3.5-lightning-30b-a3b"
STRONG = "nvidia/nemotron-3-ultra-550b-a55b"
MINI   = "nvidia/nemotron-mini-4b-instruct"

# Published per-million-token rates (OpenRouter, Aug 2026).
PRICE = {
    WEAK:   {"in": 0.08, "out": 0.20},
    STRONG: {"in": 0.50, "out": 2.20},
    MINI:   {"in": 0.04, "out": 0.10},   # estimate; 4B is cheaper than Lightning
}

# Descriptive difficulty labels - no borrowed taxonomy.
QUERIES = [
    ("List all plates captured at Gate-3.",                              "weak",   "lookup"),
    ("How many people were detected at Bay-4?",                          "weak",   "lookup"),
    ("Show the most recent detection at Gate-A.",                        "weak",   "lookup"),
    ("What is the average detection confidence per gate?",               "weak",   "aggregate"),
    ("How many vehicles versus people were seen at each gate?",          "weak",   "aggregate"),
    ("Total detections per colour, sorted descending.",                  "weak",   "aggregate"),
    ("Identify people appearing at more than one gate within the same "
     "hour and describe their movement pattern.",                        "strong", "reason"),
    ("Work out which vehicles loitered at Bay-4 for over ten minutes, "
     "showing your reasoning step by step.",                             "strong", "reason"),
    ("Reconstruct the sequence of events at Gate-C and summarise what "
     "likely happened.",                                                 "strong", "reason"),
    ("Find plates seen at Gate-A and Gate-C within 10 minutes of each "
     "other, then correlate them with the people in those vehicles.",    "strong", "multi-hop"),
    ("Build a timeline linking detections at Gate-A, Gate-3 and Bay-4 "
     "that share a plate, and infer the route taken.",                   "strong", "multi-hop"),
    ("Identify pairs of people who consistently appear together across "
     "multiple gates and time windows, and explain the correlation.",    "strong", "multi-hop"),
]

MAX_TOK = 96


def routes_yaml(classifier):
    return f"""
defaults:
  base_url: {API}
  api_key: ${{NVIDIA_API_KEY}}
  format: openai

routes:
  bench-router:
    type: deterministic
    enable_stats: true
    fallback_target_on_evict: weak
    weak:
      model: {WEAK}
    strong:
      model: {STRONG}
    classifier:
      model: {classifier}
"""


def post(url, model, prompt, timeout=240):
    """Return (ms, prompt_tokens, completion_tokens, ok)."""
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAX_TOK}).encode()
    hdrs = {"Content-Type": "application/json"}
    if url == API:
        hdrs["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(f"{url}/chat/completions", data=body, headers=hdrs)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        u = data.get("usage") or {}
        return ((time.time() - t0) * 1000,
                u.get("prompt_tokens", 0), u.get("completion_tokens", 0), True)
    except Exception:
        return ((time.time() - t0) * 1000, 0, 0, False)


def stats():
    try:
        with urllib.request.urlopen("http://localhost:4000/v1/stats", timeout=10) as r:
            return json.load(r)
    except Exception:
        return {}


def tier_counts(s):
    return {k: v.get("calls", 0) for k, v in s.get("tiers", {}).items()}


def restart(cfg):
    subprocess.run("lsof -ti:4000 | xargs -r kill", shell=True,
                   stderr=subprocess.DEVNULL, executable="/bin/bash")
    time.sleep(2)
    open("/tmp/bench_routes.yaml", "w").write(cfg)
    subprocess.run(
        "nohup ~/.local/bin/switchyard serve -c /tmp/bench_routes.yaml -p 4000 "
        "> /tmp/bench_sy.log 2>&1 &", shell=True, executable="/bin/bash")
    for _ in range(60):
        if stats():
            return True
        time.sleep(2)
    return False


def cost(model, pin, pout):
    p = PRICE[model]
    return pin / 1e6 * p["in"] + pout / 1e6 * p["out"]


def run_direct(label, model):
    print(f"\n### {label}", flush=True)
    lat, usd, ok_n = [], 0.0, 0
    for i, (q, _, kind) in enumerate(QUERIES, 1):
        ms, pin, pout, ok = post(API, model, q)
        lat.append(ms); ok_n += ok
        usd += cost(model, pin, pout)
        print(f"   [{i:2}/{len(QUERIES)}] {kind:<10} {ms:7.0f}ms  {pin:>5}in {pout:>4}out", flush=True)
    return {"label": label, "lat": lat, "usd": usd, "ok": ok_n,
            "served": {}, "correct": None, "overhead": 0}


def run_routed(label, classifier):
    print(f"\n### {label}", flush=True)
    if not restart(routes_yaml(classifier)):
        print("   switchyard failed to start"); return None
    lat, usd, ok_n, correct = [], 0.0, 0, 0
    served = {"weak": 0, "strong": 0}
    for i, (q, expect, kind) in enumerate(QUERIES, 1):
        before = tier_counts(stats())
        ms, pin, pout, ok = post(PROXY, "bench-router", q)
        after = tier_counts(stats())
        got = next((k for k in after if after.get(k, 0) > before.get(k, 0)), None)
        lat.append(ms); ok_n += ok
        if got:
            served[got] = served.get(got, 0) + 1
            if got == expect:
                correct += 1
            usd += cost(WEAK if got == "weak" else STRONG, pin, pout)
        # the classifier call is billed too
        cs = stats().get("classifier", {})
        usd += cost(classifier,
                    cs.get("avg_prompt_tokens", 438),
                    cs.get("avg_completion_tokens", 106))
        print(f"   [{i:2}/{len(QUERIES)}] {kind:<10} want={expect:<6} got={str(got):<6} "
              f"{ms:7.0f}ms {pin:>5}in {pout:>4}out", flush=True)
    oh = stats().get("routing_overhead", {})
    return {"label": label, "lat": lat, "usd": usd, "ok": ok_n, "served": served,
            "correct": correct, "overhead": oh.get("p50_ms", oh.get("avg_ms", 0))}


results = [
    run_direct("always STRONG (550B)", STRONG),
    run_direct("always WEAK (30B)", WEAK),
    run_routed("routed: judge = Lightning 30B", WEAK),
    run_routed("routed: judge = Mini 4B", MINI),
]

n = len(QUERIES)
base = next(r["usd"] for r in results if r and r["label"].startswith("always STRONG"))

print("\n\n" + "=" * 108)
print(f"{'config':<32} {'p50 ms':>7} {'ok':>4} {'weak/strong':>12} "
      f"{'acc':>5} {'$ / 1k req':>11} {'vs STRONG':>10}")
print("-" * 108)
for r in results:
    if not r:
        continue
    lat = sorted(r["lat"])
    p50 = lat[len(lat) // 2]
    split = f"{r['served'].get('weak',0)}/{r['served'].get('strong',0)}" if r["served"] else "-"
    acc = f"{100*r['correct']/n:.0f}%" if r["correct"] is not None else "-"
    per_1k = r["usd"] / n * 1000
    delta = f"{100*(r['usd']-base)/base:+.0f}%" if base else "-"
    print(f"{r['label']:<32} {p50:>7.0f} {r['ok']:>4} {split:>12} {acc:>5} "
          f"${per_1k:>10.2f} {delta:>10}")
print("=" * 108)
print(f"\nprices $/M tokens: Lightning in {PRICE[WEAK]['in']} out {PRICE[WEAK]['out']} | "
      f"Ultra in {PRICE[STRONG]['in']} out {PRICE[STRONG]['out']}")
json.dump(results, open("/tmp/bench2_results.json", "w"), indent=1, default=str)
print("saved /tmp/bench2_results.json")
