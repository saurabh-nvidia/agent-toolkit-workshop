"""Routing-overhead benchmark: does tiered routing pay for itself, and what drives the cost?

Runs a fixed query set through several Switchyard configurations and a no-routing baseline,
measuring per-request latency, which tier served it, and routing accuracy against
hand-labelled expected tiers.
"""
import json, os, subprocess, time, urllib.request, statistics as st

KEY      = os.environ["NVIDIA_API_KEY"]
API      = "https://integrate.api.nvidia.com/v1"
PROXY    = "http://localhost:4000/v1"
WEAK     = "nvidia/nemotron-3.5-lightning-30b-a3b"
STRONG   = "nvidia/nemotron-3-ultra-550b-a55b"
MINI     = "nvidia/nemotron-mini-4b-instruct"
MAX_TOK  = 64          # capped: we measure routing, not answer quality

# Four difficulty tiers, from single-table lookups to multi-hop correlation.
# L1/L2 are simple -> should route weak. L3/L4 are multi-step -> should route strong.
QUERIES = [
    ("List all plates captured at Gate-3.", "weak", "L1"),
    ("How many people were detected at Bay-4?", "weak", "L1"),
    ("Show the most recent detection at Gate-A.", "weak", "L1"),
    ("Which gates exist in the detections table?", "weak", "L1"),
    ("Count the red vehicles.", "weak", "L1"),
    ("Count unique people per hour per gate last week.", "weak", "L2"),
    ("What is the average detection confidence per gate?", "weak", "L2"),
    ("How many vehicles versus people were seen at each gate?", "weak", "L2"),
    ("Which gate had the most detections between 06:00 and 09:00?", "weak", "L2"),
    ("Total detections per colour, sorted descending.", "weak", "L2"),
    ("Show images of red vehicles seen near Gate-3 yesterday and explain why each matched.",
     "strong", "L3"),
    ("Identify people who appear at more than one gate within the same hour, and describe "
     "their movement pattern.", "strong", "L3"),
    ("Work out which vehicles loitered at Bay-4 for more than ten minutes, showing your "
     "reasoning step by step.", "strong", "L3"),
    ("Determine whether any person entered a restricted zone after hours, and justify the "
     "conclusion from the data.", "strong", "L3"),
    ("Reconstruct the sequence of events at Gate-C and summarise what likely happened.",
     "strong", "L3"),
    ("Find plates seen at Gate-A and Gate-C within 10 minutes of each other, then correlate "
     "them with the people detected in those vehicles.", "strong", "L4"),
    ("Trace every person seen with a red upper garment across all gates in time order, then "
     "identify which vehicles they were near.", "strong", "L4"),
    ("Correlate vehicle plates across gates to find repeat visitors, then cross-reference "
     "with people detections at the same timestamps.", "strong", "L4"),
    ("Build a timeline linking detections at Gate-A, Gate-3 and Bay-4 that share a plate, "
     "and infer the route taken.", "strong", "L4"),
    ("Identify pairs of people who consistently appear together across multiple gates and "
     "time windows, and explain the correlation.", "strong", "L4"),
]

def routes_yaml(classifier, affinity=False):
    aff = "    session_affinity: true\n" if affinity else ""
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
{aff}    weak:
      model: {WEAK}
    strong:
      model: {STRONG}
    classifier:
      model: {classifier}
"""

def post(url, model, prompt, timeout=180):
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
            r.read()
        return (time.time() - t0) * 1000, True
    except Exception:
        return (time.time() - t0) * 1000, False

def stats():
    try:
        with urllib.request.urlopen("http://localhost:4000/v1/stats", timeout=10) as r:
            return json.load(r)
    except Exception:
        return {}

def tier_counts(s):
    return {k: v.get("calls", 0) for k, v in s.get("tiers", {}).items()}

def restart_switchyard(cfg_text):
    subprocess.run("tmux kill-session -t bench 2>/dev/null", shell=True)
    open("/tmp/bench_routes.yaml", "w").write(cfg_text)
    subprocess.run(
        "tmux new-session -d -s bench 'source ~/.env_keys; "
        "~/.local/bin/switchyard serve -c /tmp/bench_routes.yaml -p 4000 > /tmp/bench_sy.log 2>&1'",
        shell=True, executable="/bin/bash")
    for _ in range(40):
        if stats():
            return True
        time.sleep(2)
    return False

def run_routed(label, classifier, affinity=False):
    print(f"\n### {label}", flush=True)
    if not restart_switchyard(routes_yaml(classifier, affinity)):
        print("   switchyard failed to start"); return None
    lat, correct, served = [], 0, {"weak": 0, "strong": 0}
    for i, (q, expected, tier_name) in enumerate(QUERIES, 1):
        before = tier_counts(stats())
        ms, ok = post(PROXY, "bench-router", q)
        after = tier_counts(stats())
        got = next((k for k in after if after.get(k, 0) > before.get(k, 0)), None)
        if got:
            served[got] = served.get(got, 0) + 1
            if got == expected:
                correct += 1
        lat.append(ms)
        print(f"   [{i:2}/{len(QUERIES)}] {tier_name} expect={expected:<6} got={str(got):<6} {ms:7.0f}ms",
              flush=True)
    s = stats()
    oh = s.get("routing_overhead", {})
    return {"label": label, "lat": lat, "correct": correct, "served": served,
            "overhead_avg": oh.get("avg_ms", 0), "overhead_p50": oh.get("p50_ms", 0),
            "overhead_p99": oh.get("p99_ms", 0),
            "tokens": s.get("total_tokens", {}).get("total", 0)}

def run_direct(label, model):
    print(f"\n### {label}", flush=True)
    lat = []
    for i, (q, _, tier_name) in enumerate(QUERIES, 1):
        ms, ok = post(API, model, q)
        lat.append(ms)
        print(f"   [{i:2}/{len(QUERIES)}] {tier_name} {ms:7.0f}ms", flush=True)
    return {"label": label, "lat": lat, "correct": None, "served": {},
            "overhead_avg": 0, "overhead_p50": 0, "overhead_p99": 0, "tokens": 0}

results = []
results.append(run_direct("baseline: always STRONG (550B)", STRONG))
results.append(run_direct("baseline: always WEAK (30B)", WEAK))
results.append(run_routed("routed: judge = Lightning 30B", WEAK))
results.append(run_routed("routed: judge = Nemotron Mini 4B", MINI))

print("\n\n" + "=" * 104)
print(f"{'config':<34} {'p50 ms':>8} {'p95 ms':>8} {'overhead p50':>13} {'weak/strong':>13} {'accuracy':>10}")
print("-" * 104)
for r in results:
    if not r:
        continue
    lat = sorted(r["lat"])
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95) - 1]
    split = f"{r['served'].get('weak',0)}/{r['served'].get('strong',0)}" if r["served"] else "-"
    acc = f"{100*r['correct']/len(QUERIES):.0f}%" if r["correct"] is not None else "-"
    print(f"{r['label']:<34} {p50:>8.0f} {p95:>8.0f} {r['overhead_p50']:>13.0f} {split:>13} {acc:>10}")
print("=" * 104)
json.dump(results, open("/tmp/bench_results.json", "w"), indent=1)
print("\nsaved /tmp/bench_results.json")
