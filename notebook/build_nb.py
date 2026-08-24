"""Builds the the audience workshop notebook: NVIDIA Agent Toolkit levers on an existing agent."""
import json

C = []
def md(s):   C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})
def code(s): C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md("""
# Taking an existing agent further with NVIDIA open-source tooling

**The premise: nothing here replaces what you have built.**

You already run a LangGraph agent that answers questions over a detections database. This
notebook does not rewrite it, does not change your framework, and does not change how you
serve your model. It adds three independent capabilities on top:

| Lever | What it adds | What you change |
|---|---|---|
| **NeMo Relay** — visibility | full execution trajectory, latency attribution | 2 lines |
| **NeMo Relay** — control | policy enforced *at the runtime*, not in your tool code | 1 function + 1 registration |
| **NeMo Switchyard** — routing | cheap model for simple questions, big model for hard ones | 1 URL |

Every section shows the **before** and the **after**, so the difference is visible rather
than asserted.
""")

md("""
---
## 1. Is this machine ready?

Your environment was built by a setup script when the instance started. If you opened this
notebook quickly, that script may still be running.

**Run this cell first.** It waits for setup to finish and tells you what is happening.
""")

code('''
import os
import pathlib
import sys
import time

READY  = pathlib.Path.home() / ".workshop_ready"    # written when setup succeeds
STATUS = pathlib.Path.home() / ".workshop_status"   # current phase, for a useful message


def wait_for_setup(timeout_s: int = 900) -> None:
    """Block until the setup script signals it has finished."""
    if READY.exists():
        print("environment ready")
        return

    print("Setup is still running. This takes 2-5 minutes on a fresh machine.")
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        if READY.exists():
            print("\\nenvironment ready")
            return
        phase = STATUS.read_text().strip() if STATUS.exists() else "starting"
        if phase != last:
            print(f"  ... {phase}")
            last = phase
        if phase == "failed":
            raise RuntimeError("Setup failed. See ~/workshop_setup.log")
        time.sleep(5)
    raise TimeoutError("Setup did not finish in time. See ~/workshop_setup.log")


wait_for_setup()

# Confirm we are on the kernel the setup script built, not the host Python.
print(f"python {sys.version.split()[0]}")
if sys.version_info < (3, 12):
    print("\\nWARNING: wrong kernel. Select 'Agent Workshop (Python 3.12)' "
          "from the kernel menu, then re-run this cell.")
''')

md("""
---
## 2. Your API key

This notebook calls NVIDIA's hosted models, so it needs an API key. There is no GPU
involved and nothing to install.

**If you supplied a key when you deployed this Launchable**, it is already in place — run
the cell below and it will confirm that.

**If you did not**, paste your key into the cell below and run it. You can get one free at
**https://build.nvidia.com** — open any model page, click *Get API Key*, and copy the value
starting with `nvapi-`. It takes about two minutes.
""")

code('''
# ---------------------------------------------------------------------------
# Paste your key between the quotes ONLY if you did not supply one when the
# Launchable was deployed. Otherwise leave this empty and just run the cell.
# ---------------------------------------------------------------------------
MY_API_KEY = ""      # e.g. "nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"


if MY_API_KEY.strip():
    os.environ["NVIDIA_API_KEY"] = MY_API_KEY.strip()
    print("key set from this cell")
elif os.environ.get("NVIDIA_API_KEY"):
    print("key already present from the Launchable")
else:
    print(
        "No API key found.\\n\\n"
        "Get one free at https://build.nvidia.com - open any model page,\\n"
        "click 'Get API Key', then paste it into MY_API_KEY above and re-run\\n"
        "this cell. It starts with 'nvapi-'.\\n\\n"
        "You can read the whole notebook without a key; you just cannot run\\n"
        "the cells that call a model."
    )
''')

md("""
---
## 3. Setup

There is also a **local GPU path** — the same notebook against a vLLM server you run
yourself. It is entirely optional and covered in `optional/GPU_PATH.md`; switching to it is
a one-line change to `LLM_MODE`, and nothing else in the notebook changes. That is worth
noticing in itself: the agent, tools, Relay wiring and guardrails are all model-agnostic.
""")

code('''
# "cloud" = NVIDIA's hosted API. No GPU needed. This is the workshop default.
# "local" = a vLLM server you run yourself - see optional/GPU_PATH.md.
LLM_MODE = "cloud"

if LLM_MODE == "cloud":
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODEL    = "nvidia/nemotron-3.5-lightning-30b-a3b"  # 30B MoE, 3B active
    API_KEY  = os.environ.get("NVIDIA_API_KEY")         # from the environment, never in code

    if not API_KEY:
        raise RuntimeError(
            "No API key.\\n\\n"
            "Scroll up to section 2, paste your key into MY_API_KEY, run that cell,\\n"
            "then run this one again.\\n\\n"
            "Get a free key at https://build.nvidia.com"
        )
else:
    BASE_URL = "http://localhost:8000/v1"
    MODEL    = "qwen3-coder-30b"                        # served by your own vLLM
    API_KEY  = "not-needed-for-local-vllm"              # vLLM ignores it; the client requires it

print(f"mode={LLM_MODE}  model={MODEL}")
''')

code('''
import sqlite3
import textwrap
import time
from datetime import datetime

import nemo_relay
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

print("imports ok")
''')

md("""
---
## 4. The database

One row per object seen by one camera at one moment — what a video pipeline writes after
detection, tracking and attribute classification.

Note there are **two** tables. `detections` belongs to the case we are investigating.
`case_9931_detections` belongs to a *different* case and must never be readable from this
session. That second table matters in section 5.
""")

code('''
DB = "/tmp/workshop.db"

def build_database() -> None:
    """Create two case tables and fill them with deterministic demo rows."""
    conn = sqlite3.connect(DB)
    conn.execute("DROP TABLE IF EXISTS detections")
    conn.execute("DROP TABLE IF EXISTS case_9931_detections")

    conn.execute("""
        CREATE TABLE detections (
            id           INTEGER PRIMARY KEY,
            ts           TEXT,    -- when it was seen
            gate         TEXT,    -- which camera
            object_type  TEXT,    -- 'person' or 'vehicle'
            plate        TEXT,    -- licence plate, vehicles only
            colour       TEXT
        )
    """)
    gates   = ["Gate-A", "Gate-3", "Gate-C", "Bay-4"]
    colours = ["red", "blue", "black", "white"]
    rows = []
    for i in range(200):
        is_vehicle = (i % 2 == 1)                     # exactly half are vehicles
        rows.append((
            i,
            f"2026-08-06T{6 + i // 30:02d}:{i % 60:02d}:00",
            gates[i % 4],
            "vehicle" if is_vehicle else "person",
            f"PL-{1000 + i % 57}" if is_vehicle else None,
            colours[i % 4],
        ))
    conn.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?)", rows)

    # A different investigation's data. Same shape, must stay unreachable.
    conn.execute("CREATE TABLE case_9931_detections (id INTEGER, gate TEXT, object_type TEXT)")
    conn.executemany("INSERT INTO case_9931_detections VALUES (?,?,?)",
                     [(i, "Gate-X", "person") for i in range(7)])
    conn.commit()
    conn.close()

build_database()

conn = sqlite3.connect(DB)
print("detections rows :", conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0])
print("of which vehicles:", conn.execute("SELECT COUNT(*) FROM detections WHERE object_type='vehicle'").fetchone()[0])
conn.close()
''')

md("""
---
## 5. The agent you already have

Three tools and a ReAct loop — the model reasons, picks a tool, sees the result, repeats.
The `@tool` decorator exposes each function's name, arguments and docstring to the model;
the docstring is the specification the model reads, not a comment.

**There is nothing about Relay, Switchyard or NVIDIA tooling in this section.** This is the
starting point.
""")

code('''
@tool
def list_tables() -> str:
    """List the tables available in the detections database."""
    c = sqlite3.connect(DB)
    names = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    c.close()
    return ", ".join(names)


@tool
def get_schema(table: str) -> str:
    """Return the column names and types for one table."""
    c = sqlite3.connect(DB)
    cols = c.execute(f"PRAGMA table_info({table})").fetchall()
    c.close()
    return "\\n".join(f"{col[1]} ({col[2]})" for col in cols)


@tool
def run_sql(query: str) -> str:
    """Run a read-only SQL query against the detections database and return the rows."""
    c = sqlite3.connect(DB)
    try:
        rows = c.execute(query).fetchall()
    except Exception as exc:
        return f"SQL error: {exc}"          # hand errors back so the model can self-correct
    finally:
        c.close()
    return "\\n".join(str(r) for r in rows[:30]) if rows else "no rows"


TOOLS = [list_tables, get_schema, run_sql]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an investigation assistant for a video analytics platform.
    Answer questions about the detections database.
    Always use run_sql for factual claims. Never invent numbers.
    Answer in one short sentence.
""").strip()

model = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL, temperature=0)

print("tools:", [t.name for t in TOOLS])
''')

md("""
---
## 6. Lever 1 — NeMo Relay for visibility

### BEFORE: run the agent as it exists today
""")

code('''
plain_agent = create_agent(model, TOOLS, system_prompt=SYSTEM_PROMPT)

def ask_plain(question: str) -> str:
    result = plain_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content

started = time.time()
answer = ask_plain("How many vehicles are in the detections table?")
elapsed = time.time() - started

print(f"ANSWER: {answer}")
print(f"took {elapsed:.1f}s")
''')

md("""
### The answer is probably right. Can you prove it?

Scroll back to section 2: there are 200 rows, of which exactly 100 are vehicles. So the
number matches.

But you cannot tell *why* it matches. Did the agent run `WHERE object_type='vehicle'`, or
did it run `COUNT(*)` against a table that happens to be half vehicles and get lucky? How
many times did it call the model? Was the time spent in your SQL or in inference?

You cannot answer any of those from here. The agent returned a string and discarded
everything else — so a correct answer and a lucky answer look identical. On a harder
question it will be wrong the same way: confidently, and unverifiably.

That is the gap. Now we close it.
""")

md("""
### AFTER: add Relay

Two additions, and **the agent definition above is untouched**:

1. `subscribers.register(...)` — a callback that receives every runtime event
2. `NemoRelayMiddleware()` — passed to `create_agent`

The middleware matters. Relay ships two integrations: a *callback handler* that observes
LangChain callbacks, and this *middleware* that routes model and tool calls through Relay's
own execution path. Only the middleware produces real `tool` / `llm` scopes and lets the
policy layer in section 5 intercept anything.
""")

code('''
from nemo_relay.integrations.langgraph import NemoRelayMiddleware

# Collect every event Relay emits into a plain Python list.
events: list[dict] = []
nemo_relay.subscribers.register("collector", lambda e: events.append(e.to_dict()))


def flush_events() -> None:
    """Wait for Relay's async event delivery to drain before we read `events`.

    flush() refuses to block when an asyncio loop is already running - which happens if the
    agent call raised from inside async code. The events still arrive; we just cannot wait.
    """
    try:
        nemo_relay.subscribers.flush()
    except RuntimeError:
        pass


# Same model, same tools, same prompt - one extra argument.
agent = create_agent(model, TOOLS, system_prompt=SYSTEM_PROMPT,
                     middleware=[NemoRelayMiddleware()])

def ask(question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content

events.clear()
started = time.time()
answer = ask("How many vehicles are in the detections table?")
wall_ms = (time.time() - started) * 1000
flush_events()

print(f"ANSWER: {answer}")
print(f"\\nRelay captured {len(events)} events in {wall_ms:.0f}ms")
''')

md("""
### What Relay captured

Relay emits events in a standard format (ATOF). Two kinds matter here:

- `kind="scope"` — a unit of work with a start and an end (the agent, a tool call, a model
  call). Each has a `uuid` and a `parent_uuid`, so the events form a tree.
- `kind="mark"` — a point-in-time note inside a scope.
""")

code('''
def parse_ts(ts: str) -> datetime:
    """Parse an ATOF timestamp.

    Relay emits nanosecond precision (9 digits); Python accepts at most 6, so trim.
    """
    if "." in ts:
        head, frac = ts.split(".", 1)
        digits = "".join(ch for ch in frac if ch.isdigit())[:6]
        tail = frac[len(digits):].lstrip("0123456789") or "+00:00"
        ts = f"{head}.{digits}{tail}"
    return datetime.fromisoformat(ts)


def show_trajectory(events: list[dict]) -> None:
    """Print the scopes in the order they executed.

    Scopes nest via parent_uuid, so this indents children under parents. In a flat
    agent loop most scopes are siblings, which is why the output reads as a sequence:
    model call, tool call, model call, and so on.
    """
    starts = [e for e in events if e["kind"] == "scope" and e["scope_category"] == "start"]
    children: dict = {}
    for e in starts:
        children.setdefault(e["parent_uuid"], []).append(e)
    known = {e["uuid"] for e in starts}

    def walk(node, depth):
        print("   " * depth + f"|- [{node['category']}] {node['name']}")
        for kid in children.get(node["uuid"], []):
            walk(kid, depth + 1)

    for root in [e for e in starts if e["parent_uuid"] not in known]:
        walk(root, 0)

show_trajectory(events)
''')

md("""
### Where did the time actually go?

Each scope has a start and an end event. Subtracting the timestamps gives a duration;
grouping by category gives latency attribution.

We report `llm` and `tool` against measured wall-clock. Those two are the leaves of the
tree — the `agent` scope simply wraps them, so including it would double-count.
""")

code('''
def show_costs(events: list[dict], wall_ms: float) -> None:
    """Pair scope starts with ends and report time spent in models vs your own code."""
    open_scopes, done = {}, []
    for e in events:
        if e["kind"] != "scope":
            continue
        if e["scope_category"] == "start":
            open_scopes[e["uuid"]] = e
        elif e["uuid"] in open_scopes:
            begin = open_scopes.pop(e["uuid"])
            ms = (parse_ts(e["timestamp"]) - parse_ts(begin["timestamp"])).total_seconds() * 1000
            done.append((begin["category"] or "other", begin["name"], ms))

    totals: dict = {}
    for category, name, ms in done:
        entry = totals.setdefault((category, name), [0, 0.0])
        entry[0] += 1
        entry[1] += ms

    print(f"{'category':<10} {'name':<34} {'calls':>5} {'ms':>9} {'% wall':>8}")
    print("-" * 70)
    for (category, name), (calls, ms) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
        pct = f"{100 * ms / wall_ms:7.1f}%" if category in ("llm", "tool") else "       -"
        print(f"{str(category):<10} {name[:34]:<34} {calls:>5} {ms:>9.1f} {pct}")
    print(f"\\nmeasured wall clock: {wall_ms:.0f}ms")

show_costs(events, wall_ms)
''')

md("""
**How to read this.** If `llm` dominates, tuning SQL is wasted effort — the fix is fewer
model round-trips (a tighter prompt, or caching the schema so the agent stops rediscovering
it). If a tool appears more often than you expected, that is a design bug you could not
previously see.

None of this required changing the agent.
""")

md("""
---
## 7. Lever 2 — NeMo Relay for control

Visibility is half of it. Relay also enforces policy **before** a call executes.

### BEFORE: the agent can read any table it can see
""")

code('''
events.clear()
answer = ask("How many rows are in the case_9931_detections table?")
flush_events()
print(f"ANSWER: {answer}")
''')

md("""
That table belongs to a **different investigation**. The agent read it because nothing
stopped it. Your tool code was perfectly correct — it ran a valid read-only `SELECT`. The
problem is that "which tables may this session touch" is a *policy* question, and policy
lived nowhere.

### AFTER: register the policy with the runtime

`register_tool_conditional_execution` takes a function receiving `(tool_name, args)`:

- return `None` → allow the call
- return a **string** → block it

This is enforced by the runtime, not inside your tool. The model cannot rewrite its query
to get around it, and the same policy applies to every agent registered with this runtime.
""")

code('''
# Only these tables may be queried in this session.
ALLOWED_TABLES = {"detections"}
FORBIDDEN = {"case_9931_detections"}

def sql_guardrail(tool_name: str, args) -> str | None:
    """Reject SQL that leaves this case's scope or tries to write.

    Returning None allows the call; returning a string blocks it.
    """
    if tool_name != "run_sql":
        return None                                    # not our concern

    query = (args.get("query", "") if isinstance(args, dict) else str(args)).lower()

    for table in FORBIDDEN:                            # case scoping
        if table in query:
            return f"BLOCKED: '{table}' is outside this case's scope."

    if not query.strip().startswith("select"):         # reads only
        return "BLOCKED: only SELECT statements are permitted."

    for verb in ("insert", "update", "delete", "drop", "alter", "attach", "pragma"):
        if verb in query:
            return f"BLOCKED: '{verb}' is not permitted."

    return None


nemo_relay.guardrails.register_tool_conditional_execution(
    "sql-guard",   # name, so it can be replaced or removed later
    100,           # priority - lower runs first
    sql_guardrail,
)
print("guardrail registered")
''')

code('''
events.clear()
try:
    answer = ask("How many rows are in the case_9931_detections table?")
    print(f"ANSWER: {answer}")
except Exception as exc:
    # A blocked tool call raises and halts the run. That IS the guardrail working:
    # the call never reached your tool, so the data was never read.
    print(f"RUN HALTED BY THE RUNTIME:\\n  {type(exc).__name__}: {str(exc)[:170]}")
finally:
    flush_events()

# Prove the model still produced valid SQL - it was the runtime that refused.
show_trajectory(events)
''')

md("""
Two things worth noticing.

**The block is hard, not advisory.** The run stops. The model wrote perfectly valid SQL and
never got to execute it. Compare that with asking a model nicely in a system prompt not to
touch certain tables.

**The guardrail is itself a scope.** Look at the trajectory above — `[guardrail] sql-guard`
appears in the tree. So the record shows not just *that* a call was refused but *which
policy refused it*, which is what an auditable access log actually needs.

A hand-built version of this is a multi-stage SQL validation pipeline living inside the
agent. Here it is one function, one registration, enforced by the runtime and reusable
across every agent you run.
""")

md("""
---
## 8. Lever 3 — NeMo Switchyard: what does this actually cost you?

Investigation questions are not equally hard. "All plates at Gate-3 yesterday" is a single
filter. "Plates seen at Gate-A and Gate-C within 10 minutes, then the people in those
vehicles" is multi-hop reasoning.

Today both go to the same model, so you pay frontier-model prices for trivial lookups.
Switchyard is a **proxy**: it classifies each request and sends it to the right tier.

The interesting question is not "does it route" — it does. It is **what routing costs you
and what it saves you**, because the classifier is itself a model call on every single
request. This section measures that rather than asserting it.

### The routing profile
""")

code('''
ROUTES_YAML = """
defaults:
  base_url: https://integrate.api.nvidia.com/v1
  api_key: ${NVIDIA_API_KEY}
  format: openai            # must be 'openai' - the docs' 'openai_chat' is rejected

routes:
  tiered-router:
    type: deterministic     # classify first, then serve
    display_name: "Tiered investigation router"
    enable_stats: true
    fallback_target_on_evict: weak
    weak:
      model: nvidia/nemotron-3.5-lightning-30b-a3b   # 30B MoE, 3B active
    strong:
      model: nvidia/nemotron-3-ultra-550b-a55b       # 550B MoE, 55B active
    classifier:
      model: nvidia/nemotron-3.5-lightning-30b-a3b   # the judge - change this freely
"""

with open("/tmp/routes.yaml", "w") as fh:
    fh.write(ROUTES_YAML)

print(ROUTES_YAML)
''')

md("""
### Start the proxy

The next cell launches Switchyard for you and waits until it accepts requests, so you
do not need a second terminal.

Install note - the published package is missing two runtime dependencies (`pyyaml` and
`uvicorn`), so the documented install command does not work. Use this instead:

```bash
uv tool install --force --with pyyaml --with uvicorn --with fastapi \\
  --python 3.12 'nemo-switchyard[cli]'
```
""")

code('''
import shutil
import subprocess
import urllib.request

SWITCHYARD = shutil.which("switchyard") or os.path.expanduser("~/.local/bin/switchyard")


def switchyard_up() -> bool:
    """Return True once the proxy is accepting requests."""
    try:
        urllib.request.urlopen("http://localhost:4000/v1/models", timeout=3)
        return True
    except Exception:
        return False


def start_switchyard(config="/tmp/routes.yaml", port=4000, wait_s=120):
    """Launch the proxy as a child process and block until it is ready.

    The child inherits this process environment, which is how the ${NVIDIA_API_KEY}
    reference inside the routing profile gets resolved.
    """
    if switchyard_up():
        print("already running")
        return None

    log = open("/tmp/switchyard.log", "w")
    proc = subprocess.Popen([SWITCHYARD, "serve", "-c", config, "-p", str(port)],
                            stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if switchyard_up():
            print(f"switchyard ready on port {port}")
            return proc
        if proc.poll() is not None:                       # it died - surface why
            tail = open("/tmp/switchyard.log").read()[-500:]
            raise RuntimeError(f"switchyard exited:\\n{tail}")
        time.sleep(2)
    raise TimeoutError("switchyard did not become ready in time")


switchyard_proc = start_switchyard()
''')

md("""
### BEFORE and AFTER

The only change is `base_url` and the model name. No SDK, no code change, no framework
coupling — which is why this works with any agent, not just this one.
""")

code('''
L1 = "List all plates captured at Gate-3."
L4 = ("Find plates seen at Gate-A and Gate-C within 10 minutes of each other, then "
      "correlate them with the people detected in those vehicles, explaining each step.")

routed = ChatOpenAI(
    base_url="http://localhost:4000/v1",   # <- the proxy, not the model endpoint
    api_key=API_KEY,
    model="tiered-router",               # <- the route, not a model
    temperature=0,
)

for label, q in (("L1 (simple)", L1), ("L4 (multi-hop)", L4)):
    started = time.time()
    routed.invoke(q)
    print(f"{label:16} -> {time.time() - started:5.1f}s")
''')

code('''
# Which tier actually served each request?
import json as _json

with urllib.request.urlopen("http://localhost:4000/v1/stats", timeout=10) as resp:
    stats = _json.load(resp)

print(f"{'tier':<8} {'model':<42} {'calls':>5} {'req %':>7}")
print("-" * 66)
for tier, row in stats.get("tiers", {}).items():
    print(f"{tier:<8} {row['model']:<42} {row['calls']:>5} {row['request_pct']:>6.0f}%")

overhead = stats.get("routing_overhead", {})
print(f"\\nrouting overhead: avg {overhead.get('avg_ms', 0):.0f}ms "
      f"over {overhead.get('count', 0)} requests")
print("That is the judge model call itself - not free. See BENCHMARK_NOTES.md.")
''')

md("""
The simple question went to the 30B model; the multi-hop one went to the 550B. **The calling
code did not change** — only the URL.

### Do not trust that overhead number

It came from **two requests**. Across four repeated runs on identical configuration the same
measurement returned 1820 ms, 2174 ms, 3378 ms and 5796 ms — a **3.2x spread**. The number
printed above, which you just generated, is noise. So is any other two-request figure.

So we measured it properly: 20 questions spanning the L1–L4 difficulty tiers, run through
four configurations.

| config | p50 ms | p95 ms | overhead p50 | weak/strong | accuracy |
|---|---|---|---|---|---|
| always STRONG (550B) | 2119 | 23695 | 0 | – | – |
| always WEAK (30B) | 1244 | 3994 | 0 | – | – |
| routed, judge = Lightning 30B | 3359 | 26762 | **2113** | 6/11 | 94% (*) |
| routed, judge = Nemotron Mini 4B | 2125 | 27994 | **122** | **0/18** | 50% |

(*) of decisions that completed — see the caveat below.

**Three things this tells you, none of which the marketing does.**

**1. A cheap classifier destroys the entire benefit.** Swapping the judge to a 4B model cut
overhead 17x — and routed **0 of 18** requests to the cheap tier. It sent everything to the
550B. That is not routing, it is a passthrough with extra latency. The judge has to be
capable enough to recognise an easy question.

**2. Routing makes you slower, not faster.** Routed p50 is 3359 ms against 2119 ms for
always-STRONG, because every request pays a classifier call before anything else happens.
Always-WEAK is fastest at 1244 ms. **The win is cost, not speed.** If someone sells you
routing as a latency improvement, they have not measured it.

**3. With a capable judge the routing itself is accurate.** Per tier: L1 5/5 correct,
L3 5/5, L4 5/5, L2 1/5. That is 16 of 17 completed decisions right, and the single genuine
misroute over-escalated an L2 to the strong tier — the safe direction.

**The caveat:** 3 of 20 requests failed outright (all returned in under a second, i.e. API
errors rather than routing decisions). Counting those as misses is what drags the headline
figure to 80%. Rerun with retries before quoting any of these numbers externally.

### So does it pay for you?

That depends entirely on your query mix. If most of your traffic is L1/L2 lookups, routing
those to a 30B instead of a 550B is a large saving and the ~2 s overhead is worth it. If
most of your traffic is genuinely hard, you pay the classifier tax on every request and
route almost everything to the strong tier anyway.

**Measure it against your own questions.** The next cell is the benchmark that produced the
table above — it is commented out because it takes roughly 35 minutes. Replace `QUERIES`
with your own labelled questions and run it offline.
""")

code('''
# ---------------------------------------------------------------------------
# Routing benchmark - COMMENTED OUT ON PURPOSE. Runtime is roughly 35 minutes.
#
# Full script: bench.py   Full write-up: BENCHMARK_NOTES.md
#
# To run it against your own workload, replace QUERIES with your questions and the
# tier you expect each to route to, then uncomment.
# ---------------------------------------------------------------------------

# QUERIES = [
#     ("List all plates captured at Gate-3.",                      "weak",   "L1"),
#     ("Count unique people per hour per gate last week.",          "weak",   "L2"),
#     ("Show images of red vehicles near Gate-3 and explain each.", "strong", "L3"),
#     ("Correlate plates across gates, then the people in them.",   "strong", "L4"),
#     # ... 20 queries total in bench.py
# ]
#
# For each configuration the benchmark:
#   1. rewrites the routing profile and restarts the proxy
#   2. sends every query through it
#   3. diffs /v1/stats tier counters before and after each request to learn which
#      tier actually served it - that is what makes routing ACCURACY measurable
#   4. reports p50 / p95 latency, routing overhead, tier split and accuracy
#
# The accuracy column is the one that matters. A cheap router that misroutes hard
# questions to the small model saves money and returns worse answers - which is
# strictly worse than not routing at all.

print("See bench.py to run this offline (~35 min).")
''')

md("""
---
## 9. What this adds up to

| Capability | Cost to adopt |
|---|---|
| Full execution trajectory of an existing agent | 2 lines |
| Latency attribution: models vs your own code | read the events |
| Case-scoped access policy, enforced at the runtime | 1 function + 1 registration |
| Auditable record of which policy refused a call | free with the above |
| Cheap/expensive model routing | 1 URL |

**Nothing about the agent changed.** Same LangGraph loop, same tools, same prompt, same
model, same serving stack. Each lever is independent — adopt one, or none, or all three.

### Cleaning up
""")

code('''
nemo_relay.subscribers.deregister("collector")
nemo_relay.guardrails.deregister_tool_conditional_execution("sql-guard")

if switchyard_proc is not None:
    switchyard_proc.terminate()
    switchyard_proc.wait(timeout=20)
    print("switchyard stopped")

print("cleaned up")
''')

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Agent Workshop (Python 3.12)", "language": "python", "name": "relayenv"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

out = "agent_workshop.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out}: {len(C)} cells")
