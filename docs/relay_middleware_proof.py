"""Prove: (1) tool+llm scopes appear, (2) guardrails actually BLOCK - deterministically."""
import os, sqlite3
import nemo_relay
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from nemo_relay.integrations.langgraph import NemoRelayMiddleware

DB = "/tmp/det.db"
conn = sqlite3.connect(DB)
conn.execute("DROP TABLE IF EXISTS detections")
conn.execute("DROP TABLE IF EXISTS case_9931_detections")
conn.execute("CREATE TABLE detections (id INT, gate TEXT, object_type TEXT)")
conn.executemany("INSERT INTO detections VALUES (?,?,?)",
                 [(i, f"Gate-{i%3}", "vehicle" if i % 2 else "person") for i in range(50)])
# A table belonging to a DIFFERENT case - must never be readable from this session.
conn.execute("CREATE TABLE case_9931_detections (id INT, gate TEXT, object_type TEXT)")
conn.executemany("INSERT INTO case_9931_detections VALUES (?,?,?)", [(i, "Gate-X", "person") for i in range(7)])
conn.commit(); conn.close()

@tool
def list_tables() -> str:
    """List tables in the detections database."""
    c = sqlite3.connect(DB)
    r = ", ".join(x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    c.close(); return r

@tool
def run_sql(query: str) -> str:
    """Run a read-only SQL query and return rows."""
    c = sqlite3.connect(DB)
    try:
        return "\n".join(str(r) for r in c.execute(query).fetchall()[:20]) or "no rows"
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        c.close()

# --- The guardrail: mirrors a case-scoped table allowlist ---
ALLOWED_TABLES = {"detections"}
blocked = []

def sql_guardrail(tool_name, args):
    if tool_name != "run_sql":
        return None
    q = (args.get("query", "") if isinstance(args, dict) else str(args)).lower()
    for t in ("case_9931_detections",):
        if t in q:
            blocked.append(q)
            return f"BLOCKED: '{t}' is outside this case's scope."
    if not q.strip().startswith("select"):
        blocked.append(q)
        return "BLOCKED: only SELECT statements are permitted."
    return None

nemo_relay.guardrails.register_tool_conditional_execution("sql-guard", 100, sql_guardrail)

events = []
nemo_relay.subscribers.register("c", lambda e: events.append(e.to_dict()))

model = ChatOpenAI(base_url="https://integrate.api.nvidia.com/v1",
                   api_key=os.environ["NVIDIA_API_KEY"],
                   model="nvidia/nemotron-3.5-lightning-30b-a3b", temperature=0)
agent = create_agent(model, [list_tables, run_sql],
                     system_prompt="You answer questions about a video detections database. Always use run_sql for facts.",
                     middleware=[NemoRelayMiddleware()])

def ask(q):
    return agent.invoke({"messages": [{"role": "user", "content": q}]})["messages"][-1].content

print("=== Q1: allowed query ===")
print(ask("How many vehicles are in the detections table?")[:200])

print("\n=== Q2: query that needs an out-of-scope table ===")
try:
    print(ask("How many rows are in the case_9931_detections table?")[:250])
except Exception as exc:
    print(f"RUN HALTED BY RUNTIME: {type(exc).__name__}: {str(exc)[:160]}")

nemo_relay.subscribers.flush()

print("\n=== BLOCKED BY GUARDRAIL ===")
for b in blocked: print("  ", b[:110])

cats = {}
for e in events:
    if e["kind"] == "scope" and e["scope_category"] == "start":
        cats.setdefault(e["category"], set()).add(e["name"])
print("\n=== SCOPE CATEGORIES ===")
for k, v in sorted(cats.items(), key=lambda x: str(x[0])):
    print(f"  {k}: {sorted(v)[:6]}")
