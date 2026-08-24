# Brev Launchable — build configuration

Settings to enter in the Brev console (**Launchables → Create Launchable**).
Everything not listed here can stay at its default.

---

## 1. Details

| Field | Value |
|---|---|
| Name | `NVIDIA Agent Toolkit — hands-on` |
| Description | Add NeMo Relay observability, runtime guardrails and Switchyard model routing to an existing LangGraph agent. No GPU required. |

## 2. Compute

| Field | Value |
|---|---|
| Instance | **`m8i.xlarge`** (4 vCPU, 16 GB, x86_64) — CPU only |
| Disk | **100 GB** (default is 10 GB and is too small) |
| GPU | none |

The workshop path is cloud-inference only, so no GPU is needed. This also keeps cost at
about **$0.25/hr per attendee**.

## 3. Software / runtime mode

**VM Mode** — "Use a VM with Python, CUDA, and Docker."

- **Install Jupyter on the host:** ✅ yes
- **Setup script:** paste the contents of `setup.sh` from this repo

## 4. Code

**Attach a git repository.**

| Field | Value |
|---|---|
| Repo URL | `https://github.com/saurabh-nvidia/agent-toolkit-workshop` |
| Branch | `main` |

The setup script expects the repo to be cloned into the home directory and will symlink
`notebook/agent_workshop.ipynb` to the Jupyter working directory.

## 5. Launch parameters

Add **one optional parameter**:

| Field | Value |
|---|---|
| Name | `NVIDIA_API_KEY` |
| Type | text |
| Required | ❌ **no** |
| Default | *(leave empty)* |
| Description | Optional. Your key from build.nvidia.com, starting with `nvapi-`. You can also paste it into the notebook after it opens. |

**Deliberately optional.** Someone without a key can still deploy, open the notebook and
read it — they are not locked out at the door. Section 2 of the notebook detects a missing
key and offers a cell to paste one into.

When supplied, Brev passes it to the setup script as an environment variable, and the
script injects it into the Jupyter kernel environment so `os.environ["NVIDIA_API_KEY"]`
works with no attendee action.

**Do not set a default value.** A shared key across 30 attendees risks rate limiting and
puts your credential on their machines.

## 6. Network

| Port | Expose | Why |
|---|---|---|
| Jupyter | ✅ (Brev handles this automatically in VM mode) | the notebook UI |
| 4000 | ❌ no | Switchyard is started by the notebook and only ever addressed on `localhost` |

---

## Repository layout

```
agent-toolkit-workshop/
├── README.md                  entry point for attendees
├── ATTENDEE_PREWORK.md        accounts + API key steps (send a week ahead)
├── LAUNCHABLE_CONFIG.md       this file
├── setup.sh                   Launchable setup script
├── notebook/
│   └── agent_workshop.ipynb  the workshop - edit this directly
├── benchmark/
│   ├── bench.py               routing benchmark (~35 min)
│   └── BENCHMARK_NOTES.md     results and method
└── optional/
    └── GPU_PATH.md            running against a local vLLM instead of the API
```

## Why a repo rather than an embedded file

- You can fix a bug the morning of the workshop without rebuilding the Launchable
- `bench.py` and the notes ship alongside, so the offline follow-up material is already there
- Attendees keep a copy they can clone and re-run afterwards

---

## Before you publish

1. Deploy the Launchable yourself on a brand-new VM
2. Confirm the notebook runs top to bottom with **no manual steps**
3. Time it from "Deploy" to "first cell runs" — that number sets your session timing
4. Deploy it a second time with a deliberately wrong API key and confirm the error message
   is understandable
