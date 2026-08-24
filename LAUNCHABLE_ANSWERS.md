# Brev Launchable — field-by-field answers

Fill in at https://brev.nvidia.com/launchables/create

> ⚠️ Two things first:
> 1. Push this repo to GitHub (must be **public**) and note the URL.
> 2. Edit line 16 of `setup.sh` — it still says `CHANGE-ME`.

---

## 1. Details

**Name**
```
NVIDIA Agent Toolkit — hands-on
```

**Description**
```
Add observability, runtime guardrails and model routing to an existing LangGraph agent
using NVIDIA open-source tooling. Nothing is rewritten — three levers layer on top.
Runs on CPU; no GPU required. ~60 minutes.
```

## 2. Compute

| Field | Value |
|---|---|
| GPU | None (CPU only) |
| Instance type | `m8i.xlarge` — 4 vCPU, 16 GB, x86_64 |
| **Disk** | **100 GB** ← default is 10 GB and will fail |
| Region | default |

## 3. Software / runtime

| Field | Value |
|---|---|
| Mode | **VM Mode** ("Use a VM with Python, CUDA, and Docker") |
| Install Jupyter | ✅ Yes |
| Setup script | paste the entire contents of `setup.sh` |

## 4. Code

| Field | Value |
|---|---|
| Source | Git repository |
| URL | `https://github.com/<your-org>/agent-toolkit-workshop` |
| Branch | `main` |

Brev requires a **public** repo: *"Provide a public repository, notebook, or Markdown file URL."*

## 5. Launch parameters

One parameter:

| Field | Value |
|---|---|
| Name | `NVIDIA_API_KEY` |
| Type | Text |
| Required | ❌ **No** |
| Default | *(empty)* |
| Description | `Optional. Your free API key from build.nvidia.com, starting with nvapi-. You can also paste it into section 2 of the notebook after it opens.` |

Optional on purpose: someone without a key can still deploy, open the notebook and read it.
Section 2 detects the missing key and offers a cell to paste one in.

**Do not set a default.** A shared key across 30 people rate-limits on the 550B model and
puts your credential on their machines.

## 6. Network

| Port | Expose |
|---|---|
| Jupyter | ✅ automatic in VM mode |
| 4000 | ❌ no — Switchyard is localhost-only, started by the notebook |

## 7. Access

Generate a **shareable link**. Each attendee deploys their own instance from it.

---

## Test before sharing

Deploy it twice yourself:

1. **With a key** — notebook runs top to bottom, no manual steps.
   Time from *Deploy* to *first cell runs* — that number sets your session timing.
2. **Without a key** — section 2 should print "No API key found" with usable
   instructions, not a traceback.

## On the day

- 30 attendees × `m8i.xlarge` ≈ **$7.50/hr** total
- Instances run until stopped — tell everyone to delete theirs at the end
