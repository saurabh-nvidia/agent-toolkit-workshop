# Optional — running against your own GPU

The workshop runs against NVIDIA's hosted API so it needs no GPU. This document covers the
same notebook against a **local vLLM server**, which is what you would do on-premise.

**You do not need this for the workshop.** It is here for afterwards.

---

## Why bother

The point is that nothing except the endpoint changes. Same agent, same tools, same Relay
wiring, same guardrails — only `LLM_MODE`. If that holds on your own hardware with your own
model, it holds in your production stack too.

## What you need

- A GPU with **≥ 45 GB** of VRAM (an L40S 48 GB is enough, and is what this was tested on)
- Docker with the NVIDIA container runtime
- **Driver 570+.** Current vLLM images need CUDA 12.8; a 565 driver fails with
  *"NVIDIA driver on your system is too old"*. This is the single most common failure.
- ~35 GB of disk for the model weights

## Start the server

```bash
docker run --rm --name vllm --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/vllm:/root/.cache/vllm \
  -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3-coder-30b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Two flags matter more than they look:

- **`--enable-auto-tool-choice` and `--tool-call-parser`.** Without them every agent request
  fails with *"auto tool choice requires --enable-auto-tool-choice and --tool-call-parser to
  be set"*. An agent cannot call a tool without them.
- **`--tool-call-parser qwen3_coder`** must match the model's output format. Getting this
  wrong fails *silently*: the server returns `tool_calls: None` and the tool call stays
  trapped in the message text, so the agent simply never calls anything.

The `~/.cache/vllm` mount preserves `torch.compile` output. First start takes several
minutes; later ones take about 40 seconds.

## Point the notebook at it

```python
LLM_MODE = "local"
```

That is the whole change.

## Wait for readiness

```bash
curl -s http://localhost:8000/v1/models
```

Only proceed once this returns the model. vLLM accepts connections before it can serve.

---

## Model / parser compatibility

| Model | Parser | Works? |
|---|---|---|
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | `qwen3_coder` | ✅ verified |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8` | `hermes` | ❌ emits `<TOOLCALL>[...]`, which `hermes` cannot parse |

Nemotron Nano needs a parser plugin that vLLM does not bundle. It works fine through the
hosted API, where NVIDIA runs the parser server-side — which is a good illustration that
"the model supports tool calling" and "your serving stack can parse its tool calls" are two
different questions.

## Sizing note

A 30B FP8 model is roughly 31 GB of weights, leaving ~14 GB for KV cache on a 48 GB card at
`--gpu-memory-utilization 0.90`. If you want a longer context, either reduce
`--max-model-len` or use a larger card.
