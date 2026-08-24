# Before the workshop — 10 minutes

Please complete these steps **before** the session. They cannot be done for you, and
without them you will not be able to run the hands-on material.

If any of it doesn't work, come to the workshop anyway — you will still be able to follow
along, and we can sort the key out there.

---

## 1. Create a Brev account (2 min)

Brev is where the workshop environment runs. You do not need to install anything.

1. Go to **https://brev.nvidia.com**
2. Sign in — you can use an existing NVIDIA account, Google, or GitHub
3. Verify your email if prompted
4. You should land on the Brev console showing an empty instance list

✅ **Done when:** you can see the Brev console.

---

## 2. Get an NVIDIA API key (5 min)

This is the key the notebook uses to call models. It is free and comes with credits.

1. Go to **https://build.nvidia.com**
2. Sign in with the same NVIDIA account
3. Open this model page:
   **https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b**
4. Click **Get API Key** (top right of the code panel), then **Generate Key**
5. Copy the key — it starts with `nvapi-`

⚠️ **Copy it now and save it somewhere.** The key is shown once and cannot be retrieved
later. If you lose it, generate a new one.

✅ **Done when:** you have a string starting with `nvapi-` saved somewhere you can paste
from during the session.

---

## 3. Check your key works (2 min)

The workshop uses **two** models. Your key should reach both. Run this from any terminal,
substituting your key:

```bash
export NVIDIA_API_KEY=nvapi-your-key-here

# should print 200
curl -s -o /dev/null -w "%{http_code}\n" \
  https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NVIDIA_API_KEY"
```

Then confirm both models respond:

```bash
for M in nvidia/nemotron-3.5-lightning-30b-a3b nvidia/nemotron-3-ultra-550b-a55b; do
  echo -n "$M -> "
  curl -s https://integrate.api.nvidia.com/v1/chat/completions \
    -H "Authorization: Bearer $NVIDIA_API_KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"say OK\"}],\"max_tokens\":5}" \
    | grep -o '"content":"[^"]*"' | head -1
done
```

✅ **Done when:** the first command prints `200` and both models return content.

**If you get 401:** the key is wrong or was copied incompletely.
**If you get 403 on one model:** tell us before the session — you may need model access
granted, and that takes time to resolve on the day.

---

## Checklist

- [ ] Brev account, can see the console
- [ ] `nvapi-` key saved somewhere I can paste from
- [ ] Key returns `200` and both models respond

## On the day

You will be given a Launchable link. It asks for your API key — you can paste it there, or
skip it and paste the key straight into the notebook once it opens. Either works.

**Everything else is automatic** — no installs, no terminal, no configuration.
