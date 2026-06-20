# llm-provider-cascade

A tiny, dependency-light helper that makes LLM calls **never hard-fail**. It tries
providers in order and falls through on rate-limits/outages — ending at a **local
Ollama** model so you always get an answer, even when every cloud provider is down.

```
Groq (fast/cheap)  →  Anthropic (best)  →  Gemini  →  Ollama (local, always up)
```

I built this because I run 80+ agents that make thousands of LLM calls a day — when
one provider 429s at 3am, the whole system can't go dark.

## Usage
```python
from cascade import llm

answer = llm("Summarize this support ticket in one sentence:\n" + ticket)
# Set any of: GROQ_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
# Falls back to a local Ollama model (default: qwen2.5:14b) if all else fails.
```

## Why it's nice
- **Zero hard dependencies** — just `requests`.
- **Graceful degradation** — quality drops before availability does.
- **One env-var per provider** — keys never touch the code.
- **Drop-in** — one function, returns a string.

MIT licensed.
