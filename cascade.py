"""Multi-provider LLM cascade: Groq -> Anthropic -> Gemini -> local Ollama.

Keys are read from the environment; nothing is hard-coded. Each provider is tried
in order and we fall through on any error or empty response, so a single function
call degrades in quality before it ever degrades in availability.
"""
import os
import requests

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")


def _groq(prompt, max_tokens):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={"model": "llama-3.3-70b-versatile", "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _anthropic(prompt, max_tokens):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
        json={"model": "claude-sonnet-4-6", "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    r.raise_for_status()
    for block in r.json().get("content", []):
        if block.get("type") == "text":
            return block["text"].strip()
    return ""


def _gemini(prompt, max_tokens):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={GEMINI_KEY}",
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _ollama(prompt, max_tokens):
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
              "options": {"num_predict": max_tokens}},
        timeout=200,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def llm(prompt: str, max_tokens: int = 600) -> str:
    """Return an LLM completion, trying providers in order of speed/cost/quality.

    Falls through on any error or short/empty response. The local Ollama step has
    no rate limit, so this only returns "" if Ollama is also unreachable.
    """
    chain = []
    if GROQ_KEY:
        chain.append(_groq)
    if ANTHROPIC_KEY:
        chain.append(_anthropic)
    if GEMINI_KEY:
        chain.append(_gemini)
    chain.append(_ollama)  # always-available local fallback

    for provider in chain:
        try:
            out = provider(prompt, max_tokens)
            if out and len(out) > 1:
                return out
        except Exception:
            continue
    return ""


if __name__ == "__main__":
    print(llm("Say hello in one short sentence."))
