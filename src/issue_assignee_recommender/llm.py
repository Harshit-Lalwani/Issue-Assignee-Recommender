"""Phase 5: multi-provider LLM client with an explicit fallback chain.

Four providers are configured via .env (see .env.example): Groq, OpenRouter, Gemini, NVIDIA
NIM. Three of the four (Groq, OpenRouter, NVIDIA NIM) speak the OpenAI-compatible chat
completions format; Gemini uses Google's own generateContent format.

Provider choice and model choice per provider are recorded in docs/decisions.md along with the
benchmark that justified them -- don't change the defaults here without updating that doc.
"""
import re
import time
from pathlib import Path

import requests

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_env() -> dict:
    """.env uses `KEY = value` (spaces around `=`), which plain `source` can't parse -- read
    it directly instead of shelling out to a dotenv loader. Also strips trailing `# comment`
    annotations after the value (e.g. `HF_TOKEN = hf_xxx # Full access`) -- a naive strip()
    would otherwise fold the comment text into the secret itself."""
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip()
            if " #" in v:
                v = v.split(" #", 1)[0].strip()
            env[k.strip()] = v
    return env


_ENV = load_env()

TIMEOUT_S = 30
# NVIDIA NIM sometimes cold-starts a model on first call and can take >30s; it's the last
# fallback in the chain so a longer timeout there costs nothing in the common case.
PROVIDER_TIMEOUT_S = {"nvidia_nim": 60}

# Provider -> (base_url, api_key_env, header_style). Fallback order is Groq -> OpenRouter ->
# Gemini -> NVIDIA NIM, encoded as a plain list consumers iterate over.
PROVIDER_CHAIN = ["groq", "openrouter", "gemini", "nvidia_nim"]

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "nvidia/nemotron-3-super-120b-a12b:free",
    "gemini": "gemini-3.1-flash-lite",
    "nvidia_nim": "meta/llama-3.3-70b-instruct",
}


class LLMError(Exception):
    pass


def _openai_compatible_call(
    base_url, api_key, model, system, user, max_tokens=600, timeout=None, max_retries=3
):
    last_err = None
    for attempt in range(max_retries):
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
            timeout=timeout or TIMEOUT_S,
        )
        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise LLMError(f"{base_url} {model}: unexpected response shape: {data}") from e
        if resp.status_code == 429:
            # respect the server's requested backoff when it tells us one (Groq's free tier
            # TPM limits are tight enough that this fires constantly under any real load)
            wait_s = 2.0 * (attempt + 1)
            match = re.search(r"try again in ([\d.]+)s", resp.text)
            if match:
                wait_s = float(match.group(1)) + 0.2
            last_err = LLMError(f"{base_url} {model}: HTTP 429: {resp.text[:200]}")
            time.sleep(wait_s)
            continue
        raise LLMError(f"{base_url} {model}: HTTP {resp.status_code}: {resp.text[:300]}")
    raise last_err


def call_groq(system, user, model=None):
    model = model or DEFAULT_MODELS["groq"]
    return _openai_compatible_call(
        "https://api.groq.com/openai/v1", _ENV["GROQ_API_KEY"], model, system, user
    )


def call_openrouter(system, user, model=None):
    model = model or DEFAULT_MODELS["openrouter"]
    return _openai_compatible_call(
        "https://openrouter.ai/api/v1", _ENV["OPEN_ROUTER_KEY"], model, system, user
    )


def call_nvidia_nim(system, user, model=None):
    model = model or DEFAULT_MODELS["nvidia_nim"]
    return _openai_compatible_call(
        "https://integrate.api.nvidia.com/v1", _ENV["NVIDIA_NIM_KEY"], model, system, user,
        timeout=PROVIDER_TIMEOUT_S.get("nvidia_nim"),
    )


def call_gemini(system, user, model=None):
    model = model or DEFAULT_MODELS["gemini"]
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"Content-Type": "application/json"},
        params={"key": _ENV["GEMINI_API_KEY"]},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 600},
        },
        timeout=TIMEOUT_S,
    )
    if resp.status_code != 200:
        raise LLMError(f"gemini {model}: HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"gemini {model}: unexpected response shape: {data}") from e


_CALLERS = {
    "groq": call_groq,
    "openrouter": call_openrouter,
    "gemini": call_gemini,
    "nvidia_nim": call_nvidia_nim,
}


def call(provider: str, system: str, user: str, model=None) -> str:
    return _CALLERS[provider](system, user, model)


def call_with_fallback(system: str, user: str, chain=None) -> tuple[str, str, float]:
    """Try providers in order, return (text, provider_used, latency_seconds) from the first
    one that succeeds. Raises LLMError if every provider in the chain fails."""
    chain = chain or PROVIDER_CHAIN
    errors = []
    for provider in chain:
        t0 = time.time()
        try:
            text = call(provider, system, user)
            return text, provider, time.time() - t0
        except Exception as e:  # noqa: BLE001 -- deliberately broad, this is a fallback loop
            errors.append(f"{provider}: {e}")
    raise LLMError("all providers failed: " + " | ".join(errors))
