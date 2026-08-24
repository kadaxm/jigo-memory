"""
Provider layer for every LLM text call in Jigo.

Chain: OpenRouter / Ox Alpha (primary) -> Gemini (key-rotating fallback).
memory.py, jigo_voice.py and web_ui.py all route through chat()/chat_json()
so the provider can be swapped in one place.
"""

import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "stealth/ox-alpha")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "models/gemini-3-flash-preview")
GEMINI_FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "models/gemini-3.5-flash-lite")

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

_GEMINI_KEY = os.getenv("GEMINI_API_KEY")


def _gemini_clients():
    """All configured Gemini clients (GEMINI_API_KEY, then _2, _3...)."""
    keys = []
    if _GEMINI_KEY:
        keys.append(_GEMINI_KEY)
    i = 2
    while True:
        k = os.getenv(f"GEMINI_API_KEY_{i}")
        if not k:
            break
        keys.append(k)
        i += 1
    return [genai.Client(api_key=k) for k in keys]


def _openrouter_chat(prompt, system=None, max_tokens=500, temperature=0.4, timeout=40):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = httpx.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"openrouter: {data['error']}")
    content = (data["choices"][0]["message"]["content"] or "").strip()
    if not content:
        # reasoning models can burn the whole token budget on hidden thinking
        # and return empty visible content — treat as a failure so the
        # fallback provider answers instead.
        raise RuntimeError("openrouter: empty content (reasoning exhausted tokens)")
    return content


def _gemini_chat(prompt, max_tokens=500, temperature=0.4, model=None):
    model = model or GEMINI_TEXT_MODEL
    last_err = None
    for client in _gemini_clients():
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return (response.text or "").strip()
        except Exception as e:
            msg = str(e)
            last_err = e
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                continue  # this key is out -> try the next
            raise
    if last_err:
        raise last_err
    raise RuntimeError("No Gemini API keys configured")


def fast_gemini_chat(prompt, max_tokens=600, temperature=0.4):
    """Gemini on the fast stable lite model — used when the primary provider
    returns low-quality output and speed matters."""
    return _gemini_chat(prompt, max_tokens, temperature, model=GEMINI_FAST_MODEL)


def chat(prompt, system=None, max_tokens=500, temperature=0.4, timeout=40):
    """Plain-text completion. OpenRouter primary, Gemini fallback.
    timeout: hard ceiling for the OpenRouter attempt (seconds) — on timeout the
    Gemini fallback answers instead, so a slow provider can never stall the loop."""
    errors = []
    if OPENROUTER_API_KEY:
        try:
            return _openrouter_chat(prompt, system, max_tokens, temperature, timeout)
        except Exception as e:
            errors.append(f"openrouter: {e}")
    try:
        return _gemini_chat(prompt, max_tokens, temperature)
    except Exception as e:
        errors.append(f"gemini: {e}")
    raise RuntimeError(" | ".join(errors))


def chat_json(prompt, system=None, max_tokens=500, temperature=0.2, timeout=40):
    """chat() that must return JSON. Tolerates ``` fences AND trailing junk
    after the first JSON object (some models append commentary)."""
    raw = chat(prompt, system, max_tokens, temperature, timeout)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        dec = json.JSONDecoder()
        for i, ch in enumerate(raw):
            if ch in "{[":
                obj, _ = dec.raw_decode(raw[i:])
                return obj
        raise
