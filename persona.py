"""Persona reply modes for Jigo's voice assistant.

masala uncle (default ON for sad/stressed users): warm Gujarati optimist.
Signature phrases ground the register; the LLM improvises around them,
grounded only in retrieved memories. Romanized Gujarati + light English
code-mixing (Hinglish register), so the reply flows through the existing
TTS ladder (XTTS clone primary, ElevenLabs fallback).

roast mode (OFF by default, comedy toggle): user shares good news or sounds
happy/excited -> playful theatrical dismissal. Hinglish register, 1-2
sentences, comedic exaggeration, never genuinely cruel.

Auto-trigger: masala uncle fires on detected user emotion only (emotion.py
label sad/stressed); roast fires on happy/excited when the toggle is on.
Both are reply-composition modes — the live TTS ladder renders them.
"""
import os
import random

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

_PERSONA_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONA_LINES_PATH = os.path.join(_PERSONA_DIR, "persona_lines.txt")

PERSONA_MODE = os.getenv("JIGO_PERSONA", "1") not in ("0", "false", "False")
ROAST_MODE = os.getenv("JIGO_ROAST", "0") in ("1", "true", "True")

# uncle mode fires on these detected user-emotion labels
UNCLE_TRIGGERS = ("sad", "stressed")
# roast mode fires on these (only when ROAST_MODE)
ROAST_TRIGGERS = ("happy", "excited")

import llm  # noqa: E402  (project-local provider layer)


def _load_persona_lines():
    lines = []
    try:
        with open(PERSONA_LINES_PATH, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(line)
    except OSError:
        pass
    return [l for l in lines if l and not l.isupper()]


def compose_persona_reply(query, results, emo_label=None):
    """Uncle-mode reply: warm Gujarati optimist, grounded only in memories.

    Signature phrases anchor the register; the LLM improvises 1-2 short
    spoken sentences in romanized Gujarati + light English. Falls back to a
    random signature line when the LLM is unavailable or returns junk.
    """
    sig = _load_persona_lines()
    sig_block = "\n".join(f"- {l}" for l in sig) if sig else "- Thai jase bhai, evu to thatu rhe lya."
    mem_lines = "\n".join(f"- {_clean(r['content'])[:280]}" for r in (results or [])[:8])
    user_mood = emo_label or "sad"
    prompt = (
        "You are Jigo in 'masala uncle' mode: a warm middle-aged Gujarati optimist "
        "who has heard it all and is never fazed. The user just said:\n"
        f"\"{query}\"\n"
        f"The user's detected mood: {user_mood}.\n\n"
        "Your signature phrases (style anchors, romanized Gujarati):\n"
        + sig_block + "\n\n"
        + (("Relevant memories of the user:\n" + mem_lines + "\n\n") if mem_lines.strip() else "")
        + "Rules:\n"
        "1. Reply in 1-2 short spoken sentences, romanized Gujarati with light\n"
        "   English code-mixing (Hinglish register), like the signature phrases.\n"
        "2. Practical, warm, completely unfazed — dissolve the worry, never mock.\n"
        "3. If memories are relevant, weave one specific detail in naturally.\n"
        "4. Never quote memory titles, markdown, or speak English-only.\n"
        "5. Never use the phrase '" + _refusal() + "'.\n"
    )
    try:
        out = _clean(llm.fast_gemini_chat(prompt, max_tokens=220, temperature=0.5))
        if out and not out.isupper():
            return out
    except Exception as e:
        print(f"[persona: fast path failed ({str(e)[:80]})]")
    try:
        out = _clean(llm.chat(
            prompt,
            system="1-2 short spoken sentences, romanized Gujarati optimist register. "
                   "Never quote memory titles or markdown.",
            max_tokens=220,
            temperature=0.5,
            timeout=14,
        ))
        if out and not out.isupper():
            return out
    except Exception as e:
        print(f"[persona: retry failed ({str(e)[:80]})]")
    if sig:
        return random.choice(sig)
    return "Thai jase bhai, evu to thatu rhe lya."


def compose_roast_reply(query, results, emo_label=None):
    """Roast-mode reply: playful theatrical demotivator.

    Fires only when the user is happy/excited and ROAST_MODE is on.
    Hinglish register, 1-2 short sentences, comedic exaggeration.
    """
    mem_lines = "\n".join(f"- {_clean(r['content'])[:280]}" for r in (results or [])[:8])
    user_mood = emo_label or "happy"
    prompt = (
        "You are Jigo in 'demotivator' mode: a theatrical, dramatic pessimist who "
        "delivers playful roasts. The user just said:\n"
        f"\"{query}\"\n"
        f"The user's detected mood: {user_mood}.\n\n"
        + (("Context memories:\n" + mem_lines + "\n\n") if mem_lines.strip() else "")
        + "Rules:\n"
        "1. Reply in 1-2 short spoken sentences, Hinglish register "
        "(romanized Hindi/Gujarati with English code-mixing).\n"
        "2. Playful roast: theatrical disappointment, mock sighs, exaggerated\n"
        "   pessimism. Comedy, not cruelty — clearly a joke.\n"
        "3. End with one soft word that betrays you secretly care.\n"
        "4. Never quote memory titles or markdown.\n"
    )
    try:
        out = _clean(llm.fast_gemini_chat(prompt, max_tokens=220, temperature=0.7))
        if out and not out.isupper():
            return out
    except Exception as e:
        print(f"[roast: fast path failed ({str(e)[:80]})]")
    try:
        out = _clean(llm.chat(
            prompt,
            system="1-2 short spoken sentences, playful Hinglish roast register. "
                   "Never quote memory titles or markdown.",
            max_tokens=220,
            temperature=0.7,
            timeout=14,
        ))
        if out and not out.isupper():
            return out
    except Exception as e:
        print(f"[roast: retry failed ({str(e)[:80]})]")
    return "Haan haan, bada aaya duniya badalne. <chuckle> Chal theek hai, khush."


def _clean(text):
    """Shared cleanup so persona replies never echo markdown or titles."""
    import re
    s = re.sub(r"[*_#`>]+", "", text or "")
    return " ".join(s.split())


def _refusal():
    """Avoid a circular import from jigo_voice.REFUSAL."""
    return "I don't have anything on that yet."


def pick_mode(emo_label):
    """Return 'persona' | 'roast' | None for a detected user-emotion label."""
    label = (emo_label or "").lower()
    if PERSONA_MODE and label in UNCLE_TRIGGERS:
        return "persona"
    if ROAST_MODE and label in ROAST_TRIGGERS:
        return "roast"
    return None
