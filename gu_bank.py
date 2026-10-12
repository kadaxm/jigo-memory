"""
Gujarati voice-line bank rendered with rumik-oss-1 (offline, once).

rumik-oss-1 speaks Gujarati natively but runs ~0.1x realtime on a 4 GB GPU, so it
cannot answer live. The compromise that keeps it audible in the product: a bank of
short comfort / celebration lines is rendered ahead of time (render_gu_bank.py), and
the dashboard plays the matching line instantly when the user's statement carries
an emotion. XTTS-v2 (the voice clone) has no Gujarati at all.

Lines live in gu_bank/lines.jsonl (Gujarati script + English gloss, draft - edit
freely, then re-render). Audio files are gu_bank/audio/<name>.wav.
"""
import json
import os
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent / "gu_bank"
AUDIO = ROOT / "audio"
ENABLED = os.getenv("JIGO_GU_BANK", "1") not in ("0", "false", "False")
_last = {}


def _verified():
    """name -> ok, from verify_gu_bank.py (Scribe heard it as Gujarati). Missing file = trust all."""
    p = ROOT / "verified.json"
    if not p.exists():
        return None
    try:
        return {k: bool(v.get("ok")) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    except Exception:
        return None


def _lines():
    ver = _verified()
    out = []
    p = ROOT / "lines.jsonl"
    if not p.exists():
        return out
    for raw in p.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            d = json.loads(raw)
            d["available"] = (AUDIO / f"{d['name']}.wav").exists() and (ver is None or ver.get(d["name"], False))
            out.append(d)
    return out


def audio_path(line_id):
    p = AUDIO / f"{line_id}.wav"
    return p if p.exists() else None


def pick(emotion):
    """Best rendered line for an emotion label (sad/stressed/angry/happy/excited);
    falls back to the 'general' group. None when nothing is rendered yet."""
    if not ENABLED:
        return None
    lines = [l for l in _lines() if l["available"]]
    pool = [l for l in lines if l["emotion"] == emotion] or [l for l in lines if l["emotion"] == "general"]
    if not pool:
        return None
    fresh = [l for l in pool if l["name"] != _last.get(emotion)] or pool
    choice = random.choice(fresh)
    _last[emotion] = choice["name"]
    return {
        "id": choice["name"],
        "text": choice["text"],
        "gloss": choice.get("gloss", ""),
        "emotion": choice["emotion"],
        "engine": "rumik-oss-1",
        "audio_url": f"/gu_audio/{choice['name']}",
    }


def status():
    ls = _lines()
    return {"enabled": ENABLED, "lines": len(ls), "rendered": sum(1 for l in ls if l["available"])}
