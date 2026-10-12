"""Control arm for the Gujarati IndicEmo-lite judgments.

judge_gemini.py only scored the delivery-conditioned clips, so the judge's
baseline leniency was unmeasured. This script scores each NEUTRAL twin with the
exact same rubric and the exact same "Requested delivery: X" prompt, and fills
in any delivery judgments that failed (quota) in the original run.

Output: judgments/judgments_neutral_raw.json  (+ judgments/judgments_raw.json patched)
Run:    python benchmarks/gujarati_indicemo_lite/judge_neutral_control.py
"""
import json
import os
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BD = pathlib.Path(__file__).resolve().parent
PROJECT = BD.parents[1]
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))

from llm import _gemini_clients, GEMINI_FAST_MODEL  # noqa: E402
from google.genai import types  # noqa: E402

AUDIO = BD / "audio"
OUTD = BD / "judgments"
JUDGE_MODEL = GEMINI_FAST_MODEL  # same model as the original run

prompts = {}
for line in open(BD / "prompts.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        p = json.loads(line)
        prompts[p["id"]] = p

clients = _gemini_clients()
print(f"gemini clients: {len(clients)} | judge model: {JUDGE_MODEL}", flush=True)
if not clients:
    sys.exit("no gemini keys configured")


def judge_clip(client, delivery, wav_path):
    # prompt text is byte-for-byte the one in judge_gemini.py
    b = open(wav_path, "rb").read()
    prompt = (
        f"You are judging expressive delivery in Gujarati speech (code-switched with English). "
        f"Requested delivery: {delivery.upper()}. Rate ONLY the expressive delivery quality "
        f"on this anchored 1-5 scale:\n"
        f"1 = flat/monotone, no requested delivery\n"
        f"2 = barely perceptible delivery\n"
        f"3 = recognizable but weak delivery\n"
        f"4 = clear, convincing delivery\n"
        f"5 = strong, sustained, natural performance\n"
        f"Axes: tone match, dynamics, phrasing, sustained expression. "
        f'Respond ONLY with JSON: {{"score": <1-5 float>, "note": "<max 10 words>"}}'
    )
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[types.Part.from_bytes(data=b, mime_type="audio/wav"), prompt],
        config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=100),
    )
    raw = (resp.text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()
    return json.loads(raw)


def run(pid, kind, ci, client):
    rec = {"id": pid, "delivery": prompts[pid]["delivery"], "judge": f"gemini_{ci}", "clip": kind}
    t0 = time.time()
    try:
        out = judge_clip(client, prompts[pid]["delivery"], AUDIO / f"{pid}_{kind}.wav")
        rec.update({"score": out.get("score"), "note": out.get("note", ""), "ok": True})
    except Exception as e:
        rec.update({"ok": False, "error": str(e)[:150]})
    rec["ms"] = round((time.time() - t0) * 1000)
    print(f"{pid} {kind:8s} judge{ci}: {rec.get('score')} {'OK' if rec['ok'] else rec.get('error','')[:60]}", flush=True)
    time.sleep(4)  # stay under the free-tier per-minute quota
    return rec


# 1) neutral twins, every prompt x every key
neutral_path = OUTD / "judgments_neutral_raw.json"
prev = json.load(open(neutral_path, encoding="utf-8")) if neutral_path.exists() else []
neutral = [r for r in prev if r.get("ok") and r.get("score") is not None]  # resume: keep good ones
done = {(r["id"], r["judge"]) for r in neutral}
for pid in sorted(prompts):
    if not (AUDIO / f"{pid}_neutral.wav").exists():
        continue
    for ci, c in enumerate(clients):
        if (pid, f"gemini_{ci}") in done:
            continue
        neutral.append(run(pid, "neutral", ci, c))
json.dump(neutral, open(OUTD / "judgments_neutral_raw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 2) back-fill failed delivery judgments
raw_path = OUTD / "judgments_raw.json"
delivery = json.load(open(raw_path, encoding="utf-8"))
have = {(r["id"], r["judge"]) for r in delivery if r.get("ok") and r.get("score") is not None}
todo = [(pid, ci) for pid in sorted(prompts) for ci in range(len(clients))
        if (pid, f"gemini_{ci}") not in have and (AUDIO / f"{pid}_delivery.wav").exists()]
print(f"back-filling {len(todo)} failed delivery judgments", flush=True)
for pid, ci in todo:
    rec = run(pid, "delivery", ci, clients[ci])
    if rec["ok"]:
        delivery = [r for r in delivery if not (r["id"] == pid and r["judge"] == f"gemini_{ci}")]
        rec.pop("clip", None)
        delivery.append(rec)
json.dump(delivery, open(raw_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("DONE", flush=True)
