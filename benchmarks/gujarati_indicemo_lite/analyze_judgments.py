"""Paired delivery-vs-neutral analysis of the Gujarati IndicEmo-lite judgments.

For each prompt: delta = mean judge score of the delivery clip - mean judge score
of its neutral twin (same text, same "Requested delivery: X" prompt). A raw
delivery score alone cannot show delivery adherence, because the judge is told
which emotion to look for and is lenient on upbeat TTS.

Run: python benchmarks/gujarati_indicemo_lite/analyze_judgments.py
"""
import json
import pathlib
import random
import statistics as st
from collections import defaultdict

BD = pathlib.Path(__file__).resolve().parent
J = BD / "judgments"
CATS = ["happy", "excited", "sad", "angry", "professional"]


def load(p):
    return [r for r in json.load(open(p, encoding="utf-8")) if r.get("ok") and r.get("score") is not None]


def per_prompt(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["id"]].append(float(r["score"]))
    return {k: st.mean(v) for k, v in d.items()}, d


deliv, deliv_all = per_prompt(load(J / "judgments_raw.json"))
neut, neut_all = per_prompt(load(J / "judgments_neutral_raw.json"))
cat = {}
for line in open(BD / "prompts.jsonl", encoding="utf-8"):
    if line.strip():
        p = json.loads(line)
        cat[p["id"]] = p["delivery"]

ids = sorted(set(deliv) & set(neut))
delta = {i: deliv[i] - neut[i] for i in ids}


def boot_ci(vals, n=10000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


print(f"prompts with both clips judged: {len(ids)}/24")
print(f"{'delivery':13s}{'n':>3s}{'delivery':>10s}{'neutral':>9s}{'delta':>8s}   95% CI (bootstrap)")
rows_out = {}
for c in CATS + ["ALL"]:
    sel = [i for i in ids if c == "ALL" or cat[i] == c]
    if not sel:
        continue
    d = [delta[i] for i in sel]
    lo, hi = boot_ci(d) if len(d) > 1 else (d[0], d[0])
    dm, nm = st.mean(deliv[i] for i in sel), st.mean(neut[i] for i in sel)
    print(f"{c:13s}{len(sel):>3d}{dm:>10.2f}{nm:>9.2f}{st.mean(d):>+8.2f}   [{lo:+.2f}, {hi:+.2f}]")
    rows_out[c] = {"n": len(sel), "delivery_mean": round(dm, 3), "neutral_mean": round(nm, 3),
                   "delta_mean": round(st.mean(d), 3), "ci95": [round(lo, 3), round(hi, 3)]}

# the judge's two "panelists" use the same model with different API keys: how much do they differ?
both = [(v[0], v[1]) for v in list(deliv_all.values()) + list(neut_all.values()) if len(v) >= 2]
agree = st.mean(abs(a - b) for a, b in both)
print(f"\ntwo judge keys (same model): mean |diff| = {agree:.2f} over {len(both)} clips")
json.dump({"paired": rows_out, "judge_key_mean_abs_diff": round(agree, 3),
           "note": "single judge model (gemini flash-lite) via 2 API keys; free-tier; preliminary"},
          open(J / "findings_paired.json", "w", encoding="utf-8"), indent=1)
