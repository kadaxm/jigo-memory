"""Aggregate Gujarati IndicEmo-lite judgments: per-category means (median of
judges per clip), overall score, coverage. Outputs results CSV + findings JSON."""
import json
import os
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BD = r"C:\Users\kadam\Documents\Jigo-memory\benchmarks\gujarati_indicemo_lite"

prompts = {}
with open(os.path.join(BD, "prompts.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            p = json.loads(line)
            prompts[p["id"]] = p

with open(os.path.join(BD, "judgments", "judgments_raw.json"), encoding="utf-8") as f:
    raw = json.load(f)

ok = [r for r in raw if r.get("ok") and r.get("score") is not None]
by_clip = {}
for r in ok:
    by_clip.setdefault(r["id"], []).append(r["score"])

cat_scores = {}
for pid, scores in sorted(by_clip.items()):
    d = prompts[pid]["delivery"]
    med = statistics.median(scores)
    cat_scores.setdefault(d, []).append(med)

DELIVERIES = ["happy", "sad", "angry", "excited", "professional"]
cat_means = {d: statistics.mean(cat_scores[d]) for d in DELIVERIES if d in cat_scores}
emotion_only = [v for d, v in cat_means.items() if d != "professional"]
overall = statistics.mean([v for d in cat_scores for v in cat_scores[d]])

rows = []
for pid in sorted(by_clip):
    d = prompts[pid]["delivery"]
    scores = by_clip[pid]
    rows.append((pid, d, statistics.median(scores), len(scores)))
with open(os.path.join(BD, "judgments", "results.csv"), "w", encoding="utf-8") as f:
    f.write("id,delivery,median_score,n_judges\n")
    for pid, d, med, n in rows:
        f.write(f"{pid},{d},{med:.2f},{n}\n")

findings = {
    "n_prompts_with_scores": len(by_clip),
    "n_judge_calls_ok": len(ok),
    "n_judge_calls_total": len(raw),
    "overall_mean": round(overall, 2),
    "emotions_only_mean": round(statistics.mean(emotion_only), 2) if emotion_only else None,
    "category_means": {d: round(v, 2) for d, v in sorted(cat_means.items())},
    "note": "PRELIMINARY extension, substituted free-tier judge panel (not comparable to IndicEmo published scores)",
}
with open(os.path.join(BD, "judgments", "findings.json"), "w", encoding="utf-8") as f:
    json.dump(findings, f, ensure_ascii=False, indent=2)

print("=== GUJARATI INDICEMO-LITE — FINDINGS ===", flush=True)
print(f"overall: {overall:.2f}/5 (n={len(by_clip)} prompts)", flush=True)
print(f"emotions-only: {findings['emotions_only_mean']}/5", flush=True)
for d in DELIVERIES:
    if d in cat_means:
        print(f"  {d:14} {cat_means[d]:.2f}/5  (n={len(cat_scores[d])})", flush=True)
print(f"coverage: {findings['n_judge_calls_ok']}/{findings['n_judge_calls_total']} judge calls ok", flush=True)
print("AGGREGATE_DONE", flush=True)