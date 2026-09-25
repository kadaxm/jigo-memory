"""Speaker-drift audit: 6 drift clips (same text, 5 emotions + neutral anchor).
Speaker similarity via MFCC timbre vector + cosine (approximate, labeled).
Outputs: drift_matrix.csv + drift_findings.json"""
import itertools
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DIR = r"C:\Users\kadam\Documents\Jigo-memory\benchmarks\gujarati_indicemo_lite\drift_audio"
OUTD = r"C:\Users\kadam\Documents\Jigo-memory\benchmarks\gujarati_indicemo_lite"

import librosa  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["drift_neutral", "drift_happy", "drift_sad", "drift_angry", "drift_excited", "drift_professional"]
LABELS = {"drift_neutral": "neutral", "drift_happy": "happy", "drift_sad": "sad",
          "drift_angry": "angry", "drift_excited": "excited", "drift_professional": "professional"}


def timbre_vector(path):
    y, sr = librosa.load(path, sr=24000, mono=True)
    # trim silence edges
    idx = np.abs(y) > 0.01
    if idx.any():
        y = y[idx]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # robust stats per coefficient: mean + std -> fixed-size timbre vector
    v = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return v / (np.linalg.norm(v) + 1e-9)


vecs = {}
for name in ORDER:
    p = os.path.join(DIR, name + ".wav")
    vecs[name] = timbre_vector(p)
    print(f"{name}: vector dim {len(vecs[name])}", flush=True)

rows = []
pairs = list(itertools.combinations(ORDER, 2))
for a, b in pairs:
    sim = float(np.dot(vecs[a], vecs[b]))
    rows.append((LABELS[a], LABELS[b], sim))

# key comparisons vs the neutral anchor (same text, no delivery conditioning)
anchor_sims = {b: s for (a, b, s) in rows if a == "neutral"}
emo_sims = {k: v for k, v in anchor_sims.items() if k != "neutral"}
findings = {
    "method": "MFCC(20) mean+std timbre vector, cosine similarity (approximate speaker/timbre proxy)",
    "text": "identical Gujarati sentence across all conditions",
    "anchor_sims_vs_neutral": anchor_sims,
    "emotion_vs_neutral_mean": float(np.mean(list(emo_sims.values()))),
    "emotion_vs_neutral_min": float(np.min(list(emo_sims.values()))),
    "emotion_vs_neutral_max": float(np.max(list(emo_sims.values()))),
    "pairs": [{"a": a, "b": b, "cosine": round(s, 4)} for a, b, s in rows],
}
with open(os.path.join(OUTD, "drift_findings.json"), "w", encoding="utf-8") as f:
    json.dump(findings, f, ensure_ascii=False, indent=2)

print("\n=== similarity vs neutral anchor (same text) ===", flush=True)
for k, v in sorted(anchor_sims.items(), key=lambda x: -x[1]):
    print(f"  {k:14} {v:.4f}", flush=True)
print(f"\nmean across emotions: {findings['emotion_vs_neutral_mean']:.4f} "
      f"(min {findings['emotion_vs_neutral_min']:.4f}, max {findings['emotion_vs_neutral_max']:.4f})", flush=True)
print("DRIFT_AUDIT_DONE", flush=True)