import json, glob, os, itertools, collections
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

import pathlib
B = str(pathlib.Path(__file__).resolve().parent)
ROOT = str(pathlib.Path(B).parents[1])
enc = VoiceEncoder("cpu")


def emb(path):
    return enc.embed_utterance(preprocess_wav(path))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def stats(v):
    v = np.array(v)
    return f"mean={v.mean():.3f} sd={v.std():.3f} min={v.min():.3f} max={v.max():.3f} n={len(v)}"


# ---- embeddings
E = {}
for line in open(B + r"\batch.jsonl", encoding="utf-8"):
    d = json.loads(line)
    n = d["name"]
    cat = "neutral" if n.endswith("_neutral") else d["description"].split(",")[0].strip()
    E[n] = (cat, emb(f"{B}\\audio\\{n}.wav"))
drift = {os.path.basename(p)[6:-4]: emb(p) for p in glob.glob(B + r"\drift_audio\*.wav")}

others = {}
for label, p in [("user_voice_sample", ROOT + r"\voice_sample.wav"),
                 ("user_input", ROOT + r"\input.wav"),
                 ("xtts_clone_reply", ROOT + r"\reply.wav")]:
    if os.path.exists(p):
        try:
            others[label] = emb(p)
        except Exception as e:
            print("skip", label, e)
for p in glob.glob(B + r"\xtts_audio\*.wav"):
    try:
        others["xtts:" + os.path.basename(p)[:-4]] = emb(p)
    except Exception as e:
        print("skip", p, e)

neutral = {n: e for n, (c, e) in E.items() if c == "neutral"}
deliv = {n: (c, e) for n, (c, e) in E.items() if c != "neutral"}

print("=== BASELINES (resemblyzer 256-d, cosine)")
# within-speaker: different sentences, both neutral, same voice (Ira)
within = [cos(a, b) for (na, a), (nb, b) in itertools.combinations(neutral.items(), 2)]
print("same voice, different sentence, both neutral:", stats(within))

# cross-speaker: Ira neutral vs any non-Ira reference voice
for k, v in others.items():
    s = [cos(v, e) for e in neutral.values()]
    print(f"Ira-neutral vs {k:28s}: {stats(s)}")

print("\n=== EMOTION vs NEUTRAL, same sentence (24 twin pairs)")
bycat = collections.defaultdict(list)
for n, (c, e) in deliv.items():
    twin = neutral[n.replace("_delivery", "_neutral")]
    bycat[c].append(cos(e, twin))
allv = []
for c in ["happy", "excited", "sad", "angry", "professional"]:
    print(f"{c:13s}", stats(bycat[c]))
    allv += bycat[c]
print("ALL          ", stats(allv))

print("\n=== EMOTION vs NEUTRAL centroid of OTHER sentences (leave-one-out)")
bycat2 = collections.defaultdict(list)
for n, (c, e) in deliv.items():
    twin_name = n.replace("_delivery", "_neutral")
    cen = np.mean([v for k, v in neutral.items() if k != twin_name], axis=0)
    bycat2[c].append(cos(e, cen))
loo_neutral = []
for k, v in neutral.items():
    cen = np.mean([x for kk, x in neutral.items() if kk != k], axis=0)
    loo_neutral.append(cos(v, cen))
print("neutral vs neutral centroid (LOO):", stats(loo_neutral))
for c in ["happy", "excited", "sad", "angry", "professional"]:
    print(f"{c:13s}", stats(bycat2[c]))

print("\n=== OLD DRIFT SENTENCE (same text, 6 conditions)")
if "neutral" in drift:
    for k, v in drift.items():
        if k != "neutral":
            print(f"neutral vs {k:13s} {cos(v, drift['neutral']):.3f}")

json.dump({
    "within_speaker_neutral_pairs": stats(within),
    "per_category_vs_twin": {c: stats(v) for c, v in bycat.items()},
    "per_category_vs_neutral_centroid": {c: stats(v) for c, v in bycat2.items()},
    "neutral_loo": stats(loo_neutral),
}, open(os.path.join(B, "drift_speaker_embedding_results.json"), "w"), indent=1)
