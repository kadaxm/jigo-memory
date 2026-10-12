"""
Calibrate the recall answer gate (jigo_voice.ANSWER_GATE) on a labelled set.

Finding (2026-10): at the old threshold 0.45 the gate refused 0/40 unanswerable
questions - unrelated questions score 0.55-0.69 on raw embedding similarity
(1/(1+cosine distance)). On 52 answerable vs 40 unanswerable queries the raw
score alone has AUC 0.887; adding top-1/top-2 margin and query/memory word
overlap in a logistic gate did not help (CV AUC 0.886), so the gate stays a
single threshold. 'Hard' negatives (questions about stored entities asking for
an attribute that was never stored) are mostly NOT separable by similarity -
the grounded LLM in compose_reply has to catch those.

Prints the refuse/wrongly-refuse curve. Run: python gate_calibration.py
"""
import chromadb
import numpy as np
from sklearn.metrics import roc_auc_score

import ablation_memory as A

M = A.M
EASY_EXTRA = [
    "Which hospital was I born in?", "What is my PAN number?", "Who is my favourite singer?",
    "What is my mother's maiden name?", "Which city did I grow up in before college?",
    "What is my favourite dessert?", "Which social media app do I use the most?",
    "What is my shoe brand?", "How tall am I?", "Which festival do I enjoy the most?",
]
HARD = [  # mention stored entities but ask for an attribute that was never stored
    "What is my sister's age?", "Where does my brother work?", "What is Dr Mehta's phone number?",
    "What is the name of my guitar teacher?", "Which train do I take back home?",
    "What time does my bike service start?", "What is my grandfather's name?",
    "How much did the keyboard cost?", "What is my dentist's name?", "Which chai brand do I buy?",
]


def build_collection():
    A.configure(A.VARIANTS["V8 prod, rec weight .05"])
    col = chromadb.EphemeralClient().create_collection("gate_cal", metadata={"hnsw:space": "cosine"})
    for age, text, tag, payload in A.build_events():
        A._clock[0] = A.T0 - age * A.DAY
        M.add_memory(text, source="gate", col=col)
    A._clock[0] = A.T0
    return col


def main():
    col = build_collection()
    rows = ([(q, "answerable") for _, _, q, _ in A.FACTS]
            + [(u[5], "answerable") for u in A.UPDATES]
            + [(q, "easy") for q in A.ABSTAIN + EASY_EXTRA]
            + [(q, "hard") for q in HARD])
    raw = np.array([M.search_memory(q, top_k=8, associative=False, col=col)[0]["raw_similarity"] for q, _ in rows])
    kind = np.array([k for _, k in rows])
    y = (kind == "answerable").astype(int)
    print(f"n={len(y)} answerable={y.sum()} easy={(kind=='easy').sum()} hard={(kind=='hard').sum()}  AUC(raw)={roc_auc_score(y, raw):.3f}")
    print("answerable raw 5th pct:", round(float(np.percentile(raw[y == 1], 5)), 2))
    print(f"{'gate':>8} {'refuse easy':>12} {'refuse hard':>12} {'wrongly refused':>16}")
    for t in (0.45, 0.55, 0.60, 0.62, 0.63, 0.65, 0.68):
        p = raw >= t
        print(f"{t:>8.2f} {np.mean(~p[kind=='easy']):>12.0%} {np.mean(~p[kind=='hard']):>12.0%} {np.mean(~p[y==1]):>16.0%}")


if __name__ == "__main__":
    main()
