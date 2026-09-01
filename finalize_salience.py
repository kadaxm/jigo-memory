"""Finalize the salience model: fit Ridge on the full judged dataset and save
deployment artifact (salience_ridge.npz). Model selection was done via
compare_salience.py (5-seed CV: Ridge beat MLP on Pearson/Spearman/MAE).
Deployment inference clamps output to [0,1] (salience_model.py).
"""
import json
import os
import sys

import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.linear_model import Ridge


def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    def rank(x):
        order = np.argsort(x)
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x))
        return ranks
    return pearson(rank(a), rank(b))


def main():
    from memory import model

    with open(os.path.join(PROJECT, "salience_dataset.json"), encoding="utf-8") as f:
        data = json.load(f)
    items = data["judged"]
    texts = [x["text"] for x in items]
    ys = np.array([x["salience"] for x in items], dtype=np.float32)
    print(f"dataset: {len(items)}")

    E = np.array([model.encode(t) for t in texts], dtype=np.float64)

    rg = Ridge(alpha=1.0).fit(E, ys)

    # in-sample sanity (deployment metrics come from the 5-seed CV in compare_salience.py)
    p = rg.predict(E)
    print(f"in-sample pearson {pearson(p, ys):.3f} spearman {spearman(p, ys):.3f} "
          f"mae {float(np.abs(p - ys).mean()):.3f}")

    out = os.path.join(PROJECT, "salience_ridge.npz")
    np.savez(out, W=rg.coef_.astype(np.float64), b=np.float64(rg.intercept_))
    print(f"saved {out}")

    # sanity: clamp range check on real embeddings
    preds = np.clip(rg.predict(E), 0, 1)
    print(f"clamped pred range: [{preds.min():.3f}, {preds.max():.3f}]")


if __name__ == "__main__":
    main()
